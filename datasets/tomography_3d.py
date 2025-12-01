"""3D Tomography dataset for inverse problems benchmarking.

This dataset uses 3D CT data from HuggingFace (Walnut cone-beam CT) and creates
multiple tomography operators with ASTRA for distributed reconstruction.
"""
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from deepinv.physics import TomographyWithAstra
from deepinv.utils.demo import load_torch_url
from deepinv.distrib import DistributedContext

from benchopt import BaseDataset
from benchopt import config


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = 'tomography_3d'

    parameters = {
        'num_operators': [4],
        'num_projections': [100],
        'geometry_type': ['conebeam'],
        'use_dataset_sinogram': [True],
        'seed': [42],
    }

    def __init__(self, num_operators=2, num_projections=100, 
                 geometry_type='conebeam',
                 use_dataset_sinogram=True, seed=42):
        """Initialize the dataset.
        
        Parameters
        ----------
        num_operators : int
            Number of tomography operators (angle splits).
        num_projections : int
            Number of projection angles to use.
        geometry_type : str
            Geometry type for ASTRA ('conebeam' or 'parallel3d').
        use_dataset_sinogram : bool
            If True, use sinogram from dataset. If False, generate via forward pass.
        seed : int
            Random seed for reproducibility.
        """
        self.num_operators = num_operators
        self.num_projections = num_projections
        self.geometry_type = geometry_type
        self.use_dataset_sinogram = use_dataset_sinogram
        self.seed = seed

    def _load_or_download_dataset(self, data_dir: Path, filename: str = "Walnut-CBCT_8.pt") -> Dict[str, torch.Tensor]:
        """Load 3D CT dataset from cache or download from HuggingFace.
        
        Parameters
        ----------
        data_dir : Path
            Directory to store cached data.
        filename : str
            Name of the cached file.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'dense_reconstruction': Ground truth 3D volume
            - 'sinogram': Measurement sinogram
            - 'vecs': Geometry vectors for cone-beam setup
        """
        cache_path = data_dir / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"Loading cached 3D CT data from {cache_path}")
            dataset = torch.load(cache_path)
        else:
            print(f"Downloading 3D CT data from HuggingFace...")
            url = "https://huggingface.co/datasets/romainvo/ct_examples/resolve/main/Walnut-CBCT_8.pt"
            dataset = load_torch_url(url)
            print(f"Saving 3D CT data to {cache_path}")
            torch.save(dataset, cache_path)

        return dataset

    def _create_operator_factory(self, dataset: Dict, img_shape: tuple, device_str: str):
        """Create a factory function for 3D tomography operators.
        
        ASTRA operators must be instantiated on the correct device,
        so we return a factory that creates them on demand.
        
        Parameters
        ----------
        dataset : dict
            Dataset dictionary with trajectory vectors.
        img_shape : tuple
            Shape of the ground truth volume (D, H, W).
        device_str : str
            Device string for reference (not used in factory).
            
        Returns
        -------
        callable
            Factory function(index, device, shared) -> TomographyWithAstra operator.
        """
        trajectory = dataset["vecs"]
        
        # Subsample trajectory if needed
        num_angles_total = trajectory.shape[0]
        if self.num_projections < num_angles_total:
            sparse_indexes = torch.linspace(
                0, num_angles_total - 1, 
                steps=self.num_projections, 
                dtype=torch.long
            )
            trajectory = trajectory[sparse_indexes]
            print(f"Subsampling trajectory: using {self.num_projections} angles out of {num_angles_total} available")
        
        num_measurement_angles = trajectory.shape[0]
        angles_per_operator = num_measurement_angles // self.num_operators
        
        # Detector size from dataset (typical for Walnut dataset)
        n_detector_pixels = (972, 768)  # (horizontal, vertical)
        object_spacing = (0.1, 0.1, 0.1)  # mm
        
        def factory(index: int, device: torch.device, shared: Optional[Dict] = None):
            """Create a 3D tomography operator for the given index and device.
            
            Parameters
            ----------
            index : int
                Operator index (0 to num_operators-1).
            device : torch.device
                Device to create the operator on.
            shared : dict, optional
                Shared data dictionary (not used here).
                
            Returns
            -------
            TomographyWithAstra
                3D tomography operator for the given angle range.
            """
            start_idx = index * angles_per_operator
            if index == self.num_operators - 1:
                # Last operator takes remaining angles
                end_idx = num_measurement_angles
            else:
                end_idx = (index + 1) * angles_per_operator
            
            trajectory_subset = trajectory[start_idx:end_idx]
            num_angles_subset = end_idx - start_idx
            
            print(f"Physics factory: creating operator {index} with {num_angles_subset} angles [{start_idx}:{end_idx}]")
            
            physics = TomographyWithAstra(
                img_size=img_shape,
                num_angles=num_angles_subset,
                num_detectors=n_detector_pixels,
                object_spacing=object_spacing,
                geometry_type=self.geometry_type,
                geometry_vectors=trajectory_subset,
                normalize=False,
                device=device,
            )
            
            # Store metadata for debugging
            physics._angle_range = (start_idx, end_idx)
            physics._operator_idx = index
            
            return physics
        
        return factory

    def _create_measurements_factory(self, dataset: Dict, device_str: str):
        """Create a factory function for measurements.
        
        This factory splits the full sinogram according to angle ranges.
        
        ASTRA Shape Convention:
        The ASTRA backend expects input with shape:
            (batch, channels, detector_h, angles, detector_v)
        
        Parameters
        ----------
        dataset : dict
            Dataset dictionary with sinogram.
        device_str : str
            Device string for reference.
            
        Returns
        -------
        callable
            Factory function(index, device, shared) -> torch.Tensor measurement.
        """
        if self.use_dataset_sinogram:
            sinogram = dataset["sinogram"]
            
            print(f"\n=== Loading Sinogram from Dataset ===")
            print(f"Original sinogram shape: {sinogram.shape}")
            
            # Subsample sinogram if needed
            num_angles_total = sinogram.shape[0]
            if self.num_projections < num_angles_total:
                sparse_indexes = torch.linspace(
                    0, num_angles_total - 1, 
                    steps=self.num_projections, 
                    dtype=torch.long
                )
                sinogram = sinogram[sparse_indexes]
                print(f"Subsampling sinogram: using {self.num_projections} angles")
            
            # Convert to tensor and permute to ASTRA format
            # Original: (angles, detector_h, detector_v)
            # Target: (batch, channels, detector_h, angles, detector_v)
            device = torch.device(device_str)
            sinogram_tensor = sinogram.float().to(device)
            sinogram_tensor = sinogram_tensor.permute(1, 0, 2)  # (h, angles, v)
            sinogram_tensor = sinogram_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h, angles, v)
            
            print(f"Final sinogram tensor shape: {sinogram_tensor.shape}")
            print(f"  Format: (batch, channels, detector_h, angles, detector_v)")
            print(f"=== Sinogram Loading Complete ===\n")
            
            num_angles = sinogram_tensor.shape[3]
            angles_per_op = num_angles // self.num_operators
            
            def factory(index: int, device: torch.device, shared: Optional[Dict] = None):
                """Return a slice of the dataset sinogram.
                
                Parameters
                ----------
                index : int
                    Measurement index.
                device : torch.device
                    Device to place the measurement on.
                shared : dict, optional
                    Shared data dictionary (not used here).
                    
                Returns
                -------
                torch.Tensor
                    Measurement tensor in ASTRA format.
                """
                start_idx = index * angles_per_op
                if index == self.num_operators - 1:
                    end_idx = num_angles
                else:
                    end_idx = (index + 1) * angles_per_op
                
                # Slice along the angles dimension (dim 3)
                measurement = sinogram_tensor[:, :, :, start_idx:end_idx, :].to(device)
                print(f"Measurements factory: operator {index} gets angles [{start_idx}:{end_idx}], shape {measurement.shape}")
                
                return measurement
            
            return factory
        else:
            raise NotImplementedError("Forward pass generation not yet implemented for benchopt datasets")

    def get_data(self):
        """Load the data for this Dataset.

        Creates 3D tomography operators factory and measurements.
        Returns dictionary with keys expected by Objective.set_data().
        """
        # Check if distributed environment is already set up
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            # Try to initialize
            try:
                import submitit
                submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
                print("Initialized distributed environment via submitit in dataset")
            except ImportError:
                print("submitit not installed, dataset will run in non-distributed mode")
            except RuntimeError as e:
                # This could be SLURM not available or other runtime issues
                error_msg = str(e).lower()
                if "slurm" in error_msg or "environment" in error_msg:
                    print(f"SLURM environment not available in dataset: {e}")
                else:
                    print(f"RuntimeError initializing submitit in dataset: {e}")
        else:
            print(f"Distributed environment already initialized in dataset: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

        # Use cleanup=False to keep process group alive for solver
        # Solver will handle cleanup when it's done
        with DistributedContext(seed=42, cleanup=False) as ctx:
            print(f"DistributedContext: rank {ctx.rank} / {ctx.world_size}")

            # Setup device
            device = ctx.device
            
            # Get data directory
            data_path = config.get_data_path(key="tomography_3d")
            

            dataset = self._load_or_download_dataset(
                data_dir=Path(data_path),
                filename="Walnut-CBCT_8.pt"
            )

            # Extract ground truth volume
            ref_rc = dataset["dense_reconstruction"]
            ground_truth = ref_rc.float().to(device)
            
            # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
            if ground_truth.dim() == 3:
                ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)
            
            img_shape = ground_truth.shape[-3:]  # (D, H, W)
            
            print(f"Loaded 3D ground truth with shape: {ground_truth.shape}")
            print(f"Image shape (D, H, W): {img_shape}")
        
            # Create measurements factory
            measurements_factory = self._create_measurements_factory(
                dataset=dataset,
                device_str=str(device)
            )
            
            # For benchopt, we need to return a list of measurements
            # Create them by calling the factory for each operator
            measurement_list = []
            for i in range(self.num_operators):
                meas = measurements_factory(i, device, None)
                measurement_list.append(meas)
            
            print(f"Created {len(measurement_list)} measurements")
            
            # Ensure data is on correct device
            ground_truth = ground_truth.to(device)
            measurement_list = [m.to(device) for m in measurement_list]
            
            # Create operator factory (always needed, doesn't use cached data)
            physics_factory = self._create_operator_factory(
                dataset=dataset,
                img_shape=img_shape,
                device_str=str(device)
            )
        
        return dict(
            ground_truth=ground_truth,
            measurement=measurement_list,
            physics=physics_factory,
            min_pixel=ground_truth.min().item(),
            max_pixel=ground_truth.max().item(),
            ground_truth_shape=ground_truth.shape,
            num_operators=self.num_operators,
        )