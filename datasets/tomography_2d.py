"""2D Tomography dataset for inverse problems benchmarking.

This dataset uses the Shepp-Logan phantom and creates multiple tomography
operators with different angle ranges for distributed reconstruction.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import requests
import torch
from benchopt import BaseDataset, config
from deepinv.distributed import DistributedContext
from deepinv.physics import GaussianNoise, Tomography
from deepinv.utils.demo import get_image_url
from PIL import Image
from torchvision import transforms

from benchmark_utils import save_measurements_figure


def load_shepp_logan_image(
    img_size: int = 256,
    grayscale: bool = True,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """Load Shepp-Logan phantom image with caching.

    Downloads the original image once and caches it, then applies resizing transform.
    This avoids storing multiple copies of the same image at different sizes.

    Parameters
    ----------
    img_size : int
        Size of the image (height and width).
    grayscale : bool
        Whether to convert to grayscale.
    device : str
        Device to load the image on.
    cache_dir : Path, optional
        Directory to cache the original image.

    Returns
    -------
    torch.Tensor
        Shepp-Logan phantom image of shape (1, 1, H, W) or (1, 3, H, W).
    """
    # Setup cache for original image
    if cache_dir is None:
        cache_dir = Path(".")
    cache_dir.mkdir(parents=True, exist_ok=True)

    original_cache_file = cache_dir / "shepp_logan_original.pt"

    # Load or download original image
    if original_cache_file.exists():
        # Load cached original image
        img_tensor = torch.load(
            original_cache_file, map_location="cpu", weights_only=True
        )
    else:
        # Download and cache original image
        url = get_image_url("SheppLogan.png")
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Convert to tensor without resizing
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale())
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        img_tensor = transform(img).unsqueeze(0)

        # Cache the original image
        torch.save(img_tensor, original_cache_file)

    # Apply resizing transform
    if img_size is not None:
        resize_transform = transforms.Resize(img_size)
        img_tensor = resize_transform(img_tensor)

    # Move to device
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    return img_tensor


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "tomography_2d"
    requirements = [
        "torch",
        "numpy",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
    ]

    parameters = {
        "img_size": [512],
        "num_operators": [1],
        "num_angles": [100],
        "noise_level": [0.01],
        "circle": [True],
        "seed": [42],
    }

    def __init__(
        self,
        img_size=256,
        num_operators=2,
        num_angles=180,
        noise_level=0.05,
        circle=True,
        seed=42,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        img_size : int
            Size of the Shepp-Logan phantom image.
        num_operators : int
            Number of tomography operators (angle splits).
        num_angles : int
            Total number of projection angles.
        noise_level : float
            Noise level (sigma) for Gaussian noise.
        circle : bool
            Whether to use circular mask in tomography.
        seed : int
            Random seed for reproducibility.
        """
        self.img_size = img_size
        self.num_operators = num_operators
        self.num_angles = num_angles
        self.noise_level = noise_level
        self.circle = circle
        self.seed = seed

    def _create_operator_factory(
        self, device: str, angles_list: list, noise_sigma: float
    ):
        """Create a factory function for tomography operators.

        This factory is needed because tomography operators must be instantiated
        on the correct device in distributed settings.

        Parameters
        ----------
        device : str
            Device string for the reference (not used in factory).
        angles_list : list
            List of angle tensors, one for each operator.
        noise_sigma : float
            Noise level for Gaussian noise.

        Returns
        -------
        callable
            Factory function(index, device, shared) -> Tomography operator.
        """

        def factory(index: int, device: torch.device, shared: Optional[Dict] = None):
            """Create a tomography operator for the given index and device.

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
            Tomography
                Tomography operator for the given angle range.
            """
            angles = angles_list[index].to(device)

            # Create noise model with reproducible seed
            rng = torch.Generator(device=device).manual_seed(self.seed + index)
            noise_model = GaussianNoise(sigma=noise_sigma, rng=rng)

            # Create tomography operator
            physics = Tomography(
                img_width=self.img_size,
                angles=angles,
                circle=self.circle,
                device=device,
                noise_model=noise_model,
                normalize=False,
            )

            return physics

        return factory

    def get_data(self):
        """Load the data for this Dataset.

        Creates tomography operators factory and measurements.
        Returns dictionary with keys expected by Objective.set_data().
        """

        # Check if distributed environment is already set up
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            # Try to initialize
            try:
                import submitit

                submitit.helpers.TorchDistributedEnvironment().export(
                    set_cuda_visible_devices=False
                )
                print("Initialized distributed environment via submitit in dataset")
            except ImportError:
                print(
                    "submitit not installed, dataset will run in non-distributed mode"
                )
            except RuntimeError as e:
                # This could be SLURM not available or other runtime issues
                error_msg = str(e).lower()
                if "slurm" in error_msg or "environment" in error_msg:
                    print(f"SLURM environment not available in dataset: {e}")
                else:
                    print(f"RuntimeError initializing submitit in dataset: {e}")
        else:
            print(
                f"Distributed environment already initialized in dataset: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
            )

        # Use cleanup=False to keep process group alive for solver
        # Solver will handle cleanup when it's done
        with DistributedContext(seed=42, cleanup=False) as ctx:
            print(f"DistributedContext: rank {ctx.rank} / {ctx.world_size}")

            # Setup device
            device = ctx.device

            # Setup caching
            data_path = config.get_data_path(key="tomography_2d")
            cache_dir = Path(data_path)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Load Shepp-Logan phantom (caches original image, resizes as needed)
            ground_truth = load_shepp_logan_image(
                img_size=self.img_size,
                grayscale=True,
                device=str(device),
                cache_dir=cache_dir,
            )

            print(f"Loaded Shepp-Logan phantom: {ground_truth.shape}")

            # Create angle ranges for each operator
            # For parallel beam, angles typically go from 0 to 180 degrees
            angles_total = torch.linspace(
                0, 180, self.num_angles + 1, dtype=torch.float32, device=device
            )[:-1]

            # Split angles among operators
            angles_per_operator = self.num_angles // self.num_operators
            angles_list = []

            for i in range(self.num_operators):
                start_idx = i * angles_per_operator
                if i == self.num_operators - 1:
                    # Last operator gets any remaining angles
                    end_idx = self.num_angles
                else:
                    end_idx = (i + 1) * angles_per_operator

                # Extract angle subset
                angles_subset = angles_total[start_idx:end_idx]
                angles_list.append(angles_subset)

            print(f"Split {self.num_angles} angles into {self.num_operators} operators")

            # Create full operator on GPU to generate measurements
            # We need all angles to generate realistic measurements
            full_angles = angles_total
            rng_full = torch.Generator(device=device).manual_seed(self.seed)
            noise_model_full = GaussianNoise(sigma=self.noise_level, rng=rng_full)

            full_physics = Tomography(
                img_width=self.img_size,
                angles=full_angles,
                circle=self.circle,
                device=device,
                noise_model=noise_model_full,
                normalize=False,
            )

            # Generate full measurements
            with torch.no_grad():
                full_measurement = full_physics(ground_truth)

            print(f"Generated full measurement: {full_measurement.shape}")

            # Split measurements according to angle division
            # Each measurement subset corresponds to an operator's angle range
            # Full measurement shape is (B, C, detector_width, num_angles) = (1, 1, 128, 90)
            # We need to split along the last dimension (angles)
            measurement_list = []

            for i in range(self.num_operators):
                start_idx = i * angles_per_operator
                if i == self.num_operators - 1:
                    end_idx = self.num_angles
                else:
                    end_idx = (i + 1) * angles_per_operator

                # Extract measurement subset along the angle dimension (dimension 3)
                meas_subset = full_measurement[:, :, :, start_idx:end_idx]
                measurement_list.append(meas_subset)
                print(
                    f"Measurement {i}: shape {meas_subset.shape}, angles [{start_idx}:{end_idx}]"
                )

            # Ensure data is on correct device
            ground_truth = ground_truth.to(device)
            measurement_list = [m.to(device) for m in measurement_list]

            # Create factory function for operators
            physics_factory = self._create_operator_factory(
                device=str(device),
                angles_list=angles_list,
                noise_sigma=self.noise_level,
            )

            if ctx.rank == 0:
                # Save debug visualization (show sinograms)
                save_measurements_figure(
                    ground_truth,
                    measurement_list,
                    filename="tomography_2d_measurements.png",
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
