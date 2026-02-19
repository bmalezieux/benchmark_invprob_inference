"""Radio interferometry dataset using Karabo MeerKAT simulation.

This dataset simulates MeerKAT observations of real images (converted to SkyModels).
It uses pre-generated data from benchopt install.
"""
import torch
import numpy as np
import os
from pathlib import Path

from benchopt import BaseDataset
from benchopt import config
from benchmark_utils.radio_utils import get_meerkat_visibilities_path


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = 'radio_interferometry'
    
    install_cmd = 'shell'
    install_script = 'install_radio.sh'

    parameters = {
        'image_size': [256],
        'noise_level': [0.1],
        'seed': [42],
    }

    @classmethod
    def is_installed(cls, env_name=None, quiet=True, **kwargs):
        # 1. Check if module can be imported (dependencies present)
        try: 
            import astropy
            import casacore
            import deepinv
            import torchkbnufft
            from deepinv.distributed import DistributedContext

        except ImportError:
            return False
        
        # 2. Check if data is present
        repo_root = Path(__file__).parent.parent
        ms_cache_dir = repo_root / "data" / "radio_interferometry" / "meerkat_cache"
        
        if not ms_cache_dir.exists() or not any(ms_cache_dir.iterdir()):
            return False

        return True

    def __init__(self, image_size=256, noise_level=0.1, seed=42):
        """Initialize the dataset."""
        self.image_size = image_size
        self.noise_level = noise_level
        self.seed = seed

    def get_data(self):
        """Load the data for this Dataset.

        Generates visibilities using MeerKAT simulation and creates 
        RadioInterferometry physics operator.
        """
        from deepinv.physics import GaussianNoise
        from deepinv.distributed import DistributedContext
        from benchmark_utils import load_cached_example
        from benchmark_utils.deepinv_imager import DeepinvDirtyImager, DirtyImagerConfig

        # Check if distributed environment is already set up
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
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

        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            device = ctx.device
            
            # Use specific data path for caching
            data_path = Path(config.get_data_path(key="radio_interferometry"))
            data_path.mkdir(parents=True, exist_ok=True)

            # Cache directory for MS files
            ms_cache_dir = data_path / "meerkat_cache"
            gt_path = ms_cache_dir / f"image_{self.image_size}.npy"
            
            if gt_path.exists():
                 img_np = np.load(gt_path)
                 img = torch.from_numpy(img_np)
            else:
                # Ensure file is downloaded
                load_cached_example(
                    "m1_n.fits",
                    cache_dir=data_path, 
                    grayscale=True,
                    device="cpu",
                )
                
                # Fallback: Load and resize using shared utility
                from benchmark_utils.radio_utils import load_and_resize_image
                fits_path = data_path / "m1_n.fits"
                img_np = load_and_resize_image(fits_path, self.image_size)
                
                img = torch.from_numpy(img_np)
                img = torch.clamp(img, 0, 1)

            # Ensure (1, C, H, W)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            
            ground_truth = img.to(device)
            _, _, h, w = ground_truth.shape

            # Simulation parameters
            pixel_size_arcsec = 1.0

            # Get path to visibilities
            ms_path = get_meerkat_visibilities_path(
                img, # This must match the image used for generation! 
                     # If we resize here, we must use resized image for hash.
                ms_cache_dir,
                pixel_size_arcsec=pixel_size_arcsec,
                freq_hz=1e9,
                obs_duration=600
            )

            if not ms_path.exists():
                raise FileNotFoundError(
                    f"Measurement Set file not found at {ms_path}. "
                    "Please run 'benchopt install -d radio_interferometry' to generate the data."
                )
            
            # Create Physics Operator
            imager_config = DirtyImagerConfig(
                imaging_npixel=self.image_size,
                imaging_cellsize=np.deg2rad(pixel_size_arcsec / 3600.0),
                combine_across_frequencies=False
            )
            
            imager = DeepinvDirtyImager(imager_config, device=device)
            
            # create_deepinv_physics loads the MS and builds the operator
            physics, measurements = imager.create_deepinv_physics(
                visibility_path=str(ms_path),
                visibility_format="MS",
                visibility_column="DATA"
            )
            
            # measurements come from simulation (clean or with Karabo noise)
            # Add explicit noise for benchmarking control
            if self.noise_level > 0:
                physics.noise_model = GaussianNoise(sigma=self.noise_level)
                measurements = physics.noise_model(measurements)
            
            return dict(
                ground_truth=ground_truth,
                measurement=measurements,
                physics=physics,
                min_pixel=0.0,
                max_pixel=1.0,
                ground_truth_shape=ground_truth.shape,
            )
