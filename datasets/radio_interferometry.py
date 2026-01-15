"""Radio interferometry dataset using Karabo MeerKAT simulation.

This dataset simulates MeerKAT observations of real images (converted to SkyModels).
It uses Karabo-Pipeline for simulation and deepinv for the physics operator.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from deepinv.physics import GaussianNoise
from deepinv.distributed import DistributedContext

from benchopt import BaseDataset
from benchopt import config
from benchmark_utils import load_cached_example
from benchmark_utils.radio_utils import get_meerkat_visibilities

from karabo.PnP.numpex.imager import DeepinvDirtyImager, DirtyImagerConfig

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
    def is_installed(cls, env_name=None, quiet=True):
        # 1. Check if module can be imported (dependencies present)
        try:
            import deepinv
            import karabo.simulation
        except ImportError:
            return False
            
        # 2. Check data files
        try:
            data_path = Path(config.get_data_path(key="radio_interferometry"))
            ms_cache_exists = (data_path / "meerkat_cache").exists() and \
                              any((data_path / "meerkat_cache").glob("*.ms"))
            return ms_cache_exists
        except:
            return False

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
        # Check if distributed environment is already set up
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            try:
                import submitit
                submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
            except ImportError:
                pass
            except RuntimeError:
                pass

        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            device = ctx.device
            
            # Use specific data path for caching
            data_path = Path(config.get_data_path(key="radio_interferometry"))
            data_path.mkdir(parents=True, exist_ok=True)

            # Load example image (CBSD0010)
            # transform to grayscale as we simulate intensity
            img = load_cached_example(
                "m1_n.fits",
                cache_dir=data_path, 
                grayscale=True,
                device="cpu" # Load on CPU first
            )

            # Resize image
            _, _, h, w = img.shape
            if h != self.image_size or w != self.image_size:
                img = F.interpolate(img, size=(self.image_size, self.image_size), mode='bicubic')
                img = torch.clamp(img, 0, 1)
            
            ground_truth = img.to(device) # (1, 1, H, W)
            
            # Simulation parameters
            pixel_size_arcsec = 1.0
            
            # Cache directory for MS files
            ms_cache_dir = data_path / "meerkat_cache"
            
            # Generate or load visibilities
            ms_path = get_meerkat_visibilities(
                img,
                ms_cache_dir,
                pixel_size_arcsec=pixel_size_arcsec,
                freq_hz=1e9,
                obs_duration=600
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
                visibility_path=ms_path,
                visibility_format="MS",
                visibility_column="DATA"
            )
            
            # measurements come from simulation (clean or with Karabo noise)
            # Add explicit noise for benchmarking control
            if self.noise_level > 0:
                # Generate complex noise
                rng = torch.Generator(device=device).manual_seed(self.seed)
                noise = torch.randn(measurements.shape, generator=rng, device=device) + \
                        1j * torch.randn(measurements.shape, generator=rng, device=device)
                
                # Scale noise (sigma / sqrt(2) per component gives sigma for magnitude)
                noise = noise * self.noise_level / np.sqrt(2)
                measurements = measurements + noise
            
            # Attach noise model to physics
            physics.noise_model = GaussianNoise(sigma=self.noise_level)
            
            return dict(
                ground_truth=ground_truth,
                measurement=measurements,
                physics=physics,
                min_pixel=0.0,
                max_pixel=1.0,
                ground_truth_shape=ground_truth.shape,
            )
