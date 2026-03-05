"""Radio interferometry dataset using Karabo MeerKAT simulation.

This dataset simulates MeerKAT observations of real images (converted to SkyModels).
It uses pre-generated data from benchopt install.
"""
import torch
import numpy as np
import os
import json
from pathlib import Path
from astropy.io import fits

from benchopt import BaseDataset
from benchopt import config
from benchmark_utils.radio_utils import load_new_header


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = 'radio_interferometry'
    
    install_cmd = 'shell'
    install_script = 'install_radio.sh'

    parameters = {
        'image_size': [256],
        'noise_level': [0.1],
        'seed': [42],
        'fits_name': [None],
        'simulator_hash': [None],
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

    def __init__(
        self,
        image_size=256,
        noise_level=0.1,
        seed=42,
        fits_name=None,
        simulator_hash=None,
    ):
        """Initialize the dataset."""
        super().__init__()
        self.image_size = image_size
        self.noise_level = noise_level
        self.seed = seed
        self.fits_name = fits_name
        self.simulator_hash = simulator_hash

    @staticmethod
    def _list_available_simulations(ms_cache_dir: Path):
        simulations = []
        for metadata_path in sorted(ms_cache_dir.glob("*.meta.json")):
            simulation_hash = metadata_path.name.removesuffix(".meta.json")
            ms_path = ms_cache_dir / f"{simulation_hash}.ms"
            if not ms_path.exists():
                continue
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                metadata = {"metadata_error": str(exc)}
            simulations.append((simulation_hash, metadata))
        return simulations

    @classmethod
    def _format_available_simulations(cls, ms_cache_dir: Path) -> str:
        simulations = cls._list_available_simulations(ms_cache_dir)
        if not simulations:
            return (
                f"No available simulations found in {ms_cache_dir}. "
                "Run `benchopt install -d radio_interferometry` first."
            )

        lines = [f"Available simulations in {ms_cache_dir}:"]
        for simulation_hash, metadata in simulations:
            metadata_str = json.dumps(metadata, sort_keys=True)
            lines.append(f"- simulator_hash={simulation_hash} | metadata={metadata_str}")
        return "\n".join(lines)

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

            fits_name = self.fits_name

            if fits_name is None:
                raise ValueError("fits_name is not set in the config.")

            # Cache directory for MS files
            ms_cache_dir = data_path / "meerkat_cache"
            ms_cache_dir.mkdir(parents=True, exist_ok=True)

            fits_stem = Path(fits_name).stem
            cached_resized_fits_path = ms_cache_dir / f"{fits_stem}_{self.image_size}.fits"

            from benchmark_utils.radio_utils import load_and_resize_image

            if cached_resized_fits_path.exists():
                # Cached FITS is already normalized and resized — load directly.
                with fits.open(cached_resized_fits_path, memmap=False) as hdul:
                    img_np = np.array(hdul[0].data, dtype=np.float32, copy=True)
            else:
                load_cached_example(
                    fits_name,
                    cache_dir=data_path,
                    grayscale=True,
                    device="cpu",
                )

                source_fits_path = data_path / fits_name
                img_np = load_and_resize_image(source_fits_path, self.image_size)
                new_header = load_new_header(source_fits_path, self.image_size)
                fits.PrimaryHDU(img_np, header=new_header).writeto(cached_resized_fits_path, overwrite=True)

            if not img_np.dtype.isnative:
                img_np = img_np.byteswap().view(img_np.dtype.newbyteorder("="))

            img = torch.from_numpy(img_np)

            # Ensure (1, C, H, W)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            
            ground_truth = img.to(device)
            _, _, h, w = ground_truth.shape

            simulation_hash = self.simulator_hash
            if isinstance(simulation_hash, str):
                simulation_hash = simulation_hash.strip()

            if not simulation_hash:
                raise ValueError(
                    "`simulator_hash` is not set in the dataset config.\n"
                    "Set `dataset.radio_interferometry.simulator_hash` in "
                    "`radio_config.yaml`.\n"
                    f"{self._format_available_simulations(ms_cache_dir)}"
                )

            # Get path to visibilities from the selected simulation hash.
            ms_path = ms_cache_dir / f"{simulation_hash}.ms"

            if not ms_path.exists():
                raise FileNotFoundError(
                    f"Measurement Set file not found for simulator_hash={simulation_hash} "
                    f"at {ms_path}.\n"
                    f"{self._format_available_simulations(ms_cache_dir)}"
                )

            metadata_path = ms_cache_dir / f"{simulation_hash}.meta.json"
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                imaging_cellsize = float(metadata["imaging_cellsize"])
            else:
                raise FileNotFoundError(
                    f"Metadata file not found for simulator_hash={simulation_hash}. "
                    f"Expected at {metadata_path}.\n"
                    f"{self._format_available_simulations(ms_cache_dir)}"
                )
            
            # Create Physics Operator
            imager_config = DirtyImagerConfig(
                imaging_npixel=self.image_size,
                imaging_cellsize=imaging_cellsize,
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
