"""High-resolution color image dataset for inverse problems benchmarking.

This dataset uses a real color image from the CBSD dataset and applies
multiple anisotropic Gaussian blur operators with equiangular orientations.
"""

import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from benchopt import BaseDataset, config
from deepinv.datasets import HDF5Dataset, generate_dataset
from deepinv.distributed import DistributedContext
from deepinv.physics import GaussianNoise, stack
from deepinv.physics.blur import Blur, gaussian_blur
from torch.utils.data import DataLoader, TensorDataset

from benchmark_utils import load_cached_example
from benchmark_utils.support_dataloader import ClampedHDF5Dataset, collate_deepinv_batch


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "highres_color_image"
    requirements = [
        "torch",
        "numpy",
        "pip::git+https://github.com/deepinv/deepinv.git@main",
    ]

    parameters = {
        "image_size": [512, 1024, 2048],
        "num_operators": [1, 8, 16],
        "noise_level": [0.01, 0.1],
        "seed": [42],
        "batch_size": [1],
        "num_workers": [4],
    }

    def __init__(
        self,
        image_size=512,
        num_operators=1,
        noise_level=0.01,
        seed=42,
        batch_size=1,
        num_workers=0,
    ):
        """Initialize the dataset."""
        self.image_size = image_size
        self.num_operators = num_operators
        self.noise_level = noise_level
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data(self):
        """Load the data for this Dataset.

        Creates stacked physics operators and measurements using deepinv examples.
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

            data_path = config.get_data_path(key="highres_color_image")

            # Load example image in original size
            img = load_cached_example(
                "CBSD_0010.png", cache_dir=data_path, grayscale=False, device=device
            )

            # Resize image so that max dimension equals self.image_size
            _, _, h, w = img.shape
            max_dim = max(h, w)

            if max_dim != self.image_size:
                scale_factor = self.image_size / max_dim
                new_h = int(h * scale_factor)
                new_w = int(w * scale_factor)

                ground_truth = F.interpolate(
                    img, size=(new_h, new_w), mode="bicubic", align_corners=False
                )
                print(f"Resized image from ({h}, {w}) to ({new_h}, {new_w})")
            else:
                ground_truth = img
                print(f"Image already at target size: ({h}, {w})")

            # Wrap ground_truth into a simple dataset
            gt_cpu = ground_truth.cpu()
            if gt_cpu.ndim == 3:
                dataset = TensorDataset(gt_cpu.unsqueeze(0))
            else:
                dataset = TensorDataset(gt_cpu)

            # Create anisotropic Gaussian blur kernels with equiangular directions
            physics_list = []

            # Set sigma values based on a fixed blur strength in normalized coordinates
            sigma_x = self.image_size * 0.01  # 1% of image size
            sigma_y = self.image_size * 0.005  # 0.5% of image size (anisotropic)

            # Calculate equiangular directions based on num_operators
            # Angles are evenly distributed over 180 degrees (since blur is symmetric)
            angles = np.linspace(0, 180, self.num_operators)

            for i in range(self.num_operators):
                # Calculate angle for this operator (equiangular spacing)
                angle = angles[i]

                # Create anisotropic blur kernel with specific angle
                kernel = gaussian_blur(
                    sigma=(sigma_x, sigma_y), angle=angle, device=str(device)
                )

                # Create blur operator with circular padding
                blur_op = Blur(
                    filter=kernel, padding="circular", device=str(device), use_fft=True
                )

                # Set the noise model with reproducible random generator
                rng = torch.Generator(device=device).manual_seed(self.seed + i)
                blur_op.noise_model = GaussianNoise(sigma=self.noise_level, rng=rng)
                blur_op = blur_op.to(device)

                physics_list.append(blur_op)

            # Stack physics operators into a single operator
            stacked_physics = stack(*physics_list)

            # Generate unique dataset filename to avoid conflicts between concurrent jobs
            # Use world_size + dataset parameters to create unique identifier
            param_str = f"{self.image_size}_{self.num_operators}_{self.noise_level}_{self.seed}_{ctx.world_size}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            dataset_filename = f"dset_{param_hash}_ws{ctx.world_size}"

            # Construct the dataset path (generate_dataset appends "0.h5")
            dataset_path = f"{data_path}/{dataset_filename}0.h5"

            if ctx.rank == 0:
                # Only generate if file doesn't exist (allows reuse across runs)
                if not os.path.exists(dataset_path):
                    print(f"Generating new dataset: {dataset_path}")
                    _ = generate_dataset(
                        train_dataset=dataset,
                        physics=stacked_physics,
                        save_dir=str(data_path),
                        dataset_filename=dataset_filename,
                        device=device,
                        train_datapoints=1,
                        num_workers=self.num_workers,
                        verbose=True,
                        supervised=True,
                        overwrite_existing=False,  # Don't overwrite to avoid conflicts
                    )
                else:
                    print(f"Reusing existing dataset: {dataset_path}")

            # Synchronize all ranks before loading datasets
            if ctx.use_dist:
                import torch.distributed as dist

                dist.barrier()

            # Load HDF5Dataset - it opens the file safely in read-only mode
            h5_dset = HDF5Dataset(path=dataset_path, train=True)
            h5_dset = ClampedHDF5Dataset(h5_dset, min_val=0.0, max_val=1.0)
            dataloader = DataLoader(
                h5_dset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_deepinv_batch,
            )

            # Get the single item for metadata and visualization
            ground_truth, _ = h5_dset[0]
            ground_truth = ground_truth.unsqueeze(0)  # Add batch dimension

        return dict(
            dataloader=dataloader,
            physics=stacked_physics,
            min_pixel=0.0,
            max_pixel=1.0,
            ground_truth_shape=ground_truth.shape,
            num_operators=self.num_operators,
        )
