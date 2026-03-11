"""Training dataset using Urban100 for unrolled model benchmarking.

This dataset uses Urban100HR with stacked anisotropic Gaussian blur physics operators
with equiangular orientations.
"""

import hashlib
import os
import torch
import numpy as np
from benchopt import BaseDataset, config
from deepinv.physics import Denoising, GaussianNoise, stack
from deepinv.distributed import DistributedContext
from deepinv.physics.blur import Blur, gaussian_blur
from torchvision import transforms
from deepinv.datasets import generate_dataset, HDF5Dataset
from torch.utils.data import Subset, DataLoader
import deepinv as dinv

from benchmark_utils.support_dataloader import collate_deepinv_batch


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = "urban100"
    requirements = [
        "torch",
        "numpy",
        "torchvision",
        "pip::git+https://github.com/deepinv/deepinv.git@main"
    ]

    parameters = {
        "image_size": [512, 1024, 2048],
        "num_operators": [1, 8,16],
        "noise_level": [0.01, 0.1],
        "seed": [42],
        "batch_size": [1, 2],
        "num_images": [1, 4, 8],
        "num_workers": [4],
    }

    def __init__(self, image_size=128, num_operators=2, noise_level=0.06, seed=42, num_images=4, batch_size=1, num_workers=4):
        """Initialize the training dataset."""
        self.image_size = image_size
        self.num_operators = num_operators
        self.noise_level = noise_level
        self.seed = seed
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_workers = num_workers


    def get_data(self):
        """Load the data for training.

        Creates stacked anisotropic Gaussian blur physics operators and measurements using deepinv.
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
                print("submitit not installed, dataset will run in non-distributed mode")
            except RuntimeError as e:
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
        with DistributedContext(seed=self.seed, cleanup=False) as ctx:
            print(f"DistributedContext: rank {ctx.rank} / {ctx.world_size}")

            # Setup device
            device = ctx.device

            data_path = config.get_data_path(key=self.name)
            
            # Define transform for Urban100HR
            train_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()
            ])

            # Load Urban100HR dataset
            os.makedirs(data_path, exist_ok=True)
            
            full_dataset = dinv.datasets.Urban100HR(
                root=str(data_path), 
                download=True, 
                transform=train_transform
            )
            
            # Split into train and validation datasets
            max_images = min(len(full_dataset), self.num_images)
            dataset = Subset(full_dataset, range(max_images))
            
            print(f"Using {max_images} images from Urban100HR dataset")
            
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
            param_str = f"{self.image_size}_{self.num_operators}_{self.noise_level}_{self.seed}_{self.num_images}_{ctx.world_size}"
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            dataset_filename = f"dset_{param_hash}_ws{ctx.world_size}"
         
            # Note: generate_dataset adds rank suffix when in distributed mode
            # Since only rank 0 generates, files will have "0" suffix
            dataset_path = os.path.join(str(data_path), f"{dataset_filename}0.h5")
            
            if ctx.rank == 0:
                 # Only generate if file doesn't exist (allows reuse across runs)
                if not os.path.exists(dataset_path):
                # Generate dataset using deepinv
                    # generate_dataset returns the actual path with rank suffix
                    generate_dataset(
                        train_dataset=dataset,
                        physics=stacked_physics,
                        save_dir=str(data_path),
                        dataset_filename=dataset_filename,
                        device=device,
                        train_datapoints=self.num_images,
                        num_workers=self.num_workers,
                        verbose=True,
                        supervised=True,
                        overwrite_existing=True,
                    )
                else:
                    print(f"Reusing existing dataset: {dataset_path}")
             
            
            # Synchronize all ranks before loading datasets
            if ctx.use_dist:
                import torch.distributed as dist
                dist.barrier()
            
            # Load HDF5Dataset - it opens the file safely in read-only mode
            h5_dset = HDF5Dataset(path=dataset_path, train=True)
            
            # Use custom collate function to handle TensorList objects
            dataloader = DataLoader(
                h5_dset, 
                batch_size=self.batch_size, 
                shuffle=False,  # Keep same order across ranks for distributed training
                collate_fn=collate_deepinv_batch,
                num_workers=self.num_workers,
                pin_memory=True
            )
            


            # Get ground truth shape from first sample
            ground_truth_shape = (len(h5_dset),) + tuple(h5_dset[0][0].shape)
            print(f"HDF5Dataset loaded: {len(h5_dset)} samples, ground truth shape: {ground_truth_shape}")


        return dict(
            dataloader=dataloader,
            physics=stacked_physics,
            min_pixel=0.0,
            max_pixel=1.0,
            ground_truth_shape=ground_truth_shape,
            num_operators=self.num_operators,
        )
