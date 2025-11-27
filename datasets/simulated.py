"""Simple simulated dataset for inverse problems benchmarking.

This dataset generates a synthetic image and applies multiple blur operators
for quick testing and validation of reconstruction algorithms.
"""
import os
import torch
import numpy as np

from deepinv.physics import GaussianNoise, stack
from deepinv.physics.blur import Blur, gaussian_blur
from deepinv.distrib import DistributedContext

from benchopt import BaseDataset
from benchmark_utils import save_measurements_figure


class Dataset(BaseDataset):
    """Simple simulated dataset with synthetic image and blur operators."""
    
    name = 'simulated'

    parameters = {
        'image_size': [256],
        'num_operators': [1, 4],
        'noise_level': [0.01],
        'seed': [42],
    }

    def __init__(self, image_size=128, num_operators=1, noise_level=0.01, seed=42):
        """Initialize the simulated dataset.
        
        Parameters
        ----------
        image_size : int
            Size of the square synthetic image.
        num_operators : int
            Number of blur operators to stack.
        noise_level : float
            Standard deviation of Gaussian noise.
        seed : int
            Random seed for reproducibility.
        """
        self.image_size = image_size
        self.num_operators = num_operators
        self.noise_level = noise_level
        self.seed = seed

    def get_data(self):
        """Generate synthetic data for the benchmark.

        Creates a simple geometric pattern (circles and gradients) as ground truth,
        applies stacked blur operators, and adds noise.
        
        Returns
        -------
        dict
            Dictionary with keys: ground_truth, measurement, physics, 
            min_pixel, max_pixel, ground_truth_shape, num_operators.
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
            
            # Set random seed for reproducibility
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            
            # Generate synthetic ground truth image (grayscale)
            ground_truth = self._generate_synthetic_image(device)
            
            # Create anisotropic Gaussian blur kernels with equiangular directions
            physics_list = []
            
            # Set sigma values based on a fixed blur strength in normalized coordinates
            sigma_x = self.image_size * 0.02  # 2% of image size
            sigma_y = self.image_size * 0.01  # 1% of image size (anisotropic)
            
            # Calculate equiangular directions based on num_operators
            angles = np.linspace(0, 180, self.num_operators)

            for i in range(self.num_operators):
                angle = angles[i]
                
                # Create anisotropic blur kernel with specific angle
                kernel = gaussian_blur(
                    sigma=(sigma_x, sigma_y), 
                    angle=angle, 
                    device=str(device)
                )
                
                # Create blur operator with circular padding
                blur_op = Blur(filter=kernel, padding="circular", device=str(device))

                # Set the noise model with reproducible random generator
                rng = torch.Generator(device=device).manual_seed(self.seed + i)
                blur_op.noise_model = GaussianNoise(sigma=self.noise_level, rng=rng)
                blur_op = blur_op.to(device)

                physics_list.append(blur_op)

            # Stack physics operators into a single operator
            stacked_physics = stack(*physics_list)

            # Generate measurements (returns a TensorList)
            measurement = stacked_physics(ground_truth)

            # Clamp measurements to valid range
            for i in range(len(measurement)):
                measurement[i] = torch.clamp(measurement[i], 0.0, 1.0)

            if ctx.rank == 0:
                # Save debug visualization
                save_measurements_figure(ground_truth, measurement)

        return dict(
            ground_truth=ground_truth,
            measurement=measurement,
            physics=stacked_physics,
            min_pixel=0.0,
            max_pixel=1.0,
            ground_truth_shape=ground_truth.shape,
            num_operators=self.num_operators,
        )

    def _generate_synthetic_image(self, device):
        """Generate a synthetic test image with geometric patterns.
        
        Creates a color image with circles, gradients, and shapes for testing
        reconstruction algorithms. The image has 3 color channels (RGB).
        
        Parameters
        ----------
        device : torch.device
            Device to create the tensor on.
            
        Returns
        -------
        torch.Tensor
            Synthetic image of shape (1, 3, image_size, image_size).
        """
        size = self.image_size
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size, device=device),
            torch.linspace(-1, 1, size, device=device),
            indexing='ij'
        )
        
        # Initialize RGB channels
        r_channel = torch.zeros((size, size), device=device)
        g_channel = torch.zeros((size, size), device=device)
        b_channel = torch.zeros((size, size), device=device)
        
        # Red channel: circular pattern
        radius1 = torch.sqrt(x**2 + y**2)
        r_channel += (radius1 < 0.5).float() * 0.8
        r_channel += (x + 1) / 4 * 0.3
        
        # Green channel: smaller circle and gradient
        radius2 = torch.sqrt((x - 0.3)**2 + (y - 0.3)**2)
        g_channel += (radius2 < 0.3).float() * 0.7
        g_channel += (y + 1) / 4 * 0.4
        
        # Blue channel: high-frequency pattern and gradient
        b_channel += torch.sin(x * 10) * torch.sin(y * 10) * 0.3 + 0.5
        b_channel += (x - y) / 4 * 0.2
        
        # Normalize each channel to [0, 1]
        r_channel = torch.clamp(r_channel, 0, 1)
        g_channel = torch.clamp(g_channel, 0, 1)
        b_channel = torch.clamp(b_channel, 0, 1)
        
        # Stack into RGB image: (3, H, W)
        img = torch.stack([r_channel, g_channel, b_channel], dim=0)
        
        # Add batch dimension: (1, 3, H, W)
        img = img.unsqueeze(0)
        
        return img
