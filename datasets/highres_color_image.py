"""High-resolution color image dataset for inverse problems benchmarking.

This dataset uses a real color image from the CBSD dataset and applies
multiple anisotropic Gaussian blur operators with equiangular orientations.
"""
import torch
import torch.nn.functional as F
import numpy as np

from deepinv.physics import GaussianNoise, stack
from deepinv.physics.blur import Blur, gaussian_blur

from benchopt import BaseDataset
from benchopt import config
from benchmark_utils import load_cached_example, save_measurements_figure


class Dataset(BaseDataset):
    # Name of the Dataset, used to select it in the CLI
    name = 'highres_color_image'

    parameters = {
        'image_size': [512, 1024, 2048],
        'num_operators': [1, 8, 16],
        'noise_level': [0.01, 0.1],
        'seed': [42],
    }

    def __init__(self, image_size=512, num_operators=1, noise_level=0.01, seed=42):
        """Initialize the dataset."""
        self.image_size = image_size
        self.num_operators = num_operators
        self.noise_level = noise_level
        self.seed = seed

    def get_data(self):
        """Load the data for this Dataset.

        Creates stacked physics operators and measurements using deepinv examples.
        Returns dictionary with keys expected by Objective.set_data().
        """
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_path = config.get_data_path(key="highres_color_image")

        # Load example image in original size
        img = load_cached_example(
            "CBSD_0010.png",
            cache_dir=data_path, 
            grayscale=False, 
            device=device
        )

        # Resize image so that max dimension equals self.image_size
        _, _, h, w = img.shape
        max_dim = max(h, w)
        
        if max_dim != self.image_size:
            scale_factor = self.image_size / max_dim
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            ground_truth = F.interpolate(
                img, 
                size=(new_h, new_w), 
                mode='bicubic', 
                align_corners=False
            )
            print(f"Resized image from ({h}, {w}) to ({new_h}, {new_w})")
        else:
            ground_truth = img
            print(f"Image already at target size: ({h}, {w})")

        # Create anisotropic Gaussian blur kernels with equiangular directions
        physics_list = []

        # Set sigma values based on a fixed blur strength in normalized coordinates
        sigma_x = self.image_size * 0.02  # 2% of image size
        sigma_y = self.image_size * 0.01  # 1% of image size (anisotropic)
        
        # Calculate equiangular directions based on num_operators
        # Angles are evenly distributed over 180 degrees (since blur is symmetric)
        angles = np.linspace(0, 180, self.num_operators)

        for i in range(self.num_operators):
            # Calculate angle for this operator (equiangular spacing)
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

        for i in range(len(measurement)):
            measurement[i] = torch.clamp(measurement[i], 0.0, 1.0)

        # Save debug visualization
        save_measurements_figure(ground_truth, measurement, filename="highres_color_image_measurements.png")

        return dict(
            ground_truth=ground_truth,
            measurement=measurement,
            physics=stacked_physics,
            min_pixel=0.0,
            max_pixel=1.0,
            ground_truth_shape=ground_truth.shape,
            num_operators=self.num_operators,
        )