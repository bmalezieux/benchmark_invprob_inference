import os
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from deepinv.physics import GaussianNoise, stack
from deepinv.physics.blur import Blur, gaussian_blur
from deepinv.utils.demo import download_example, load_image

from benchopt import BaseDataset


def load_cached_example(name, cache_dir=None, **kwargs):
    """Load example image with local caching.
    
    Downloads the image from HuggingFace if not already cached locally.
    This allows the benchmark to run on cluster nodes without internet access.
    
    Parameters
    ----------
    name : str
        Filename of the image from the HuggingFace dataset.
    cache_dir : str or Path, optional
        Directory to cache downloaded images. 
        Defaults to 'data' folder in benchmark root.
    **kwargs
        Keyword arguments to pass to load_image.
    
    Returns
    -------
    torch.Tensor
        The loaded image tensor.
    """
    # Default cache directory is 'data' in the benchmark root
    if cache_dir is None:
        benchmark_root = Path(__file__).parent.parent
        cache_dir = benchmark_root / "data"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / name
    
    # Download if not cached
    if not cached_file.exists():
        print(f"Downloading {name} to {cache_dir}...")
        download_example(name, cache_dir)
        print(f"Downloaded {name} successfully.")
    else:
        print(f"Using cached {name} from {cache_dir}")
    
    # Load from local file
    # For .pt files, we need to use torch.load
    if name.endswith('.pt'):
        device = kwargs.get('device', 'cpu')
        return torch.load(cached_file, weights_only=True, map_location=device)
    else:
        # For image files, use load_image
        return load_image(str(cached_file), **kwargs)


def save_debug_figure(ground_truth, measurement, output_dir="debug_output"):
    """Save a figure showing ground truth and all measurements.
    
    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth image tensor of shape (C, H, W).
    measurement : TensorList or list of torch.Tensor
        List of measurement tensors, each of shape (C, H, W).
    output_dir : str or Path
        Directory to save the debug figure.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy array suitable for imshow."""
        # Move to CPU and detach
        img = tensor.detach().cpu()
        # Remove batch dimension if present (B, C, H, W) -> (C, H, W)
        if img.ndim == 4:
            img = img.squeeze(0)
        # If shape is (C, H, W), transpose to (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        # Remove channel dim if grayscale
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        # Clip to [0, 1] range
        img = torch.clamp(img, 0, 1)
        return img.numpy()
    
    # Calculate grid size
    num_measurements = len(measurement)
    total_images = num_measurements + 1  # +1 for ground truth
    
    # Create a grid layout
    cols = min(4, total_images)  # Max 4 columns
    rows = (total_images + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    # Handle single subplot case
    if total_images == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    # Flatten axes for easier iteration
    axes_flat = [ax for row in axes for ax in row]
    
    # Plot ground truth
    gt_img = tensor_to_numpy(ground_truth)
    axes_flat[0].imshow(gt_img, cmap='gray' if gt_img.ndim == 2 else None)
    axes_flat[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes_flat[0].axis('off')
    
    # Plot measurements
    for i, meas in enumerate(measurement):
        meas_img = tensor_to_numpy(meas)
        axes_flat[i + 1].imshow(meas_img, cmap='gray' if meas_img.ndim == 2 else None)
        axes_flat[i + 1].set_title(f'Measurement {i + 1}', fontsize=12)
        axes_flat[i + 1].axis('off')
    
    # Hide unused subplots
    for i in range(total_images, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = output_path / 'measurements_debug.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Debug figure saved to {output_file}")


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

        # Load example image in original size
        img = load_cached_example(
            "CBSD_0010.png", 
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
        angle_step = 180.0 / self.num_operators if self.num_operators > 1 else 0.0

        for i in range(self.num_operators):
            # Calculate angle for this operator (equiangular spacing)
            angle = i * angle_step
            
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
        save_debug_figure(ground_truth, measurement)

        return dict(
            ground_truth=ground_truth,
            measurement=measurement,
            physics=stacked_physics,
            min_pixel=0.0,
            max_pixel=1.0,
            ground_truth_shape=ground_truth.shape,
            num_operators=self.num_operators,
        )