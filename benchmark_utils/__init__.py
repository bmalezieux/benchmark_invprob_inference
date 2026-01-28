"""Benchmark utilities for inverse problems.

This module provides shared utilities for datasets, solvers, and objectives,
including visualization and data loading helpers.
"""
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import torch
    from deepinv.utils.demo import download_example, load_image
    from deepinv.models import DRUNet
except ImportError:
    torch = None
    download_example = None
    load_image = None
    DRUNet = None


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array suitable for visualization.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (B, C, D, H, W), (B, C, H, W), (C, H, W), or (H, W).
        
    Returns
    -------
    np.ndarray
        Numpy array suitable for matplotlib imshow, with values in [0, 1].
    """
    img = tensor.detach().cpu()

    # Handle 5D tensor (B, C, D, H, W) -> take middle slice
    if img.ndim == 5:
        mid_slice = img.shape[2] // 2
        img = img[:, :, mid_slice, :, :]
    
    # Remove batch dimension if present
    if img.ndim == 4:
        img = img.squeeze(0)
    
    # Transpose from (C, H, W) to (H, W, C) for color images
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.permute(1, 2, 0)
    
    # Remove channel dimension for grayscale
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    
    # Clip to valid range
    img = torch.clamp(img, 0, 1)
    
    return img.numpy()


def save_measurements_figure(ground_truth, measurement, output_dir="debug_output", 
                             filename="measurements_debug.png"):
    """Save a figure showing ground truth and all measurements.
    
    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth image tensor of shape (C, H, W) or (B, C, H, W).
    measurement : list of torch.Tensor or TensorList
        List of measurement tensors, each of shape (C, H, W) or (B, C, H, W).
    output_dir : str or Path, optional
        Directory to save the debug figure. Default: "debug_output".
    filename : str, optional
        Name of the output file. Default: "measurements_debug.png".
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_measurements = len(measurement)
    total_images = num_measurements + 1  # +1 for ground truth
    
    # Create grid layout (max 4 columns)
    cols = min(4, total_images)
    rows = (total_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()
    
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
    
    plt.tight_layout()
    output_file = output_path / filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Debug figure saved to {output_file}")


def save_comparison_figure(ground_truth, reconstruction, metrics, output_dir="evaluation_output",
                           filename="comparison.png", evaluation_count=None):
    """Save a comparison figure showing ground truth and reconstruction side by side.
    
    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth tensor of shape (C, H, W) or (B, C, H, W).
    reconstruction : torch.Tensor
        Reconstruction tensor of same shape as ground_truth.
    metrics : dict
        Dictionary with 'psnr' and 'ssim' keys.
    output_dir : str or Path, optional
        Directory to save the comparison figure. Default: "evaluation_output".
    filename : str, optional
        Name of the output file. Default: "comparison.png".
    evaluation_count : int, optional
        Evaluation number to display in title.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot ground truth
    gt_img = tensor_to_numpy(ground_truth)
    axes[0].imshow(gt_img, cmap='gray' if gt_img.ndim == 2 else None)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot reconstruction
    recon_img = tensor_to_numpy(reconstruction)
    psnr = metrics.get('psnr', 0)
    ssim = metrics.get('ssim', 0)
    axes[1].imshow(recon_img, cmap='gray' if recon_img.ndim == 2 else None)
    axes[1].set_title(f'Reconstruction\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add overall title if evaluation count provided
    if evaluation_count is not None:
        fig.suptitle(f'Evaluation #{evaluation_count}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_path / filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def load_cached_example(name, cache_dir=None, **kwargs):
    """Load example image with local caching.
    
    Downloads the image from HuggingFace if not already cached locally.
    This allows benchmarks to run on cluster nodes without internet access.
    
    Parameters
    ----------
    name : str
        Filename of the image from the HuggingFace dataset.
    cache_dir : str or Path, optional
        Directory to cache downloaded images. 
        Defaults to 'data' folder in benchmark root.
    **kwargs
        Additional arguments passed to load_image.
    
    Returns
    -------
    torch.Tensor
        The loaded image tensor.
    """
    if cache_dir is None:
        # Get benchmark root (parent of benchmark_utils)
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
    if name.endswith('.pt'):
        device = kwargs.get('device', 'cpu')
        return torch.load(cached_file, weights_only=True, map_location=device)
    else:
        return load_image(str(cached_file), **kwargs)


def create_drunet_denoiser(ground_truth_shape, device='cpu', dtype=None):
    """Create a DRUNet denoiser appropriate for the given ground truth shape.
    
    Automatically detects whether to use:
    - Grayscale (1 channel) or color (3 channels) based on channel dimension
    - 2D or 3D based on number of spatial dimensions
    
    Parameters
    ----------
    ground_truth_shape : tuple
        Shape of the ground truth tensor.
        For 2D: (B, C, H, W)
        For 3D: (B, C, D, H, W)
    device : str or torch.device, optional
        Device to load the model on. Default: 'cpu'.
    dtype : torch.dtype, optional
        Data type for the model. Default: torch.float32.
        
    Returns
    -------
    DRUNet
        Configured DRUNet denoiser model.
    """
    if torch is None:
        raise ImportError("torch is required for create_drunet_denoiser")
        
    if dtype is None:
        dtype = torch.float32

    from .support_3d import transform_2d_to_3d, patch_drunet_3d
    
    # Determine dimensionality
    ndim = len(ground_truth_shape)
    if ndim == 4:
        # 2D case: (B, C, H, W)
        is_3d = False
        num_channels = ground_truth_shape[1]
    elif ndim == 5:
        # 3D case: (B, C, D, H, W)
        is_3d = True
        num_channels = ground_truth_shape[1]
    else:
        raise ValueError(f"Unsupported ground truth shape: {ground_truth_shape}. Expected 4D (B,C,H,W) or 5D (B,C,D,H,W).")
    
    # Determine if grayscale or color
    if num_channels == 1:
        # Grayscale: use single-channel DRUNet
        print(f"Creating grayscale DRUNet (1 channel, {'3D' if is_3d else '2D'})")
        model = DRUNet(in_channels=1, out_channels=1, pretrained="download")
    elif num_channels == 3:
        # Color: use default RGB DRUNet
        print(f"Creating color DRUNet (3 channels, {'3D' if is_3d else '2D'})")
        model = DRUNet(pretrained="download")
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1 (grayscale) or 3 (color).")
    
    # Transform to 3D if needed
    if is_3d:
        print(f"Transforming 2D DRUNet to 3D using transform_2d_to_3d")
        transform_2d_to_3d(model)
        patch_drunet_3d(model)

    # Move to device and dtype
    model = model.to(dtype).to(device)
    
    return model
