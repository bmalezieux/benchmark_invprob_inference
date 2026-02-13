import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Add benchmark root to sys.path to resolve benchmark_utils imports when run as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from benchmark_utils.karabo_utils import generate_meerkat_visibilities
from benchmark_utils.radio_utils import load_and_resize_image

def generate_data_for_size(image_size, data_path, use_gpus=False, number_of_time_steps=None, start_frequency_hz=None, end_frequency_hz=None, number_of_channels=None):
    """Generate data for a specific image size."""
    
    # Cache directory
    data_path = Path(data_path)
    ms_cache_dir = data_path / "meerkat_cache"
    ms_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify/Load image
    fits_file = data_path / "m1_n.fits"
    if not fits_file.exists():
        print(f"File not found: {fits_file}")
        # Try finding it in default benchmark data dir relative to script
        default_data_dir = Path(__file__).resolve().parent.parent / "data"
        fits_file = default_data_dir / "m1_n.fits"
        if not fits_file.exists():
            print(f"Could not find m1_n.fits in {data_path} or {default_data_dir}")
            return
            
    try:
        resized_img = load_and_resize_image(fits_file, image_size)
    except Exception as e:
        print(f"Could not load/process example image: {e}")
        return

    print(f"Generating data for image size {image_size} with use_gpus={use_gpus}")
    
    # Simulation parameters (fixed)
    pixel_size_arcsec = 1.0

    # Generate visibilities
    vis_path = generate_meerkat_visibilities(
        fits_file,
        resized_img,
        ms_cache_dir,
        pixel_size_arcsec=pixel_size_arcsec,
        use_gpus=use_gpus,
        number_of_time_steps=number_of_time_steps,
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
        number_of_channels=number_of_channels
    )
    
    # Cache the ground truth image
    gt_path = ms_cache_dir / f"image_{image_size}.npy"
    np.save(gt_path, resized_img)
    print(f"Ground truth cached at: {gt_path}")

    print(f"Visibilities ready for size {image_size}: {vis_path}")

def main_generation_loop(image_sizes, data_path, use_gpus, number_of_time_steps, start_frequency_hz, end_frequency_hz, number_of_channels):
    for size in image_sizes:
        generate_data_for_size(size, data_path, use_gpus=use_gpus, number_of_time_steps=number_of_time_steps, start_frequency_hz=start_frequency_hz, end_frequency_hz=end_frequency_hz, number_of_channels=number_of_channels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, nargs="+", default=[256])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--use_gpus", action="store_true", help="Whether to use GPUs for simulation."
    )
    parser.add_argument("--number_of_time_steps", type=int, default=256)
    parser.add_argument("--start_frequency_hz", type=float, default=100e6)
    parser.add_argument("--end_frequency_hz", type=float, default=120e6)
    parser.add_argument("--number_of_channels", type=int, default=12)
    args = parser.parse_args()
    
    # Check if GPU is available 
    # In karabo env, we assume cpu usually, or if torch is missing we definitely use cpu
    # But this script is running in karabo env WITHOUT torch.
    use_gpus = args.use_gpus
    print(f"Running generation with use_gpus={use_gpus}")

    main_generation_loop(args.image_size, args.data_path, use_gpus,
         args.number_of_time_steps, args.start_frequency_hz, args.end_frequency_hz,
         args.number_of_channels)