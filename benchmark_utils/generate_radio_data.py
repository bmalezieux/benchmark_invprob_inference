import sys
import os

import torch
import submitit
import torch.nn.functional as F
from pathlib import Path
from benchopt.config import get_data_path
from benchmark_utils.radio_utils import get_meerkat_visibilities
from datasets.radio_interferometry import Dataset
from benchmark_utils import load_cached_example

def generate_data_for_size(image_size):
    """Generate data for a specific image size."""
    
    # Setup paths
    data_path = Path(get_data_path(key="radio_interferometry"))
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Verify/Load image
    try:
        img = load_cached_example(
            "CBSD_0010.png",
            cache_dir=data_path, 
            grayscale=True, 
            device="cpu"
        )
    except Exception as e:
        print(f"Could not load example image: {e}")
        return

    # Cache directory
    ms_cache_dir = data_path / "meerkat_cache"
    ms_cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine device for this task
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating data for image size {image_size} on {device}")
    
    # Resize image
    _, _, h, w = img.shape
    resized_img = img.clone()
    if h != image_size or w != image_size:
        resized_img = F.interpolate(img, size=(image_size, image_size), mode='bicubic')
        resized_img = torch.clamp(resized_img, 0, 1)
    
    # Simulation parameters (fixed)
    pixel_size_arcsec = 1.0
    freq_hz = 1e9
    obs_duration = 600

    # Generate visibilities
    vis_path = get_meerkat_visibilities(
        resized_img,
        ms_cache_dir,
        pixel_size_arcsec=pixel_size_arcsec,
        freq_hz=freq_hz,
        obs_duration=obs_duration,
        device=device
    )
    print(f"Visibilities ready for size {image_size}: {vis_path}")


def run_process():
    """Run generation process (local or distributed)."""
    
    # Get parameters to process
    image_sizes = Dataset.parameters['image_size']
    
    # Get configuration from environment variables
    # These can be set by the user before running benchopt
    gpus_per_job = int(os.environ.get("BENCHOPT_GPUS_PER_JOB", 1))
    
    print(f"Configuration: GPUS_PER_JOB={gpus_per_job}")

    # Check for SLURM
    has_slurm = False
    try:
        import subprocess
        subprocess.run(["sbatch", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_slurm = True
    except:
        pass
        
    if has_slurm:
        print(f"SLURM detected. Submitting {len(image_sizes)} jobs...")
        executor = submitit.AutoExecutor(folder="debug_output")
        executor.update_parameters(
            timeout_min=60,
            gpus_per_node=gpus_per_job,
            nodes=1,
            cpus_per_task=10,
        )
        
        # Submit array of jobs
        # Each job will have access to 'gpus_per_job' GPUs
        jobs = executor.map_array(generate_data_for_size, image_sizes)
        
        print(f"Jobs submitted ({[j.job_id for j in jobs]}). Waiting for completion...")
        # Wait for all jobs to complete
        for job in jobs:
            job.result()
        print("All jobs completed.")
    else:
        print("Running locally...")
        if torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
            print(f"Local GPU count: {n_devices}")
            if n_devices < gpus_per_job:
                print(f"Warning: Requested {gpus_per_job} GPUs per job but only {n_devices} available locally.")
        
        for size in image_sizes:
            generate_data_for_size(size)

if __name__ == "__main__":
    run_process()

