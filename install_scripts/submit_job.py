import submitit
import yaml
import argparse
import subprocess
import os
from pathlib import Path


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent

def run_simulation(config):
    """
    This function will run on the compute node (or locally).
    It constructs the singularity command and executes the python script inside the container.
    """
    
    project_root = get_project_root()
    image_rel_path = config['singularity']['image_path']
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    
    # Config default: "install_scripts/karabo.sif"
    image_path_candidate = repo_dir / image_rel_path
    
    # Check for singularity environment variable
    env_singularity_dir = os.environ.get("SINGULARITY_ALLOWED_DIR")

    if env_singularity_dir:
        # If Singularity environment, use the allowed dir
        image_name = image_path_candidate.name
        final_image_path = Path(env_singularity_dir) / image_name
        print(f"Singularity environment detected. Looking for image in {final_image_path}")
        if not final_image_path.exists():
            print(f"Warning: Image not found at {final_image_path}, falling back to {image_path_candidate}")
            final_image_path = image_path_candidate
    else:
        final_image_path = image_path_candidate
        
    if not final_image_path.exists():
        if (repo_dir / "karabo.sif").exists():
             final_image_path = repo_dir / "karabo.sif"
        elif Path("karabo.sif").exists():
             final_image_path = Path("karabo.sif").resolve()
        else:
             raise FileNotFoundError(f"Container image not found. Searched: {final_image_path}")

    print(f"Using container image: {final_image_path}")

    # We mount the "project_root" (deepinv_benchmark) to /workspace
    # The script inside is at /workspace/benchmark_invprob_inference/benchmark_utils/generate_radio_data.py
    # working_dir from config: "/workspace/benchmark_invprob_inference"
    working_dir = config['singularity'].get('working_dir', '/workspace')
    mount_point = config['singularity'].get('mount_point', '/workspace')
    
    # Create cache dirs in project root (which is mounted)
    cache_dir = project_root / "debug_output" / "cache"
    mpl_dir = project_root / "debug_output" / "mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    
    # Path inside container
    container_cache = f"{mount_point}/debug_output/cache"
    container_mpl = f"{mount_point}/debug_output/mpl_cache"

    cmd = [
        "singularity", "exec", "--nv",
        "-B", f"{project_root}:{mount_point}",
        "--env", f"XDG_CACHE_HOME={container_cache},MPLCONFIGDIR={container_mpl}",
        "--pwd", working_dir,
        str(final_image_path),
        "python", "benchmark_utils/generate_radio_data.py",
        "--data_path", config['job']['data_path'],
    ]
    
    if config['job'].get('use_gpus'):
        cmd.append("--use_gpus")
        
    image_sizes = config['job'].get('image_size', [])
    if image_sizes:
        cmd.append("--image_size")
        cmd.extend([str(s) for s in image_sizes])

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Simulation failed with code {result.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Submit or run Karabo simulation job")
    parser.add_argument("--config", default="install_scripts/config_slurm.yaml", help="Path to YAML config")
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to Slurm")
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Warning: Ignoring unknown arguments: {unknown}")
    
    # Allow config loading from relative path
    config_path = Path(args.config)
    if not config_path.exists():
        # Try assuming we are in install_scripts
        candidate = Path(__file__).parent / args.config
        if candidate.exists():
            config_path = candidate
            
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.local:
        print("Running locally...")
        run_simulation(config)
    else:
        # Submitit Setup
        slurm_conf = config.get('slurm', {})
        folder = slurm_conf.get('folder', 'logs')
        Path(folder).mkdir(parents=True, exist_ok=True)
        
        executor = submitit.AutoExecutor(folder=folder)
        
        kwargs = {}
        # Map Slurm parameters
        for k in ['job_name', 'time', 'gres', 'cpus_per_task', 'ntasks_per_node', 'mem', 'partition', 'account']:
            if k in slurm_conf:
                kwargs[f"slurm_{k}"] = slurm_conf[k]

        if 'hint' in slurm_conf:
            kwargs['slurm_additional_parameters'] = {'hint': slurm_conf['hint']}
        
        # Setup (modules etc)
        if "setup" in slurm_conf:
            kwargs["slurm_setup"] = slurm_conf["setup"]

        executor.update_parameters(**kwargs)
        
        # Singularity: Pre-submission copy
        singularity_cmd = config['singularity'].get('singularity_install_cmd')
        # Simple check if "idrcontmgr" is available in shell to decide if we should run it
        if singularity_cmd and subprocess.run("command -v idrcontmgr", shell=True, stdout=subprocess.DEVNULL).returncode == 0:
             image_rel = config['singularity']['image_path']
             # Assuming image is relative to benchmark_invprob_inference
             # We need to find where that is relative to CWD
             script_dir = Path(__file__).resolve().parent
             repo_dir = script_dir.parent
             full_img_path = repo_dir / image_rel
             
             cmd_str = singularity_cmd.replace("{image_path}", str(full_img_path))
             print(f"Executing Singularity image copy: {cmd_str}")
             try:
                 subprocess.run(cmd_str, shell=True, check=True)
             except subprocess.CalledProcessError as e:
                 print(f"Warning: Image copy failed: {e}")
        
        print(f"Submitting job to Slurm with config: {kwargs}")
        job = executor.submit(run_simulation, config)
        print(f"Submitted job {job.job_id}, waiting for completion...")
        job.result()  # Wait for completion
        print("Job completed.")

if __name__ == "__main__":
    main()
