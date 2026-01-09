"""
Distributed Benchmark Run with torchrun (CPU)
==============================================

Load dataset and run benchmark with torchrun for distributed execution.
"""

# %%
# Setup
# =====
import os
import sys
import subprocess
from pathlib import Path

# Add repo root to path
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    # When running in Sphinx-Gallery or notebook without __file__
    cwd = Path.cwd()
    # Search upward for benchmark root (directory containing 'configs', 'datasets', 'solvers')
    current = cwd
    while current != current.parent:
        if (current / 'configs').exists() and (current / 'datasets').exists():
            project_root = current
            break
        current = current.parent
    else:
        # Fallback: use cwd
        project_root = cwd

sys.path.insert(0, str(project_root))

from benchopt import config as bench_config
from benchopt.config import yaml
bench_config.get_data_path = lambda key=None: str(project_root / "data")



# %%
# Load Configuration
# ==================
# Load config from YAML file
config_path = project_root / 'configs' / 'highres_imaging_torchrun.yml'
print(f"\nLoading config from: {config_path.name}")
with open(config_path, 'r') as f:
    config_content = yaml.safe_load(f)

# Extract dataset parameters
dataset_params = config_content['dataset'][0]['highres_color_image']
print(f"\nDataset configuration:")
for key, val in dataset_params.items():
    print(f"  - {key}: {val}")

# %%
# Load Dataset
# ============
from datasets.highres_color_image import Dataset
print("\nLoading dataset...")
ds = Dataset(**dataset_params, seed=42)
data = ds.get_data()

print(f"\nDataset loaded successfully:")
print(f"  - Ground truth shape: {data['ground_truth'].shape}")
print(f"  - Measurements: {len(data['measurement'])} operators")
print(f"  - First measurement shape: {data['measurement'][0].shape}")

# %%
# Visualize Dataset
# =================
# Optional: visualize ground truth and measurements
try:
    from deepinv.utils.plotting import plot
    
    gt = data['ground_truth']
    meas = data['measurement'][0]  # First measurement

    print("\nGenerating preview plot...")
    plot([gt, meas], titles=['Ground Truth', 'Measurement 1'])
except Exception as e:
    print(f"\nNote: Skipping visualization ({type(e).__name__})")



# %%
# Run Benchmark
# =============
# To run the distributed benchmark with torchrun, execute the command below.

print("\n" + "-" * 70)
print("To run the distributed benchmark with torchrun, execute:")
print("-" * 70)
cmd = [
    'benchopt', 'run', '.',
    '--parallel-config', './configs/torchrun_config.yml',
    '--config', './configs/highres_imaging_torchrun.yml'
]
print(f"  {' '.join(cmd)}")
print("-" * 70)

# Uncomment to run:
# subprocess.run(cmd, check=True)
