#!/bin/bash
set -e

# The main environment directory used by benchopt
ENV_DIR=$1

# 1. Install dependencies for the MAIN environment (Python 3.12 compatible)
echo "Installing main dependencies (deepinv, astropy)..."

# Use CONDA_EXE if available, or try to find conda
if [ -z "$CONDA_EXE" ]; then
    CONDA_BIN=$(which conda)
else
    CONDA_BIN="$CONDA_EXE"
fi

if [ -z "$CONDA_BIN" ]; then
    echo "Conda not found! Cannot proceed with Karabo environment creation."
    exit 1
fi

echo "Using conda: $CONDA_BIN"

# Install dependencies for main env
# We use pip for deepinv, astropy. 
$ENV_DIR/bin/pip install -e ../deepinv
$ENV_DIR/bin/pip install astropy

# 2. Setup separate Karabo environment (Python 3.9)
echo "Setting up Karabo environment..."

# Clone Karabo-Pipeline if not present
if [ ! -d "Karabo-Pipeline" ]; then
    echo "Cloning Karabo-Pipeline..."
    git clone https://github.com/aleph-group/Karabo-Pipeline.git
fi

# Initialize conda for shell interaction
eval "$($CONDA_BIN shell.bash hook)"

KARABO_ENV_NAME="karabo_env_benchopt"

# Create/Update environment from environment.yaml (which has python 3.9)
echo "Creating/Updating conda env $KARABO_ENV_NAME..."
$CONDA_BIN env update -n "$KARABO_ENV_NAME" -f benchmark_utils/karabo_env_minimal.yml

# Activate the environment
echo "Activating $KARABO_ENV_NAME..."
conda activate "$KARABO_ENV_NAME"

# Install Karabo itself in editable mode
pip install -e Karabo-Pipeline

# 3. Run data generation
echo "Running data generation with active environment..."
# Data path relative to benchmark root
DATA_PATH="./data/radio_interferometry"
mkdir -p "$DATA_PATH"

# Run generation
python benchmark_utils/generate_radio_data.py --data_path "$DATA_PATH" --image_size 256

echo "Deactivating..."
conda deactivate

echo "Installation and data generation complete."
