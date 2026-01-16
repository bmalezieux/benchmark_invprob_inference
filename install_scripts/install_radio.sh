#!/bin/bash
set -e

# Get the environment directory (passed as first argument)
ENV_DIR=$1

# Install dependencies using pip in the environment
echo "Installing dependencies..."
$ENV_DIR/bin/pip install --upgrade pip
$ENV_DIR/bin/pip install deepinv
$ENV_DIR/bin/pip install submitit

# Check if Karabo-Pipeline is in the current directory (local workspace)
if [ -d "Karabo-Pipeline" ]; then
    echo "Installing dependencies from local Karabo-Pipeline/environment.yaml..."
    # Use CONDA_EXE if available, or try to find conda
    if [ -z "$CONDA_EXE" ]; then
        CONDA_BIN=$(which conda)
    else
        CONDA_BIN="$CONDA_EXE"
    fi
    
    echo "Using conda: $CONDA_BIN"
    "$CONDA_BIN" env update -p $ENV_DIR -f Karabo-Pipeline/environment.yaml --prune
    
    echo "Installing local Karabo-Pipeline..."
    $ENV_DIR/bin/pip install -e Karabo-Pipeline
else
    echo "Cloning and installing Karabo-Pipeline..."
    $ENV_DIR/bin/pip install git+https://github.com/aleph-group/Karabo-Pipeline.git
fi

# Run the data generation script
echo "Running data generation..."
# We assume the user runs benchopt from the benchmark root directory
# so benchmark_utils/generate_radio_data.py is accessible
$ENV_DIR/bin/python benchmark_utils/generate_radio_data.py
