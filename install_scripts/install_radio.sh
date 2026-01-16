#!/bin/bash
set -e

# Get the environment directory (passed as first argument)
ENV_DIR=$1

# Install dependencies using pip in the environment
echo "Installing dependencies..."
$ENV_DIR/bin/pip install --upgrade pip
$ENV_DIR/bin/pip install deepinv torch
$ENV_DIR/bin/pip install submitit

# Clone Karabo-Pipeline if not present
if [ ! -d "Karabo-Pipeline" ]; then
    echo "Cloning Karabo-Pipeline..."
    git clone https://github.com/aleph-group/Karabo-Pipeline.git
fi

echo "Installing dependencies from Karabo-Pipeline/karabo_env.yml..."

# Use CONDA_EXE if available, or try to find conda
if [ -z "$CONDA_EXE" ]; then
    CONDA_BIN=$(which conda)
else
    CONDA_BIN="$CONDA_EXE"
fi

echo "Using conda: $CONDA_BIN"
"$CONDA_BIN" env update -p "$ENV_DIR" -f Karabo-Pipeline/karabo_env.yml

echo "Installing Karabo-Pipeline..."
"$ENV_DIR/bin/pip" install -e Karabo-Pipeline

# Run the data generation script
echo "Running data generation..."
# We assume the user runs benchopt from the benchmark root directory
# so benchmark_utils/generate_radio_data.py is accessible
$ENV_DIR/bin/python benchmark_utils/generate_radio_data.py
