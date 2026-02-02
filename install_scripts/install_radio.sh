#!/bin/bash
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 1. Check dependencies
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    log "Error: apptainer or singularity could not be found."
    exit 1
fi

# 2. Setup Directories
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

log "Repository root: $REPO_DIR"

# 3. Check for Karabo-Pipeline
if [ ! -d "Karabo-Pipeline" ]; then
    log "Cloning Karabo-Pipeline..."
    git clone https://github.com/aleph-group/Karabo-Pipeline.git
fi

# 4. Check/Build Image
IMAGE_NAME="karabo.sif"
IMAGE_PATH="$REPO_DIR/$IMAGE_NAME"

if [ ! -f "$IMAGE_PATH" ]; then
    log "Building singularity image $IMAGE_PATH..."
    if command -v apptainer &> /dev/null; then
         apptainer build "$IMAGE_PATH" "$SCRIPT_DIR/karabo.def"
    else
         singularity build "$IMAGE_PATH" "$SCRIPT_DIR/karabo.def"
    fi
else
    log "Image found at $IMAGE_PATH."
fi

# 5. Check Python Dependencies for Submission
if ! python -c "import submitit" &> /dev/null || ! python -c "import yaml" &> /dev/null; then
    log "Installing submitit and pyyaml for job submission..."
    pip install submitit pyyaml
fi

# 6. Run Submission Script
log "Running submission script..."

ARGS=("$@")

# Check if Slurm is available (check for sbatch)
if ! command -v sbatch &> /dev/null; then
    log "Slurm command 'sbatch' not found. Forcing local execution."
    ARGS+=("--local")
fi

echo "Arguments for submission script: ${ARGS[*]}"

# Pass all arguments to the python script
python "$SCRIPT_DIR/submit_job.py" "${ARGS[@]}"
