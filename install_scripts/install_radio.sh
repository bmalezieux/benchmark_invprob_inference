#!/bin/bash
set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

load_singularity_module() {
    if command -v module &> /dev/null; then
        log "Loading module 'singularity'..."
        if module load singularity; then
            return 0
        fi
        log "Warning: failed to load module 'singularity'."
    fi
    return 1
}

# 1. Check dependencies
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    load_singularity_module || true
fi
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    log "Error: apptainer or singularity could not be found (even after module load)."
    exit 1
fi

# 2. Setup directories
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

log "Repository root: $REPO_DIR"

# Benchopt may pass CONDA_PREFIX as the first positional argument.
# Consume it so it is not forwarded to the submission module.
BENCHOPT_CONDA_PREFIX="${1:-}"
if [[ -n "$BENCHOPT_CONDA_PREFIX" && -d "$BENCHOPT_CONDA_PREFIX" ]]; then
    shift
fi

# Use the Python interpreter from the current execution environment
# (e.g., `uv run ...`), not from the passed CONDA_PREFIX argument.
if command -v python &> /dev/null; then
    PYTHON_CMD="$(command -v python)"
else
    PYTHON_CMD="$(command -v python3)"
fi
log "Using Python interpreter: $PYTHON_CMD"

# 3. Check/Pull Image
IMAGE_NAME="karabo.sif"
IMAGE_PATH="$REPO_DIR/$IMAGE_NAME"
IMAGE_URI="${KARABO_IMAGE_URI:-oras://ghcr.io/bmalezieux/karabo-image:latest}"

if [ ! -f "$IMAGE_PATH" ]; then
    log "Pulling singularity image from $IMAGE_URI to $IMAGE_PATH..."
    if command -v apptainer &> /dev/null; then
         apptainer pull "$IMAGE_PATH" "$IMAGE_URI"
    else
         singularity pull "$IMAGE_PATH" "$IMAGE_URI"
    fi
else
    log "Image found at $IMAGE_PATH."
fi

# 4. Run submission script
log "Running submission script..."

ARGS=("$@")

# Check if Slurm is available.
if ! command -v sbatch &> /dev/null; then
    log "Slurm command 'sbatch' not found. Forcing local execution."
    ARGS+=("--local")
else
    # Jean Zay flow: copy image to the singularity-allowed directory.
    if [[ -n "${SINGULARITY_ALLOWED_DIR:-}" ]]; then
        load_singularity_module || true
        TARGET_IMAGE="${SINGULARITY_ALLOWED_DIR%/}/${IMAGE_NAME}"
        log "Slurm detected with SINGULARITY_ALLOWED_DIR='$SINGULARITY_ALLOWED_DIR'."
        log "Ensuring image is available at $TARGET_IMAGE"

        mkdir -p "$SINGULARITY_ALLOWED_DIR"
        if [[ ! -f "$TARGET_IMAGE" ]]; then
            if command -v idrcpy &> /dev/null; then
                idrcpy "$IMAGE_PATH" "$TARGET_IMAGE"
            elif command -v idrcontmgr &> /dev/null; then
                idrcontmgr cp "$IMAGE_PATH"
            else
                cp "$IMAGE_PATH" "$TARGET_IMAGE"
            fi
            if [[ ! -f "$TARGET_IMAGE" ]]; then
                log "Error: expected copied image at $TARGET_IMAGE, but it was not found."
                exit 1
            fi
        else
            log "Image already present in allowed directory."
        fi
        ARGS+=("--image-path" "$TARGET_IMAGE")
    fi
fi

log "Arguments for submission script: ${ARGS[*]}"

# Pass all arguments to the Python module.
"$PYTHON_CMD" -m benchmark_utils.submit_job \
    --config "$SCRIPT_DIR/config_slurm.yaml" \
    "${ARGS[@]}"
