#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=slice-diagnostics
#SBATCH --output=slice-diagnostics-%j.out
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
cd "${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
: "${SLICE_RUN:?Set SLICE_RUN to existing pilot directory}"
OUT="${DIAGNOSTICS_OUT:-$SLICE_RUN/diagnostics-${SLURM_JOB_ID}}"
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
srun "$PYTHON_BIN" -m pytest -q
srun "$PYTHON_BIN" diagnose.py --checkpoint "$SLICE_RUN/train/checkpoint.pt" --data "$SLICE_RUN/data" --out "$OUT" --device cuda --samples 8 --probe-batches 16
