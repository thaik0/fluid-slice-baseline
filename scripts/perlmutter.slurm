#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=fluid-slice
#SBATCH --output=fluid-slice-%j.out
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load pytorch/2.6.0
cd "${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${SLICE_RUN:?Set SLICE_RUN to prepared run directory}"
"$PYTHON_BIN" -m pip freeze > "$SLICE_RUN/environment.txt"
srun "$PYTHON_BIN" -m pytest -q
srun "$PYTHON_BIN" run.py train --data "$SLICE_RUN/data" --out "$SLICE_RUN/train" --device cuda --steps 1000 --width 32 --patch 64 --batch 8
srun "$PYTHON_BIN" run.py evaluate --data "$SLICE_RUN/data" --checkpoint "$SLICE_RUN/train/checkpoint.pt" --out "$SLICE_RUN/eval" --device cuda --steps 32 --samples 8 --seed 4242
