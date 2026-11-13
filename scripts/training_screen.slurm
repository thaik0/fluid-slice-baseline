#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=0-4%1
#SBATCH --job-name=slice-screen
#SBATCH --output=slice-screen-%A_%a.out
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
cd "${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
: "${SLICE_RUN:?Set SLICE_RUN to existing pilot directory}"
TRAIN_SEED="${TRAIN_SEED:-42}"
# First five arms screen duration/capacity/context; arm 5 is optional.
case "${SLURM_ARRAY_TASK_ID}" in
  0) NAME=baseline; WIDTH=32; PATCH=64; STEPS=1000 ;;
  1) NAME=longer; WIDTH=32; PATCH=64; STEPS=5000 ;;
  2) NAME=wider; WIDTH=64; PATCH=64; STEPS=5000 ;;
  3) NAME=context128; WIDTH=32; PATCH=128; STEPS=5000 ;;
  4) NAME=wholeplane; WIDTH=32; PATCH=384; STEPS=5000 ;;
  5) NAME=large; WIDTH=192; PATCH=64; STEPS=5000 ;;
  *) exit 2 ;;
esac
OUT="${SCREEN_ROOT:-$SLICE_RUN/screen-${SLURM_ARRAY_JOB_ID}}/$NAME-seed-$TRAIN_SEED"
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
srun "$PYTHON_BIN" run.py train --data "$SLICE_RUN/data" --out "$OUT/train" --device cuda --steps "$STEPS" --width "$WIDTH" --patch "$PATCH" --batch 8 --seed "$TRAIN_SEED" --checkpoint-every 1000
srun "$PYTHON_BIN" diagnose.py --checkpoint "$OUT/train/checkpoint.pt" --data "$SLICE_RUN/data" --out "$OUT/diagnostics" --device cuda --samples 4 --probe-batches 8
