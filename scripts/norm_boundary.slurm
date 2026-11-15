#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --array=0-11%1
#SBATCH --job-name=slice-factorial
#SBATCH --output=slice-factorial-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${SLICE_RUN:?Set SLICE_RUN to existing pilot directory}"
TASK="${SLURM_ARRAY_TASK_ID:?Missing array index}"
if [[ ! "$TASK" =~ ^([0-9]|1[01])$ ]]; then echo 'Expected task 0 through 11' >&2; exit 2; fi
TRAIN_SEED=$((42 + TASK / 4))
case "$((TASK % 4))" in
  0) MODEL_NORM=group; PADDING=zeros ;;
  1) MODEL_NORM=group; PADDING=circular ;;
  2) MODEL_NORM=none; PADDING=zeros ;;
  3) MODEL_NORM=none; PADDING=circular ;;
esac
OUT="${FACTORIAL_ROOT:-$SLICE_RUN/factorial-${SLURM_ARRAY_JOB_ID:-local}}/$MODEL_NORM-$PADDING-seed-$TRAIN_SEED"
if [[ "${FACTORIAL_DRY_RUN:-0}" == 1 ]]; then
  echo "seed=$TRAIN_SEED norm=$MODEL_NORM padding=$PADDING out=$OUT"
  exit 0
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
if [[ -e "$OUT/train" ]]; then echo "Existing run directory: $OUT/train; use a fresh FACTORIAL_ROOT" >&2; exit 2; fi
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
srun "$PYTHON_BIN" run.py train --data "$SLICE_RUN/data" --out "$OUT/train" --device cuda --steps 5000 --width 32 --patch 384 --batch 8 --seed "$TRAIN_SEED" --checkpoint-every 1000 --model-norm "$MODEL_NORM" --padding "$PADDING"
srun "$PYTHON_BIN" diagnose.py --checkpoint "$OUT/train/checkpoint.pt" --data "$SLICE_RUN/data" --out "$OUT/diagnostics" --device cuda --samples 4 --probe-batches 8
