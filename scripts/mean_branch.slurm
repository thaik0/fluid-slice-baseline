#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --array=0-5%1
#SBATCH --job-name=slice-mean
#SBATCH --output=slice-mean-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${SLICE_RUN:?Set SLICE_RUN to existing pilot directory}"
TASK="${SLURM_ARRAY_TASK_ID:?Missing array index}"
if [[ ! "$TASK" =~ ^[0-5]$ ]]; then echo 'Expected task 0 through 5' >&2; exit 2; fi
TRAIN_SEED=$((42 + TASK / 2))
if (( TASK % 2 == 0 )); then MEAN_MODE=implicit; else MEAN_MODE=learned; fi
OUT="${MEAN_RUN_ROOT:-$SLICE_RUN/mean-branch-${SLURM_ARRAY_JOB_ID:-local}}/$MEAN_MODE-seed-$TRAIN_SEED"
if [[ "${MEAN_DRY_RUN:-0}" == 1 ]]; then
  echo "seed=$TRAIN_SEED mean_mode=$MEAN_MODE out=$OUT"; exit 0
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
if [[ -e "$OUT/train" ]]; then echo "Existing $OUT/train; use a fresh MEAN_RUN_ROOT" >&2; exit 2; fi
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
srun "$PYTHON_BIN" run.py train --data "$SLICE_RUN/data" --out "$OUT/train" --device cuda --steps 5000 --width 32 --patch 384 --batch 8 --seed "$TRAIN_SEED" --checkpoint-every 1000 --model-norm group --padding circular --mean-mode "$MEAN_MODE"
srun "$PYTHON_BIN" diagnose.py --checkpoint "$OUT/train/checkpoint.pt" --data "$SLICE_RUN/data" --out "$OUT/diagnostics" --device cuda --samples 4 --probe-batches 8 --mean-samples 64
