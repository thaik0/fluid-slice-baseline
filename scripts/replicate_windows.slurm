#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=0-5%1
#SBATCH --job-name=slice-replicate
#SBATCH --output=slice-replicate-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${SLICE_RUN:?Set SLICE_RUN to existing pilot directory}"
case "${SLURM_ARRAY_TASK_ID:?Missing array task ID}" in
  0) TRAIN_SEED=43; SCREEN_ARM=1 ;;
  1) TRAIN_SEED=43; SCREEN_ARM=3 ;;
  2) TRAIN_SEED=43; SCREEN_ARM=4 ;;
  3) TRAIN_SEED=44; SCREEN_ARM=1 ;;
  4) TRAIN_SEED=44; SCREEN_ARM=3 ;;
  5) TRAIN_SEED=44; SCREEN_ARM=4 ;;
  *) echo 'Expected array task 0 through 5' >&2; exit 2 ;;
esac
export TRAIN_SEED SCREEN_ARM
export SCREEN_ROOT="${REPLICATION_ROOT:-$SLICE_RUN/replication-${SLURM_ARRAY_JOB_ID:-local}}"
if [[ "${REPLICATION_DRY_RUN:-0}" == 1 ]]; then
  echo "seed=$TRAIN_SEED arm=$SCREEN_ARM root=$SCREEN_ROOT"
  exit 0
fi
# Reuse the exact training and evaluation settings of the seed-42 screen.
# Child #SBATCH comments are ignored: resources belong to this outer array.
exec bash scripts/training_screen.slurm
