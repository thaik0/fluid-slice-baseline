#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=slice-coarse-repeat
#SBATCH --output=slice-coarse-repeat-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${COARSE_REPLICATION_ROOT:-$PWD/runs/coarse-replicate-${SLURM_JOB_ID:-local}}"
if [[ "${COARSE_REPLICATION_DRY_RUN:-0}" == 1 ]]; then
  echo "seeds=43,44 sequential=true gpus=1 max_job_minutes=15 max_job_gpu_hours=0.25 out=$OUT"
  exit 0
fi
: "${GPU_ACCOUNT:?Set explicit GPU_ACCOUNT and pass the same --account to sbatch}"
: "${COARSE_DATA:?Set COARSE_DATA to existing prepared96 data directory}"
if [[ "${SLURM_JOB_ACCOUNT:?}" != "$GPU_ACCOUNT" ]]; then echo 'Submitted account differs from GPU_ACCOUNT' >&2;exit 2;fi
if [[ -e "$OUT" ]]; then echo 'Replication output already exists; choose a fresh root' >&2;exit 2;fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
"$PYTHON_BIN" -c 'import json,os; from pathlib import Path; from data import digest; p=Path(os.environ["COARSE_DATA"]); m=json.loads((p/"manifest.json").read_text()); n=json.loads((p/"normalization.json").read_text()); assert m["shape"]==[96,96] and m["data_kind"]=="fourier_coarsened_prepared_planes" and n["manifest_sha256"]==digest(m)'
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$OUT/slurm-job-start.txt"
COARSE_OUTPUT="$OUT" "$PYTHON_BIN" - <<'PYBUDGET'
import json,os
from pathlib import Path
budget=dict(account=os.environ['GPU_ACCOUNT'],job_id=os.environ['SLURM_JOB_ID'],gpus=1,seeds=[43,44],execution='sequential within one allocation',max_wall_minutes=15,max_total_gpu_hours=.25,estimated_max_standard_shared_gpu_node_hours=.0625,charge_note='Resource cap for the entire job, not per seed. Actual charge depends on QOS and footprint; overrun has zero allocation charge.',actual_usage='Retrieve sacct elapsed/AllocTRES and Iris charges')
(Path(os.environ['COARSE_OUTPUT'])/'budget.json').write_text(json.dumps(budget,indent=2))
PYBUDGET
for TRAIN_SEED in 43 44; do
  RUN="$OUT/seed-$TRAIN_SEED"
  srun "$PYTHON_BIN" run.py train --data "$COARSE_DATA" --out "$RUN/train" --device cuda --steps 5000 --width 32 --patch 96 --batch 8 --seed "$TRAIN_SEED" --checkpoint-every 1000 --model-norm group --padding circular --mean-mode implicit
  srun "$PYTHON_BIN" evaluate_coarse.py --data "$COARSE_DATA" --checkpoint "$RUN/train/checkpoint.pt" --out "$RUN/eval" --device cuda --samples 8 --steps 64 --seed 4242
done
