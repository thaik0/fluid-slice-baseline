#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=slice-coarse
#SBATCH --output=slice-coarse-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${GPU_ACCOUNT:?Export the funded GPU account and pass the same --account to sbatch}"
: "${COARSE_DATA:?Set COARSE_DATA to prepared96 data directory}"
if [[ "${SLURM_JOB_ACCOUNT:?}" != "$GPU_ACCOUNT" ]]; then echo 'Submitted account differs from explicit GPU_ACCOUNT' >&2;exit 2;fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
OUT="${COARSE_RUN_ROOT:-$PWD/runs/coarse-pilot-${SLURM_JOB_ID}}"
if [[ -e "$OUT/train" ]]; then echo 'Existing run; choose a fresh output root' >&2;exit 2;fi
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -c 'import json,os; from pathlib import Path; m=json.loads((Path(os.environ["COARSE_DATA"])/"manifest.json").read_text()); assert m["shape"]==[96,96] and m["data_kind"]=="fourier_coarsened_prepared_planes"'
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
COARSE_OUTPUT="$OUT" "$PYTHON_BIN" - <<'PYBUDGET'
import json,os
from pathlib import Path
budget=dict(account=os.environ['GPU_ACCOUNT'],job_id=os.environ['SLURM_JOB_ID'],gpus=1,max_wall_minutes=15,max_gpu_hours=.25,estimated_max_standard_shared_gpu_node_hours=.0625,scope='one training seed,5000 updates,8 fixed64 samples',actual_usage='Retrieve sacct elapsed/AllocTRES; allocation charges may depend on resource footprint')
(Path(os.environ['COARSE_OUTPUT'])/'budget.json').write_text(json.dumps(budget,indent=2))
PYBUDGET
srun "$PYTHON_BIN" run.py train --data "$COARSE_DATA" --out "$OUT/train" --device cuda --steps 5000 --width 32 --patch 96 --batch 8 --seed 42 --checkpoint-every 1000 --model-norm group --padding circular --mean-mode implicit
srun "$PYTHON_BIN" evaluate_coarse.py --data "$COARSE_DATA" --checkpoint "$OUT/train/checkpoint.pt" --out "$OUT/eval" --device cuda --samples 8 --steps 64
