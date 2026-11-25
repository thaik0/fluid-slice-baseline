#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --job-name=slice-ref-check
#SBATCH --output=slice-ref-check-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${REFINEMENT_CONFIRM_DRY_RUN:-0}" == 1 ]]; then
  echo 'evaluation_only=true checkpoint=3000 euler_steps=64 fresh_contexts=32 shuffled_subset=8 gpus=1 max_job_minutes=5 max_job_gpu_hours=0.0833333333'
  exit 0
fi
: "${GPU_ACCOUNT:?Set GPU_ACCOUNT and pass the same --account to sbatch}"
: "${FINE_DATA:?Set FINE_DATA}"
: "${COARSE_DATA:?Set COARSE_DATA}"
: "${REFINEMENT_SOURCE_RUN:?Set REFINEMENT_SOURCE_RUN to completed refinement pilot}"
if [[ "${SLURM_JOB_ACCOUNT:?}" != "$GPU_ACCOUNT" ]]; then echo 'Account mismatch' >&2;exit 2;fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
OUT="${REFINEMENT_CONFIRM_ROOT:-$PWD/runs/refinement-confirm-${SLURM_JOB_ID}}"
if [[ -e "$OUT" ]]; then echo 'Choose a fresh output root' >&2;exit 2;fi
# Check the completed CPU preflight against the actual checkpoint/data/selection.
"$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
from confirm_refinement import plan
r=json.loads(Path(os.environ.get('REFINEMENT_CONFIRM_CHECK','runs/pilot/refinement-confirm-check.json')).read_text())
assert r['geometry_status']=='passed'
p=plan(os.environ['FINE_DATA'],os.environ['COARSE_DATA'],os.environ['REFINEMENT_SOURCE_RUN'])
for key in p:assert p[key]==r[key],f'Stale preflight: {key}'
PY
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$OUT/slurm-job.txt"
REFINEMENT_CONFIRM_OUTPUT="$OUT" "$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
(Path(os.environ['REFINEMENT_CONFIRM_OUTPUT'])/'budget.json').write_text(json.dumps(dict(account=os.environ['GPU_ACCOUNT'],job_id=os.environ['SLURM_JOB_ID'],gpus=1,max_wall_minutes=5,max_gpu_hours=5/60,estimated_max_standard_shared_gpu_node_hours=5/60/4,scope='saved3000/Euler64:32 new validation planes/noises plus8 matched shuffled-conditioning samples; no training',retry='none'),indent=2))
PY
srun "$PYTHON_BIN" confirm_refinement.py evaluate --data "$FINE_DATA" --coarse "$COARSE_DATA" --run "$REFINEMENT_SOURCE_RUN" --out "$OUT/reports" --device cuda --samples 32 --shuffle-samples 8 --steps 64
