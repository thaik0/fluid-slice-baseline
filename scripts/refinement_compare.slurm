#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --job-name=slice-ref-eval
#SBATCH --output=slice-ref-eval-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${REFINEMENT_COMPARE_DRY_RUN:-0}" == 1 ]]; then
  echo 'evaluation_only=true checkpoints=1000,3000,5000 euler_steps=64,128 samples_per_arm=8 gpus=1 max_job_minutes=5 max_job_gpu_hours=0.0833333333'
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
OUT="${REFINEMENT_COMPARE_ROOT:-$PWD/runs/refinement-compare-${SLURM_JOB_ID}}"
if [[ -e "$OUT" ]]; then echo 'Choose a fresh output root' >&2;exit 2;fi
# Require the separately completed CPU geometry/checkpoint preflight.
"$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
from data import digest
from prepare_coarse import file_hash
r=json.loads(Path(os.environ.get('REFINEMENT_COMPARE_CHECK','runs/pilot/refinement-compare-check.json')).read_text())
assert r['status']=='passed' and r['geometry_status']=='passed' and r['samples']==8
for env,key in [('FINE_DATA','fine_manifest_sha256'),('COARSE_DATA','coarse_manifest_sha256')]:
 assert digest(json.loads((Path(os.environ[env])/'manifest.json').read_text()))==r[key]
for u in [1000,3000,5000]:
 p=Path(os.environ['REFINEMENT_SOURCE_RUN'])/'train'/f'checkpoint-{u}.pt'
 assert file_hash(p)==r['checkpoints'][str(u)]['sha256']
PY
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$OUT/slurm-job.txt"
REFINEMENT_COMPARE_OUTPUT="$OUT" "$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
(Path(os.environ['REFINEMENT_COMPARE_OUTPUT'])/'budget.json').write_text(json.dumps(dict(account=os.environ['GPU_ACCOUNT'],job_id=os.environ['SLURM_JOB_ID'],gpus=1,max_wall_minutes=5,max_gpu_hours=5/60,estimated_max_standard_shared_gpu_node_hours=5/60/4,scope='evaluate six saved-checkpoint/sampler arms,8 matched contexts/noise seeds per arm; no training',retry='none'),indent=2))
PY
srun "$PYTHON_BIN" compare_refinement.py evaluate --data "$FINE_DATA" --coarse "$COARSE_DATA" --run "$REFINEMENT_SOURCE_RUN" --out "$OUT/reports" --device cuda --samples 8
