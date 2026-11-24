#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --job-name=slice-refine
#SBATCH --output=slice-refine-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${GPU_ACCOUNT:?Set GPU_ACCOUNT and pass the same --account to sbatch}"
: "${FINE_DATA:?Set FINE_DATA to original prepared384 directory}"
: "${COARSE_DATA:?Set COARSE_DATA to prepared96 directory}"
if [[ "${SLURM_JOB_ACCOUNT:?}" != "$GPU_ACCOUNT" ]]; then echo 'Account mismatch' >&2;exit 2;fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
module load "${PYTORCH_MODULE:-pytorch/2.6.0}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
OUT="${REFINEMENT_RUN_ROOT:-$PWD/runs/refinement-pilot-${SLURM_JOB_ID}}"
if [[ -e "$OUT" ]]; then echo 'Choose a fresh output root' >&2;exit 2;fi
mkdir -p "$OUT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/matplotlib}"
# Require the separately completed CPU preflight before consuming a GPU slot.
"$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
from data import digest
f=Path(os.environ['FINE_DATA']);c=Path(os.environ['COARSE_DATA'])
m=json.loads((f/'manifest.json').read_text());cm=json.loads((c/'manifest.json').read_text())
r=json.loads(Path(os.environ.get('REFINEMENT_CHECK','runs/pilot/refinement-check.json')).read_text())
assert m['shape']==[384,384] and cm['shape']==[96,96]
assert r['status']=='passed' and r['fine_manifest_sha256']==digest(m) and r['coarse_manifest_sha256']==digest(cm)
PY
"$PYTHON_BIN" -m pip freeze > "$OUT/environment.txt"
git rev-parse HEAD > "$OUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$OUT/slurm-job.txt"
REFINEMENT_OUTPUT="$OUT" "$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
(Path(os.environ['REFINEMENT_OUTPUT'])/'budget.json').write_text(json.dumps(dict(account=os.environ['GPU_ACCOUNT'],job_id=os.environ['SLURM_JOB_ID'],gpus=1,max_wall_minutes=15,max_gpu_hours=.25,estimated_max_standard_shared_gpu_node_hours=.0625,scope='one real-coarse conditional seed42,5000 updates,8 fixed64 residual samples',retry='none; inspect partial checkpoints before any further allocation'),indent=2))
PY
srun "$PYTHON_BIN" refinement.py train --data "$FINE_DATA" --coarse "$COARSE_DATA" --out "$OUT/train" --device cuda --updates 5000 --batch 8 --width 32 --seed 42
srun "$PYTHON_BIN" refinement.py evaluate --data "$FINE_DATA" --coarse "$COARSE_DATA" --checkpoint "$OUT/train/checkpoint.pt" --out "$OUT/eval" --device cuda --samples 8 --steps 64
