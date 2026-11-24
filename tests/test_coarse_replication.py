import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from collect_coarse import collect


def test_coarse_replication_one_allocation_dry_run():
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,SLURM_SUBMIT_DIR=str(root),SLURM_JOB_ID='test',COARSE_REPLICATION_DRY_RUN='1')
    r=subprocess.run(['bash','scripts/replicate_coarse.slurm'],cwd=root,env=env,capture_output=True,text=True,check=True)
    assert 'seeds=43,44 sequential=true gpus=1 max_job_minutes=15 max_job_gpu_hours=0.25' in r.stdout


def fixture(root):
    pilot=root/'pilot';rep=root/'repeat'
    for seed,path in [(42,pilot),(43,rep/'seed-43'),(44,rep/'seed-44')]:
        (path/'train').mkdir(parents=True);(path/'eval').mkdir()
        t=dict(args=dict(seed=seed,patch=96,steps=5000,width=32,batch=8,lr=.0001),parameters=473187,seconds=1.,peak_cuda_bytes=1,history=[dict(val_loss_t05=.3)])
        metrics=dict(spectrum=[0.]+[1.]*16,mean=[0.,0.,0.])
        d=dict(model_config=dict(channels=3,width=32,normalization='group',padding='circular',mean_mode='implicit'),manifest_sha256='same',reference=metrics,args=dict(steps=64),samples=[dict(seed=s,metrics=metrics) for s in range(4242,4250)])
        (path/'train/training.json').write_text(json.dumps(t));(path/'eval/coarse-evaluation.json').write_text(json.dumps(d))
    return pilot,rep


def test_coarse_collector_checks_comparability_and_completeness(tmp_path):
    p,r=fixture(tmp_path);result=collect(p,r)
    assert len(result['runs'])==3
    np.testing.assert_allclose(result['shells1_16_across_training_seeds']['mean'],1)
    path=r/'seed-44/eval/coarse-evaluation.json';d=json.loads(path.read_text());d['manifest_sha256']='different';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='differs'):collect(p,r)
    path.unlink()
    with pytest.raises(FileNotFoundError):collect(p,r)
