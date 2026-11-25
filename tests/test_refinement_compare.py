import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import compare_refinement as comp
from model import FlowNet
from prepare_coarse import prepare,file_hash
from test_coarse import fixture


def report(steps):
    return dict(args=dict(steps=steps),samples=[dict(validation_index=0,seed=4242,max_normalized_low_mode_error=1e-6,seconds=.1)],fine_manifest_sha256='fine',coarse_manifest_sha256='coarse',model_config=dict(width=4),reference_residual_spectrum=[0.]*48+[1.]*225,generated_residual_spectrum=[0.]*48+[.5]*225,checkpoint_sha256='hash')


def test_summary_refuses_unmatched_or_incomplete_comparisons():
    reports=[(u,s,report(s)) for u in comp.UPDATES for s in comp.SAMPLING_STEPS]
    result=comp.summarize(reports,96)
    assert len(result['rows'])==6
    assert all(r['spectrum_change_relative_to_real_residual']==0 for r in result['sampling_changes'])
    assert all(r['residual_energy_ratio']==.5 and r['residual_spectrum_relative_l1']==.5 for r in result['rows'])
    assert result['rows'][0]['bands'][0]==dict(first_shell=48,last_shell=63,energy_ratio=.5)
    with pytest.raises(ValueError,match='Incomplete'):comp.summarize(reports[:-1],96)
    bad=copy.deepcopy(reports);bad[-1][2]['samples'][0]['seed']=4243
    with pytest.raises(ValueError,match='Unmatched'):comp.summarize(bad,96)
    bad=copy.deepcopy(reports);bad[-1][2]['samples'][0]['max_normalized_low_mode_error']=.1
    with pytest.raises(ValueError,match='coarse modes'):comp.summarize(bad,96)


def test_evaluation_only_six_arm_smoke_and_checkpoint_integrity(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    source=tmp_path/'source';fixture(source);coarse=tmp_path/'coarse';prepare(source,coarse,64)
    run=tmp_path/'run';(run/'train').mkdir(parents=True)
    m=json.loads((source/'manifest.json').read_text());cm=json.loads((coarse/'manifest.json').read_text());norm=json.loads((source/'normalization.json').read_text())
    config=dict(width=4,channels=3,context_channels=3,normalization='group',padding='circular')
    model=FlowNet(**config)
    monkeypatch.setattr(comp,'UPDATES',(1,2,3));monkeypatch.setattr(comp,'SAMPLING_STEPS',(1,2))
    for u in comp.UPDATES:
        torch.save(dict(kind='real_coarse_conditional_residual',history=[dict(step=u)],manifest=m,coarse_manifest=cm,normalization=norm,config=config,args=dict(seed=42),source_hashes={},model=model.state_dict()),run/'train'/f'checkpoint-{u}.pt')
    paths=list((run/'train').iterdir());before={p:file_hash(p) for p in paths}
    out=tmp_path/'reports'
    monkeypatch.setattr(sys,'argv',['compare_refinement.py','evaluate','--data',str(source),'--coarse',str(coarse),'--run',str(run),'--out',str(out),'--samples','1'])
    comp.main()
    result=json.loads((out/'comparison.json').read_text())
    assert len(result['rows'])==6 and result['matched_contexts_and_noise']==[[0,4242]]
    assert before=={p:file_hash(p) for p in paths}
    assert not (source/'test.npy').exists()
    bad=run/'train'/'checkpoint-2.pt';ck=torch.load(bad,weights_only=False);ck['args']['seed']=43;torch.save(ck,bad)
    with pytest.raises(ValueError,match='same training'):comp.preflight(source,coarse,run,1)


def test_launcher_budget_and_account_fail_before_work():
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,REFINEMENT_COMPARE_DRY_RUN='1')
    r=subprocess.run(['bash','scripts/refinement_compare.slurm'],cwd=root,env=env,capture_output=True,text=True,check=True)
    assert 'evaluation_only=true' in r.stdout and 'max_job_minutes=5' in r.stdout
    assert 'checkpoints=1000,3000,5000 euler_steps=64,128' in r.stdout
    env.update(REFINEMENT_COMPARE_DRY_RUN='0',GPU_ACCOUNT='a',SLURM_JOB_ACCOUNT='b',FINE_DATA='none',COARSE_DATA='none',REFINEMENT_SOURCE_RUN='none')
    r=subprocess.run(['bash','scripts/refinement_compare.slurm'],cwd=root,env=env,capture_output=True,text=True)
    assert r.returncode==2 and 'Account mismatch' in r.stderr
