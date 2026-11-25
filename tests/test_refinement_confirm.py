import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from data import digest
from model import FlowNet
from prepare_coarse import file_hash
from refinement import low
from confirm_refinement import select_contexts,conditioned_sample,plan,evaluate,stats
from argparse import Namespace


def test_selection_excludes_previous_contexts_and_deranges_donors():
    ids,excluded,donors=select_contexts(128,32,8)
    assert len(set(ids))==32 and not set(ids)&set(excluded)
    assert set(np.linspace(0,127,8,dtype=int)).issubset(excluded)
    assert set(range(8)).issubset(excluded)
    assert len(donors)==8 and set(donors)==set(donors.values())
    assert all(i!=d and ids[i]//4!=ids[d]//4 for i,d in donors.items())
    assert select_contexts(128,32,8)==(ids,excluded,donors)
    with pytest.raises(ValueError):select_contexts(16,32,8)


def test_ablation_changes_only_conditioning_not_carrier_or_noise():
    class UsesContext(torch.nn.Module):
        def forward(self,x,t,context):
            # A coarse spatial feature controls fine velocity amplitude.
            n=x.shape[-1];k=torch.arange(n,device=x.device)
            wave=torch.cos(2*torch.pi*7*k/n)[None,None,None,:]
            return context.mean((-2,-1),keepdim=True)*wave.expand_as(x)
    class IgnoresContext(torch.nn.Module):
        def forward(self,x,t,context):return x*.1
    torch.set_num_threads(1);torch.manual_seed(1)
    carrier=low(torch.randn(1,3,32,32),8);noise=torch.randn_like(carrier)
    a=conditioned_sample(UsesContext(),carrier,torch.ones_like(carrier),noise,8,3)
    b=conditioned_sample(UsesContext(),carrier,torch.zeros_like(carrier),noise,8,3)
    assert (a-b).square().mean()>.1
    torch.testing.assert_close(low(a,8),carrier,atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(low(b,8),carrier,atol=2e-6,rtol=1e-6)
    c=conditioned_sample(IgnoresContext(),carrier,carrier,noise,8,3)
    d=conditioned_sample(IgnoresContext(),carrier,torch.zeros_like(carrier),noise,8,3)
    torch.testing.assert_close(c,d,atol=0,rtol=0)


def test_confirmation_pipeline_without_training_or_reserved_test(tmp_path):
    torch.set_num_threads(1);rng=np.random.default_rng(22)
    fine=tmp_path/'fine';coarse=tmp_path/'coarse';run=tmp_path/'run'
    fine.mkdir();coarse.mkdir();(run/'train').mkdir(parents=True)
    n=32;c=8
    m=dict(shape=[n,n],channels=['u','v','w'],spacing=.1,slice_indices=[0],snapshots=[dict(split='train',time=0)]+[dict(split='val',time=i+10) for i in range(32)])
    norm=dict(mean=[0.,0.,0.],std=[1.,1.,1.],manifest_sha256=digest(m))
    (fine/'manifest.json').write_text(json.dumps(m));(fine/'normalization.json').write_text(json.dumps(norm))
    for split,count in [('train',1),('val',32)]:np.save(fine/f'{split}.npy',rng.normal(size=(count,3,n,n)).astype('float32'))
    cm=dict(m,shape=[c,c],spacing=.4,data_kind='fourier_coarsened_prepared_planes',coarsening=dict(source_manifest_sha256=digest(m),source_normalization_sha256=file_hash(fine/'normalization.json'),source_array_sha256={s:file_hash(fine/f'{s}.npy') for s in ['train','val']}))
    cn=dict(norm,manifest_sha256=digest(cm));(coarse/'manifest.json').write_text(json.dumps(cm));(coarse/'normalization.json').write_text(json.dumps(cn))
    for split in ['train','val']:
        raw=torch.tensor(np.load(fine/f'{split}.npy'));arr=low(raw,c).numpy()[...,::4,::4];np.save(coarse/f'{split}.npy',arr)
    config=dict(channels=3,width=4,normalization='group',padding='circular',context_channels=3)
    path=run/'train/checkpoint-3000.pt'
    torch.save(dict(kind='real_coarse_conditional_residual',history=[dict(step=3000)],args=dict(seed=42),config=config,model=FlowNet(**config).state_dict(),manifest=m,coarse_manifest=cm,normalization=norm),path)
    before=file_hash(path);p=plan(fine,coarse,run,samples=4,shuffle_samples=2,steps=2)
    evaluate(Namespace(data=str(fine),coarse=str(coarse),out=str(tmp_path/'reports'),device='cpu'),p)
    r=json.loads((tmp_path/'reports/confirmation.json').read_text())
    assert len(r['samples'])==4 and r['conditioning']['contexts']==2
    assert file_hash(path)==before and not (fine/'test.npy').exists()
    assert all(s['max_normalized_low_mode_error']<1e-5 for s in r['samples'])
    np.testing.assert_allclose(r['all_fresh']['reference_residual_spectrum'],np.mean([s['reference_residual_spectrum'] for s in r['samples']],axis=0))
    selected=[s for s in r['samples'] if s['donor_validation_index'] is not None]
    np.testing.assert_allclose(r['matched_subset_correct']['reference_residual_spectrum'],np.mean([s['reference_residual_spectrum'] for s in selected],axis=0))
    assert r['matched_subset_correct']['reference_residual_spectrum']==r['matched_subset_shuffled']['reference_residual_spectrum']
    ck=torch.load(path,weights_only=False);ck['history'][-1]['step']=5000;torch.save(ck,path)
    with pytest.raises(ValueError,match='3000'):plan(fine,coarse,run,4,2,2)


def test_five_minute_evaluation_budget_and_account_check():
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,REFINEMENT_CONFIRM_DRY_RUN='1')
    r=subprocess.run(['bash','scripts/refinement_confirm.slurm'],cwd=root,env=env,capture_output=True,text=True,check=True)
    assert 'evaluation_only=true checkpoint=3000 euler_steps=64 fresh_contexts=32 shuffled_subset=8' in r.stdout
    assert 'max_job_minutes=5 max_job_gpu_hours=0.0833333333' in r.stdout
    env.update(REFINEMENT_CONFIRM_DRY_RUN='0',GPU_ACCOUNT='a',SLURM_JOB_ACCOUNT='b',FINE_DATA='none',COARSE_DATA='none',REFINEMENT_SOURCE_RUN='none')
    r=subprocess.run(['bash','scripts/refinement_confirm.slurm'],cwd=root,env=env,capture_output=True,text=True)
    assert r.returncode==2 and 'Account mismatch' in r.stderr
