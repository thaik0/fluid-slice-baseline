import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from model import FlowNet,validate_domain,tiled
from mean_statistics import distribution,wasserstein_1d


def test_mean_branch_preserves_backbone_and_training_rng():
    torch.manual_seed(9);control=FlowNet(3,8,padding='circular');after=torch.get_rng_state()
    torch.manual_seed(9);branch=FlowNet(3,8,padding='circular',mean_mode='learned')
    assert torch.equal(torch.get_rng_state(),after)
    for name,p in control.state_dict().items():torch.testing.assert_close(p,branch.state_dict()[name],rtol=0,atol=0)
    restored=FlowNet(**dict(channels=3,width=8,normalization='group',padding='circular'))
    restored.load_state_dict(control.state_dict(),strict=True)


def test_predicted_mean_is_head_and_ordinary_mse_decomposes():
    torch.set_num_threads(1);net=FlowNet(3,8,padding='circular',mean_mode='learned')
    x=torch.randn(2,3,16,16);t=torch.tensor([.2,.8]);truth=torch.randn_like(x)
    pred=net(x,t);avg=net.mean_head(torch.cat([x.mean((-2,-1)),t[:,None]],1))
    torch.testing.assert_close(pred.mean((-2,-1)),avg,rtol=1e-5,atol=1e-6)
    loss=(pred-truth).square().mean()
    mean_loss=(avg-truth.mean((-2,-1))).square().mean()
    centered_loss=((pred-pred.mean((-2,-1),keepdim=True))-(truth-truth.mean((-2,-1),keepdim=True))).square().mean()
    torch.testing.assert_close(loss,centered_loss+mean_loss)
    loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    with pytest.raises(ValueError,match='full square'):validate_domain('zeros',64,(384,384),'learned')
    with pytest.raises(ValueError,match='full square'):tiled(net,x,.2,8)


def test_mean_distribution_detects_collapse_at_correct_energy():
    # Correct mean-flow energy does not imply the correct distribution.
    real=np.array([[-1.,0],[1.,0]]);collapsed=np.array([[1.,0],[1.,0]])
    assert distribution(real)['mean_flow_energy']==distribution(collapsed)['mean_flow_energy']
    assert distribution(real)['std'][0]==1 and distribution(collapsed)['std'][0]==0
    assert wasserstein_1d(real[:,0],collapsed[:,0])==1
    assert wasserstein_1d(np.array([0.,1]),np.array([0.,0.,1.,1.]))==0


@pytest.mark.parametrize('task',range(6))
def test_mean_branch_launch_pairs(task):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,SLURM_SUBMIT_DIR=str(root),SLURM_ARRAY_TASK_ID=str(task),SLURM_ARRAY_JOB_ID='test',SLICE_RUN='/tmp/pilot',MEAN_DRY_RUN='1')
    r=subprocess.run(['bash','scripts/mean_branch.slurm'],cwd=root,env=env,text=True,capture_output=True,check=True)
    assert f"seed={42+task//2} mean_mode={['implicit','learned'][task%2]}" in r.stdout
