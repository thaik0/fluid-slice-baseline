import os
from pathlib import Path
import subprocess
import pytest
import torch
from model import FlowNet, validate_domain


def test_legacy_defaults_and_state_dict(tmp_path):
    old=FlowNet(channels=3,width=8)
    path=tmp_path/'old.pt'; torch.save(dict(config=dict(channels=3,width=8),model=old.state_dict()),path)
    ck=torch.load(path,weights_only=True); restored=FlowNet(**ck['config'])
    restored.load_state_dict(ck['model'],strict=True)
    explicit=FlowNet(3,8,normalization='group',padding='zeros'); explicit.load_state_dict(ck['model'])
    x=torch.randn(2,3,16,16)
    torch.testing.assert_close(restored(x,.4),explicit(x,.4),rtol=0,atol=0)


@pytest.mark.parametrize('norm',['group','none'])
@pytest.mark.parametrize('padding',['zeros','circular'])
def test_all_variants_have_finite_gradients(norm,padding):
    torch.set_num_threads(1); net=FlowNet(3,8,norm,padding)
    y=net(torch.randn(2,3,16,16),torch.tensor([.2,.8])); y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_periodic_convolution_wraps_and_downsampling_phase_is_documented():
    # Every spatial conv must wrap, not just the first layer.
    net=FlowNet(3,8,padding='circular').eval()
    assert all(m.padding_mode=='circular' for m in net.modules() if isinstance(m,torch.nn.Conv2d) and m.kernel_size==(3,3))
    conv=net.a.net[0]
    with torch.no_grad():
        conv.weight.zero_(); conv.bias.zero_(); conv.weight[0,0,1,0]=1
    x=torch.zeros(1,4,16,16); x[0,0,3,-1]=7
    assert conv(x)[0,0,3,0]==7
    # Two pooling layers permit exact roll covariance on the 4-cell sublattice.
    torch.manual_seed(7); net=FlowNet(3,8,padding='circular').eval(); x=torch.randn(1,3,16,16)
    torch.testing.assert_close(net(torch.roll(x,(4,8),(-2,-1)),.3),torch.roll(net(x,.3),(4,8),(-2,-1)),rtol=1e-4,atol=1e-5)


def test_domain_guard_and_shared_convolution_initialization():
    validate_domain('circular',384,(384,384))
    with pytest.raises(ValueError,match='full square'):validate_domain('circular',64,(384,384))
    models=[]
    for norm,padding in [('group','zeros'),('group','circular'),('none','zeros'),('none','circular')]:
        torch.manual_seed(42); models.append(FlowNet(3,8,norm,padding))
    weights=[[m.weight.detach() for m in net.modules() if isinstance(m,torch.nn.Conv2d)] for net in models]
    for other in weights[1:]:
        for a,b in zip(weights[0],other):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('task',range(12))
def test_factorial_dispatch(task):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,SLURM_SUBMIT_DIR=str(root),SLURM_ARRAY_TASK_ID=str(task),SLURM_ARRAY_JOB_ID='test',SLICE_RUN='/tmp/pilot',FACTORIAL_DRY_RUN='1')
    r=subprocess.run(['bash','scripts/norm_boundary.slurm'],cwd=root,env=env,capture_output=True,text=True,check=True)
    norm,padding=[('group','zeros'),('group','circular'),('none','zeros'),('none','circular')][task%4]
    assert f'seed={42+task//4} norm={norm} padding={padding}' in r.stdout
