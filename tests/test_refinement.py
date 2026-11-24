import argparse
import json
import numpy as np
import pytest
import torch
from model import FlowNet
from prepare_coarse import downsample,prepare
from refinement import low,high,upsample,generate,check,train,evaluate
from test_coarse import fixture


def test_interpolation_phase_dc_cutoff_and_gradient():
    n=128;x=np.arange(n)*2*np.pi/n
    raw=np.broadcast_to(2+np.cos(x+.37)+.4*np.cos(40*x),(3,n,n)).copy()
    coarse=downsample(raw,64);up=upsample(coarse,n)
    expected=np.broadcast_to(2+np.cos(x+.37),raw.shape)
    np.testing.assert_allclose(up,expected,atol=1e-13)
    np.testing.assert_allclose(downsample(up,64),coarse,atol=1e-13)
    t=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
    residual=high(t,64)
    torch.testing.assert_close(low(t,64)+residual,t)
    assert low(residual,64).abs().max()<1e-13
    assert abs((low(t,64)*residual).mean().item())<1e-13
    residual.square().mean().backward()
    assert torch.isfinite(t.grad).all() and t.grad.abs().max()>0


def test_projected_sampler_preserves_context_even_for_bad_velocity():
    class Bad(torch.nn.Module):
        def forward(self,x,t,context):return torch.ones_like(x)+context+torch.randn_like(x)
    context=low(torch.randn(2,3,32,32),8)
    pred=generate(Bad(),context,torch.randn_like(context),8,5)
    torch.testing.assert_close(low(pred,8),context,atol=2e-6,rtol=1e-6)
    model=FlowNet(width=4,context_channels=3)
    with pytest.raises(ValueError,match='conditioning'):model(torch.zeros(1,3,32,32),.5)
    assert model(torch.zeros(1,3,32,32),.5,torch.zeros(1,3,32,32)).shape==(1,3,32,32)


def test_real_context_pipeline_and_source_tampering(tmp_path):
    source=tmp_path/'source';fixture(source);coarse=tmp_path/'coarse';prepare(source,coarse,64)
    report=check(source,coarse,tmp_path/'check.json')
    assert report['status']=='passed'
    a=argparse.Namespace(data=str(source),coarse=str(coarse),out=str(tmp_path/'train'),device='cpu',updates=2,batch=1,width=4,seed=42,checkpoint=None,samples=1,steps=2,command='train')
    train(a);a.checkpoint=str(tmp_path/'train/checkpoint.pt');a.out=str(tmp_path/'eval');a.command='evaluate';evaluate(a)
    r=json.loads((tmp_path/'eval/evaluation.json').read_text())
    assert r['kind']=='oracle_real_coarse_refinement' and np.isfinite(r['residual_spectrum_relative_l1'])
    assert r['samples'][0]['max_normalized_low_mode_error']<1e-5
    assert not (source/'test.npy').exists()
    np.save(source/'val.npy',np.load(source/'val.npy')+1)
    with pytest.raises(ValueError,match='differs'):check(source,coarse,tmp_path/'tampered.json')
