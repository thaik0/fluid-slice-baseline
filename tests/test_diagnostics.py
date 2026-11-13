import json
import numpy as np
import pytest
import torch
from diagnose import integrate, shift_at, time_probe, summarize
from model import sample, FlowNet


def test_shifts_live_on_physical_time_not_update_count():
    schedule=[(i,i) for i in range(32)]
    assert shift_at(schedule,.25,'all')==(8,8)
    assert shift_at(schedule,.25,'early')==(8,8)
    assert shift_at(schedule,.75,'early')==(0,0)
    assert shift_at(schedule,.75,'late')==(24,24)
    assert shift_at(schedule,.25,'fixed')==(0,0)


def test_legacy_32_step_sampling_is_preserved():
    torch.set_num_threads(1)
    model=FlowNet(3,4).eval(); x=torch.randn(1,3,8,8)
    schedule=torch.randint(4,(32,2),generator=torch.Generator().manual_seed(18)).tolist()
    a,_=sample(model,x,32,4,True,18,2)
    b,trace=integrate(model,x,32,4,'all',schedule,'euler',2,dict(mean=[0]*3,std=[1]*3))
    torch.testing.assert_close(a,b)
    assert len(trace)==9 and trace[-1]['t']==1.


def test_heun_improves_known_ode_and_accounting():
    x=torch.ones(1,1,4,4); norm=dict(mean=[0],std=[2])
    model=lambda x,t:x
    e,_=integrate(model,x,32,4,'fixed',[(0,0)]*32,'euler',1,norm)
    h,trace=integrate(model,x,32,4,'fixed',[(0,0)]*32,'heun',1,norm)
    assert abs(h.mean().item()-np.e)<abs(e.mean().item()-np.e)
    assert trace[0]['mean']==[2.] and trace[0]['fluctuation_energy']==0.
    with pytest.raises(ValueError): integrate(model,x,33,4,'fixed',[(0,0)]*32,'euler',1,norm)


def test_time_probe_is_deterministic_and_reports_mean_error():
    arr=np.ones((3,1,8,8),dtype='float32')
    net=lambda x,t:torch.zeros_like(x)
    a=time_probe(net,arr,4,'cpu',2,2,21)
    assert a==time_probe(net,arr,4,'cpu',2,2,21)
    assert len(a)==11 and a[0]['t']==0 and a[-1]['t']==1
    assert all(r['mse']>=r['spatial_mean_error_mse'] for r in a)


def test_single_window_seams_are_finite():
    from metrics import metrics
    a=np.random.default_rng(1).normal(size=(1,3,8,8))
    assert np.isfinite(metrics(a,8,1)['max_phase_seam_ratio']).all()
