import copy
import json
import numpy as np
import pytest
import torch
import xarray as xr
from data import validate, prepare
from model import FlowNet,tiled,sample
from metrics import metrics


def fixture(tmp_path):
    rng=np.random.default_rng(1)
    a=rng.normal(size=(3,2,16,16)).astype('float32')
    ds=xr.Dataset({c:(('time','z','y','x'),a+i) for i,c in enumerate(['u','v','w'])},coords={'time':[0.,10.,20.]})
    ds.to_netcdf(tmp_path/'fluid_0.nc')
    m=dict(axes=['z','y','x'],dims=['time','z','y','x'],channels=['u','v','w'],shape=[16,16],slice_indices=[0,1],spacing=.1,minimum_time_gap=5.,status='provisional',snapshots=[dict(file='fluid_0.nc',time_index=i,time=float(i*10),split=s) for i,s in enumerate(['train','val','test'])])
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(m)); return m,path


def test_loading_training_only_norm(tmp_path):
    m,path=fixture(tmp_path); prepare(path,tmp_path,tmp_path/'out')
    train=np.load(tmp_path/'out/train.npy')
    np.testing.assert_allclose(train.mean((0,2,3)),0,atol=1e-6)
    np.testing.assert_allclose(train.std((0,2,3)),1,atol=1e-6)
    before=json.loads((tmp_path/'out/normalization.json').read_text())
    with xr.open_dataset(tmp_path/'fluid_0.nc') as ds: ds=ds.load()
    ds['u'].values[1:]+=1000; ds.to_netcdf(tmp_path/'fluid_0.nc',mode='w')
    prepare(path,tmp_path,tmp_path/'out2')
    assert json.loads((tmp_path/'out2/normalization.json').read_text())==before


def test_split_guards(tmp_path):
    m,_=fixture(tmp_path)
    bad=copy.deepcopy(m); bad['snapshots'].append(bad['snapshots'][0])
    with pytest.raises(ValueError): validate(bad)
    bad=copy.deepcopy(m); bad['minimum_time_gap']=11
    with pytest.raises(ValueError): validate(bad)


def test_tiling_and_periodic_shift():
    x=torch.arange(3*16*24).reshape(1,3,16,24).float()
    for shift in [(0,0),(7,3),(15,23)]:
        assert torch.equal(tiled(lambda p,t:p,x,0,8,shift,batch=2),x)


def test_sampler_global_update_and_rng():
    x=torch.randn(1,3,16,16)
    constant=lambda p,t:torch.ones_like(p)*2
    a,_=sample(constant,x,4,8,False,1)
    b,sh=sample(constant,x,4,8,True,1)
    torch.testing.assert_close(a,x+2); torch.testing.assert_close(a,b)
    _,again=sample(constant,x,4,8,True,1)
    assert sh==again


def test_model_backward():
    torch.set_num_threads(1)
    net=FlowNet(3,8); x=torch.randn(2,3,8,8)
    loss=net(x,torch.tensor([.2,.8])).square().mean(); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_spectrum_parseval_and_slice_divergence():
    n=32; x=np.arange(n)*2*np.pi/n
    a=np.zeros((1,3,n,n)); a[:,0]=np.sin(x)[None,:]
    m=metrics(a,8,2*np.pi/n)
    assert m['energy']==pytest.approx(.25)
    assert sum(m['spectrum'])==pytest.approx(.25)
    assert np.argmax(m['spectrum'])==1
    assert m['in_plane_divergence_rms']>0


def test_coordinate_mismatch_rejected(tmp_path):
    m,path=fixture(tmp_path)
    m['snapshots'][0]['time']=1.
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='time differs'):
        prepare(path,tmp_path,tmp_path/'out')


def test_phase_seam_detects_relocated_grid():
    a=np.random.default_rng(1).normal(0,.01,(1,3,32,32))
    for x in range(32): a[:,:,:,x] += (x//8)%2
    first=metrics(a,8,1)['max_phase_seam_ratio'][1]
    moved=metrics(np.roll(a,3,-1),8,1)['max_phase_seam_ratio'][1]
    assert first>5 and moved>5
