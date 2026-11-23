import json
import numpy as np
import pytest
from data import digest
from prepare_coarse import downsample,shells,prepare
from evaluate_coarse import lowpass


def test_fourier_amplitude_phase_dc_and_antialiasing():
    n=128;m=64;x=np.arange(n)*2*np.pi/n
    a=(2+np.cos(x+.37)[None,:]+.3*np.sin(3*x)[:,None]+.6*np.cos(40*x)[None,:])[None]
    coarse=downsample(a,m)
    expected=(2+np.cos(x[::2]+.37)[None,:]+.3*np.sin(3*x[::2])[:,None])[None]
    np.testing.assert_allclose(coarse,expected,atol=1e-13)
    np.testing.assert_allclose(coarse.mean(),a.mean(),atol=1e-14)
    np.testing.assert_allclose(shells(coarse)[1:17],shells(a)[1:17],atol=1e-14)
    # A target Nyquist mode is deliberately excluded, avoiding an ambiguous axis.
    a=np.broadcast_to(np.cos(32*x),(1,n,n))
    assert abs(downsample(a,m)).max()<1e-12


def fixture(root):
    root.mkdir();rng=np.random.default_rng(1)
    m=dict(axes=['z','y','x'],dims=['time','z','y','x'],channels=['u','v','w'],shape=[128,128],spacing=.1,slice_indices=[0],minimum_time_gap=5,snapshots=[dict(file='fake.nc',time_index=i,time=t,split=s) for i,(t,s) in enumerate([(0,'train'),(1,'train'),(10,'val'),(20,'test')])])
    norm=dict(mean=[1.,2.,3.],std=[2.,3.,4.],channels=m['channels'],manifest_sha256=digest(m))
    (root/'manifest.json').write_text(json.dumps(m));(root/'normalization.json').write_text(json.dumps(norm))
    np.save(root/'train.npy',rng.normal(size=(2,3,128,128)).astype('float32'));np.save(root/'val.npy',rng.normal(size=(1,3,128,128)).astype('float32'))
    # No test array: preparation must not touch it.


def test_preparation_geometry_train_only_statistics_and_provenance(tmp_path):
    source=tmp_path/'source';fixture(source);a=tmp_path/'a';prepare(source,a,64)
    m=json.loads((a/'manifest.json').read_text());n=json.loads((a/'normalization.json').read_text());arr=np.load(a/'train.npy')
    assert m['spacing']==.2 and m['spacing']*64==.1*128
    assert n['manifest_sha256']==digest(m) and n['count']==2*64*64
    assert not (a/'test.npy').exists()
    np.testing.assert_allclose(arr.mean((0,2,3)),0,atol=1e-7);np.testing.assert_allclose(arr.std((0,2,3)),1,atol=1e-6)
    np.save(source/'val.npy',np.load(source/'val.npy')+100)
    b=tmp_path/'b';prepare(source,b,64);n2=json.loads((b/'normalization.json').read_text())
    assert n['mean']==n2['mean'] and n['std']==n2['std']
    assert n['manifest_sha256']!=n2['manifest_sha256'] # source val hash changes
    with pytest.raises(ValueError,match='empty'):prepare(source,a,64)
    with pytest.raises(ValueError):prepare(source,tmp_path/'bad',0)


def test_visual_lowpass_removes_dc_and_unwanted_modes():
    n=64;x=np.arange(n)*2*np.pi/n
    a=np.broadcast_to(8+np.cos(x)+np.sin(8*x),(3,n,n))
    out=lowpass(a)
    np.testing.assert_allclose(out,np.broadcast_to(np.cos(x),(3,n,n)),atol=1e-13)
