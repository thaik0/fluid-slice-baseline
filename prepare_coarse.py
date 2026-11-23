"""CPU-only filtered whole-domain preparation from existing normalized planes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from data import load_manifest,digest


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def downsample(a,size):
    """Strict radial cutoff, then aligned decimation; no interpolation or FFT rescaling."""
    a=np.asarray(a,dtype='float64');h,w=a.shape[-2:]
    if h!=w or size<4 or size%4 or h%size:raise ValueError('Need square input and divisible target size (multiple of four)')
    factor=h//size
    if factor<2:raise ValueError('Target must be smaller than source')
    k=np.fft.fftfreq(h)*h
    mask=np.hypot(k[:,None],k[None,:]) < size/2
    filtered=np.fft.ifft2(np.fft.fft2(a,axes=(-2,-1))*mask,axes=(-2,-1)).real
    return filtered[...,::factor,::factor]


def shells(a,limit=16):
    h,w=a.shape[-2:];k=np.fft.fftfreq(h)*h
    bins=np.rint(np.hypot(k[:,None],k[None,:])).astype(int)
    f=a-a.mean((-2,-1),keepdims=True)
    power=.5*(abs(np.fft.fft2(f,axes=(-2,-1)))**2).sum(0)/(h*w)**2
    return np.bincount(bins.ravel(),weights=power.ravel(),minlength=limit+1)[:limit+1]


def prepare(source,out,size):
    source=Path(source);out=Path(out)
    m=load_manifest(source/'manifest.json');norm=json.loads((source/'normalization.json').read_text())
    if digest(m)!=norm['manifest_sha256']:raise ValueError('Source provenance mismatch')
    h,w=m['shape']
    if size<36 or size%4 or h!=w or size>=h or h%size:raise ValueError('Target >=36, divisible by four, smaller than source, must divide source size')
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    mu=np.array(norm['mean'])[:,None,None];sd=np.array(norm['std'])[:,None,None]
    if np.any(sd<=0):raise ValueError('Invalid source normalization')
    c=len(mu);count=0;avg=np.zeros(c);m2=np.zeros(c);checks={};hashes={}
    for split in ['train','val']: # Deliberately do not open test.npy.
        path=source/f'{split}.npy';arr=np.load(path,mmap_mode='r')
        expected=sum(r['split']==split for r in m['snapshots'])*len(m['slice_indices'])
        if arr.shape!=(expected,c,h,w):raise ValueError('Source shape/count differs from manifest')
        hashes[split]=file_hash(path)
        dest=np.lib.format.open_memmap(out/f'{split}.npy',mode='w+',dtype='float32',shape=(len(arr),c,size,size))
        original=np.zeros(17);coarse=np.zeros(17);dc_error=0.
        for i in range(len(arr)):
            raw=np.asarray(arr[i],dtype='float64')*sd+mu
            if not np.isfinite(raw).all():raise ValueError('Nonfinite source field')
            low=downsample(raw,size);dest[i]=low
            original+=shells(raw);coarse+=shells(low)
            dc_error=max(dc_error,float(abs(raw.mean((-2,-1))-low.mean((-2,-1))).max()))
            if split=='train':
                b=low.reshape(c,-1);n=b.shape[1];mean=b.mean(1);delta=mean-avg
                m2+=((b-mean[:,None])**2).sum(1)+delta**2*count*n/(count+n)
                avg+=delta*n/(count+n);count+=n
            if (i+1)%32==0:print(f'Filtered {split}: {i+1}/{len(arr)}',flush=True)
        dest.flush()
        rel=float(np.max(abs(original[1:17]-coarse[1:17])/np.maximum(original[1:17],1e-30)))
        if rel>1e-8:raise ValueError('Common low-shell energy was not preserved')
        checks[split]=dict(planes=len(arr),max_plane_mean_error_before_storage=dc_error,max_relative_shell_1_16_error_before_storage=rel,source_spectrum_shells_0_16=(original/len(arr)).tolist(),coarse_spectrum_shells_0_16=(coarse/len(arr)).tolist())
    std=np.sqrt(m2/count)
    if not np.isfinite(std).all() or np.any(std<=0):raise ValueError('Invalid coarse training statistics')
    derived=copy.deepcopy(m);derived['shape']=[size,size];derived['spacing']=m['spacing']*h/size
    derived['data_kind']='fourier_coarsened_prepared_planes'
    derived['coarsening']=dict(source_manifest_sha256=digest(m),source_normalization_sha256=file_hash(source/'normalization.json'),source_array_sha256=hashes,source_shape=[h,w],source_spacing=m['spacing'],box_length=h*m['spacing'],cutoff_integer_radius_strictly_less_than=size/2,method='radial FFT filter on source grid then aligned integer decimation',origin_policy='retain location of source sample zero; no half-cell recentering',normalization='coarse training planes only',exported_splits=['train','val'],test='Reserved; test array not read or exported')
    newnorm=dict(mean=avg.tolist(),std=std.tolist(),count=count,manifest_sha256=digest(derived),channels=m['channels'])
    normalized={}
    for split in ['train','val']:
        a=np.load(out/f'{split}.npy',mmap_mode='r+');sums=np.zeros(c);squares=np.zeros(c)
        for i in range(len(a)):
            b=(a[i].astype('float64')-avg[:,None,None])/std[:,None,None];a[i]=b
            sums+=a[i].sum((1,2),dtype='float64');squares+=(a[i].astype('float64')**2).sum((1,2))
        a.flush();pixels=len(a)*size*size
        normalized[split]=dict(mean=(sums/pixels).tolist(),std=np.sqrt(np.maximum(squares/pixels-(sums/pixels)**2,0)).tolist())
    report=dict(source_shape=[h,w],target_shape=[size,size],spacing=derived['spacing'],box_length=derived['coarsening']['box_length'],normalization=newnorm,normalized_checks=normalized,preservation=checks,warning='Preparation verifies filtering geometry, not statistical independence or generation quality. Compare only common physical shells across resolutions.')
    for name,value in [('manifest.json',derived),('normalization.json',newnorm),('coarse-preparation.json',report)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=96)
    a=p.parse_args();prepare(a.source,a.out,a.size)
