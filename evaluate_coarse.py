"""Fixed-sampler coarse pilot: individual low shells, low-pass views, no test data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from data import digest
from model import FlowNet,sample,validate_domain
from metrics import metrics
from run import seed,sync,write


def lowpass(a,max_shell=2):
    n=a.shape[-1];k=np.fft.fftfreq(n)*n
    radius=np.hypot(k[:,None],k[None,:])
    # Remove DC and retain all modes assigned to the first two radial shells.
    mask=(radius>0)&(np.rint(radius)<=max_shell)
    return np.fft.ifft2(np.fft.fft2(a,axes=(-2,-1))*mask,axes=(-2,-1)).real


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--samples',type=int,default=8);p.add_argument('--steps',type=int,default=64);p.add_argument('--seed',type=int,default=4242)
    a=p.parse_args()
    if a.samples<1 or a.steps<1:p.error('Counts must be positive')
    seed(a.seed);out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh evaluation directory')
    out.mkdir(parents=True,exist_ok=True);data=Path(a.data)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False);m=json.loads((data/'manifest.json').read_text());norm=ck['normalization']
    if digest(m)!=norm['manifest_sha256'] or json.loads((data/'normalization.json').read_text())!=norm:raise ValueError('Checkpoint/data mismatch')
    if m.get('data_kind')!='fourier_coarsened_prepared_planes':raise ValueError('Expected prepared coarse planes')
    n=m['shape'][0];validate_domain(ck['config'].get('padding','zeros'),ck['args']['patch'],m['shape'])
    model=FlowNet(**ck['config']).to(a.device);model.load_state_dict(ck['model']);model.eval()
    mu=np.array(norm['mean'])[None,:,None,None];sd=np.array(norm['std'])[None,:,None,None]
    reference=np.load(data/'val.npy',mmap_mode='r')*sd+mu
    ref=metrics(reference,n,m['spacing']);plane_count=len(m['slice_indices'])
    snapshot_rows=[r for r in m['snapshots'] if r['split']=='val']
    groups=[]
    for i,r in enumerate(snapshot_rows):
        s=metrics(reference[i*plane_count:(i+1)*plane_count],n,m['spacing'])
        groups.append(dict(time=r['time'],shells_1_16=s['spectrum'][1:17],plane_means=reference[i*plane_count:(i+1)*plane_count].mean((-2,-1)).tolist()))
    report=dict(reference=ref,reference_snapshot_groups=groups,manifest_sha256=digest(m),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),model_config=ck['config'],args=vars(a),torch=torch.__version__,numpy=np.__version__,samples=[],warning='Only common physical shells 1–16 should be compared across resolutions. high_band_energy starts at shell24 on96, not shell96 on384. Seam score for a single window is not an internal tiling diagnostic. Source time split remains provisional.')
    views=[reference[0]]
    for s in range(a.seed,a.seed+a.samples):
        noise=torch.randn((1,len(norm['mean']),n,n),device=a.device,generator=torch.Generator(device=a.device).manual_seed(s))
        with torch.no_grad():sample(model,noise,1,n,False,s,1) # untimed warmup
        if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        sync(a.device);start=time.perf_counter()
        pred,_=sample(model,noise,a.steps,n,False,s,1)
        sync(a.device);seconds=time.perf_counter()-start
        raw=pred.cpu().numpy()*sd+mu;stat=metrics(raw,n,m['spacing'])
        report['samples'].append(dict(seed=s,seconds=seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,metrics=stat,shell_ratio_1_16=(np.array(stat['spectrum'][1:17])/np.maximum(ref['spectrum'][1:17],1e-30)).tolist()))
        if len(views)<3:views.append(raw[0]);np.save(out/f'generated-{s}.npy',raw)
        write(out/'partial.json',report)
    ratios=np.array([s['shell_ratio_1_16'] for s in report['samples']]);report['shell_ratio_1_16']=dict(mean=ratios.mean(0).tolist(),min=ratios.min(0).tolist(),max=ratios.max(0).tolist())
    write(out/'coarse-evaluation.json',report)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(views),figsize=(4*len(views),7),squeeze=False,layout='constrained')
    for row in [0,1]:
        fields=[v[0] if row==0 else lowpass(v)[0] for v in views];limit=max(float(abs(v).max()) for v in fields)
        for j,v in enumerate(fields):
            im=axes[row,j].imshow(v,cmap='RdBu_r',vmin=-limit,vmax=limit);axes[row,j].axis('off');axes[row,j].set_title(('Real validation' if j==0 else f'Generated {j}')+(' u' if row==0 else ' u: shells1–2, no DC'))
        fig.colorbar(im,ax=axes[row,:].tolist(),label='Dataset velocity units')
    fig.savefig(out/'coarse-fields.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.errorbar(range(1,17),ratios.mean(0),yerr=[ratios.mean(0)-ratios.min(0),ratios.max(0)-ratios.mean(0)],fmt='o-');ax.axhline(1,color='k',ls='--');ax.set(xlabel='Common planar radial shell',ylabel='Generated energy / coarse real',title='Coarse pilot: eight-seed range is descriptive');fig.savefig(out/'low-shells.png',dpi=160);plt.close(fig)
    (out/'summary.md').write_text('# Coarse pilot\n\nValidation only; independent generated fields, not paired reconstructions.\n\n| Shell | Mean energy / real | Min | Max |\n|---|---:|---:|---:|\n'+'\n'.join(f'| {i+1} | {ratios.mean(0)[i]:.3f} | {ratios.min(0)[i]:.3f} | {ratios.max(0)[i]:.3f} |' for i in range(16))+'\n')
if __name__=='__main__':main()
