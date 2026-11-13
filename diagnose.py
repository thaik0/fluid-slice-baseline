"""Checkpoint-only diagnostics. Uses validation data; never reads the test array."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import FlowNet, tiled
from data import digest
from metrics import metrics
from run import seed, sync, write


ARMS = [('fixed',32,'euler'), ('fixed',64,'euler'), ('fixed',128,'euler'),
        ('fixed',32,'heun'), ('fixed',64,'heun'),
        ('all',32,'euler'), ('all',64,'euler'), ('all',128,'euler'),
        ('early',64,'euler'), ('late',64,'euler')]


def shift_at(schedule, t, policy):
    if policy not in ['fixed','all','early','late']: raise ValueError('Unknown policy')
    active = policy=='all' or (policy=='early' and t<.5) or (policy=='late' and t>=.5)
    return tuple(schedule[min(int(t*len(schedule)),len(schedule)-1)]) if active else (0,0)


@torch.no_grad()
def integrate(model, noise, steps, patch, policy, schedule, solver, batch, norm):
    if solver not in ['euler','heun'] or steps<1 or steps%len(schedule):
        raise ValueError('Need Euler/Heun and steps a positive multiple of shift bins')
    x=noise.clone(); trace=[]; scale=torch.tensor(norm['std'],device=x.device)[None,:,None,None]
    offset=torch.tensor(norm['mean'],device=x.device)[None,:,None,None]
    def record(t):
        raw=x*scale+offset; avg=raw.mean((-2,-1),keepdim=True)
        trace.append(dict(t=t,mean=avg.flatten().tolist(),energy=(raw.square().sum(1).mean()/2).item(),fluctuation_energy=((raw-avg).square().sum(1).mean()/2).item()))
    record(0.)
    for i in range(steps):
        t=i/steps; dt=1/steps; shift=shift_at(schedule,t,policy)
        v=tiled(model,x,t,patch,shift,batch)
        if solver=='euler': x=x+dt*v
        else:
            # Hold this interval's geometry fixed for both evaluations. Nested
            # grids align with all geometry-change boundaries.
            v2=tiled(model,x+dt*v,t+dt,patch,shift,batch)
            x=x+dt*(v+v2)/2
        if (i+1) % max(1,steps//8)==0: record((i+1)/steps)
    return x,trace


@torch.no_grad()
def time_probe(model, arr, patch, device, batches, batch, base_seed):
    rng=np.random.default_rng(base_seed)
    g=torch.Generator(device=device).manual_seed(base_seed)
    times=[0.,.01,.05,.1,.25,.5,.75,.9,.95,.99,1.]
    totals={t:[] for t in times}
    for _ in range(batches):
        crops=[]
        for _ in range(batch):
            field=arr[int(rng.integers(len(arr)))]; h,w=field.shape[-2:]
            y,x=rng.integers(h),rng.integers(w)
            crops.append(field[:,(y+np.arange(patch))%h][:,:,(x+np.arange(patch))%w])
        target=torch.tensor(np.stack(crops),device=device)
        noise=torch.randn(target.shape,generator=g,device=device); truth=target-noise
        for t in times:
            pred=model((1-t)*noise+t*target,t); err=pred-truth
            totals[t].append(dict(mse=err.square().mean().item(),channel_mse=err.square().mean((0,2,3)).tolist(),mean_error=err.mean((0,2,3)).tolist(),spatial_mean_error_mse=err.mean((-2,-1)).square().mean().item()))
    return [dict(t=t,**{k:np.mean([r[k] for r in rows],axis=0).tolist() for k in rows[0]}) for t,rows in totals.items()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--samples',type=int,default=8); p.add_argument('--seed',type=int,default=4242)
    p.add_argument('--batch',type=int,default=8); p.add_argument('--probe-batches',type=int,default=16)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--save-fields',action='store_true')
    a=p.parse_args()
    if min(a.samples,a.batch,a.probe_batches)<1: p.error('Counts must be positive')
    seed(a.seed); out=Path(a.out)
    if (out/'diagnostics.json').exists(): raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    model=FlowNet(**ck['config']).to(a.device); model.load_state_dict(ck['model']); model.eval()
    data=Path(a.data); norm=ck['normalization']; manifest=json.loads((data/'manifest.json').read_text())
    if digest(manifest)!=norm['manifest_sha256'] or json.loads((data/'normalization.json').read_text())!=norm:
        raise ValueError('Checkpoint and prepared data provenance differ')
    patch=ck['args']['patch']; h,w=manifest['shape']
    val=np.load(data/'val.npy',mmap_mode='r'); train=np.load(data/'train.npy',mmap_mode='r')
    mean=np.array(norm['mean'])[None,:,None,None]; std=np.array(norm['std'])[None,:,None,None]
    seam_period=64 if min(h,w)>=64 else patch
    reference=metrics(val*std+mean,seam_period,manifest['spacing'])
    # Preserve snapshot membership: planes within a snapshot are correlated.
    grouped=[]; planes=len(manifest['slice_indices'])
    rows=[r for r in manifest['snapshots'] if r['split']=='val']
    for i,r in enumerate(rows):
        raw=val[i*planes:(i+1)*planes]*std+mean
        grouped.append(dict(time=r['time'],mean=raw.mean((0,2,3)).tolist(),energy=float((raw**2).sum(1).mean()/2)))
    report=dict(checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),manifest_sha256=digest(manifest),args=vars(a),torch=torch.__version__,numpy=np.__version__,reference=reference,reference_snapshot_groups=grouped,seam_period=seam_period,shift_bins=32,inference_patch=patch,samples=[],probes={},source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for split,arr in [('train',train),('val',val)]:
        report['probes'][split]=time_probe(model,arr,patch,a.device,a.probe_batches,a.batch,8192)
    arms=ARMS if not a.smoke else [('fixed',32,'euler'),('all',32,'euler')]
    for s in range(a.seed,a.seed+a.samples):
        noise=torch.randn((1,ck['config']['channels'],h,w),device=a.device,generator=torch.Generator(device=a.device).manual_seed(s))
        schedule=torch.randint(patch,(32,2),generator=torch.Generator().manual_seed(s+100000)).tolist()
        for policy,steps,solver in arms:
            name=f'{policy}-{solver}-{steps}'
            # Warm up outside timing, in the same shape and batch layout.
            tiled(model,noise,0.,patch,(0,0),a.batch); sync(a.device)
            if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            pred,trace=integrate(model,noise,steps,patch,policy,schedule,solver,a.batch,norm)
            sync(a.device); seconds=time.perf_counter()-start
            raw=pred.cpu().numpy()*std+mean
            if a.save_fields: np.save(out/f'{name}-{s}.npy',raw)
            report['samples'].append(dict(arm=name,seed=s,nfe=steps*(2 if solver=='heun' else 1),seconds=seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,trace=trace,metrics=metrics(raw,seam_period,manifest['spacing'])))
            print(name,s,flush=True)
            write(out/'partial.json',report)
    report['paired_vs_fixed64'] = paired(report,'fixed-euler-64')
    report['summary']=summarize(report)
    write(out/'diagnostics.json',report)
    (out/'summary.md').write_text(markdown(report))
    plot_report(report,out)


def paired(report, baseline):
    controls={s['seed']:s for s in report['samples'] if s['arm']==baseline}
    rows=[]
    for s in report['samples']:
        if s['seed'] not in controls: continue
        c=controls[s['seed']]['metrics']; m=s['metrics']
        rows.append(dict(arm=s['arm'],seed=s['seed'],baseline=baseline,delta={k:(np.array(m[k])-np.array(c[k])).tolist() for k in ['energy','fluctuation_energy','low_band_energy','high_band_energy','mean','max_phase_seam_ratio']}))
    return rows


def summarize(report):
    result={}; ref=report['reference']
    for arm in sorted({s['arm'] for s in report['samples']}):
        ss=[s for s in report['samples'] if s['arm']==arm]
        rows=[]
        for s in ss:
            m=s['metrics']; spec=np.array(m['spectrum']); rs=np.array(ref['spectrum'])
            rows.append(dict(energy_ratio=m['energy']/ref['energy'],fluctuation_ratio=m['fluctuation_energy']/ref['fluctuation_energy'],low_ratio=m['low_band_energy']/ref['low_band_energy'],high_ratio=m['high_band_energy']/ref['high_band_energy'],seam_yx=m['max_phase_seam_ratio'],mean=m['mean'],spectrum_l1=float(abs(spec-rs).sum()/rs.sum()),seconds=s['seconds']))
        result[arm]={k:dict(mean=np.mean([r[k] for r in rows],0).tolist(),min=np.min([r[k] for r in rows],0).tolist(),max=np.max([r[k] for r in rows],0).tolist()) for k in rows[0]}
    return result


def plot_report(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for split,rows in report['probes'].items():
        axes[0].semilogy([r['t'] for r in rows],[r['mse'] for r in rows],'-o',label=split)
    axes[0].set(xlabel='Noise-to-data time t',ylabel='Normalized prediction MSE',title='Error across the sampling path')
    axes[0].legend()
    ref=report['reference']; k=np.array(ref['k'])
    axes[2].loglog(k[1:],ref['spectrum'][1:],'k',lw=2,label='Real validation')
    shown=['fixed-euler-32','fixed-euler-128','all-euler-64','early-euler-64','late-euler-64']
    for arm in shown:
        rows=[s for s in report['samples'] if s['arm']==arm]
        if not rows: continue
        tr=rows[0]['trace']; v=np.mean([[r['mean'][1] for r in s['trace']] for s in rows],0)
        axes[1].plot([r['t'] for r in tr],v,label=arm)
        spec=np.mean([s['metrics']['spectrum'] for s in rows],0)
        axes[2].loglog(k[1:],spec[1:],label=arm)
    axes[1].axhline(ref['mean'][1],ls='--',color='black',label='Pooled real v mean')
    axes[1].set(xlabel='Noise-to-data time t',ylabel='Mean v (dataset units)',title='When mean drift appears')
    axes[2].set(xlabel='Planar angular wavenumber',ylabel='Shell-summed fluctuation energy',title='Energy across scales')
    for ax in axes[1:]: ax.legend(fontsize=7)
    fig.savefig(out/'diagnostics.png',dpi=160); plt.close(fig)


def markdown(report):
    text=['# Diagnostic screen','', 'Validation reference only. Ranges span generated seeds, not confidence intervals. No automatic winner selection.','', '| Arm | Energy / real | Fluctuation / real | Low / real | High / real | Spectrum L1 |','|---|---:|---:|---:|---:|---:|']
    for name,r in report['summary'].items():
        text.append('| '+name+' | '+' | '.join(f"{r[k]['mean']:.3f}" for k in ['energy_ratio','fluctuation_ratio','low_ratio','high_ratio','spectrum_l1'])+' |')
    return '\n'.join(text)+'\n'

if __name__=='__main__': main()
