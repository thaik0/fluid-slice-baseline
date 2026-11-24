"""Oracle-context residual flow pilot. Low modes are supplied, never learned here."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from data import digest
from model import FlowNet
from prepare_coarse import downsample, file_hash
from run import seed, sync, write


def low(x, coarse_size):
    """Orthogonal radial projector including DC, excluding target Nyquist."""
    n=x.shape[-1]
    if x.shape[-2]!=n or coarse_size<4 or coarse_size>=n or n%coarse_size:
        raise ValueError('Need square fine grid and smaller divisible coarse grid')
    k=torch.fft.fftfreq(n,device=x.device)*n
    mask=torch.hypot(k[:,None],k[None,:]) < coarse_size/2
    return torch.fft.ifft2(torch.fft.fft2(x)*mask).real


def high(x, coarse_size):
    return x-low(x,coarse_size)


def upsample(a, size):
    """Aligned periodic Fourier interpolation with physical amplitude preserved."""
    a=np.asarray(a,dtype='float64');n=a.shape[-1]
    if a.shape[-2]!=n or size<=n or size%n:raise ValueError('Invalid grids')
    k=np.rint(np.fft.fftfreq(n)*n).astype(int)
    mask=np.hypot(k[:,None],k[None,:])<n/2
    f=np.fft.fft2(a)*mask
    padded=np.zeros((*a.shape[:-2],size,size),dtype='complex128')
    idx=k%size
    padded[...,idx[:,None],idx[None,:]]=f*(size/n)**2
    return np.fft.ifft2(padded).real


def load_pair(fine,coarse):
    fine=Path(fine);coarse=Path(coarse)
    m=json.loads((fine/'manifest.json').read_text());cm=json.loads((coarse/'manifest.json').read_text())
    norm=json.loads((fine/'normalization.json').read_text());cn=json.loads((coarse/'normalization.json').read_text())
    if digest(m)!=norm['manifest_sha256'] or digest(cm)!=cn['manifest_sha256']:
        raise ValueError('Manifest/normalization mismatch')
    info=cm.get('coarsening',{})
    if cm.get('data_kind')!='fourier_coarsened_prepared_planes' or info.get('source_manifest_sha256')!=digest(m) or info.get('source_normalization_sha256')!=file_hash(fine/'normalization.json'):
        raise ValueError('Coarse source provenance mismatch')
    for stats in [norm,cn]:
        if not np.isfinite(stats['mean']).all() or not np.isfinite(stats['std']).all() or np.any(np.asarray(stats['std'])<=0):
            raise ValueError('Invalid normalization')
    n=m['shape'][0];c=cm['shape'][0]
    if m['shape']!=[n,n] or cm['shape']!=[c,c] or n%c or c>=n or abs(n*m['spacing']-c*cm['spacing'])>1e-10:
        raise ValueError('Incompatible physical grids')
    arrays={}
    for split in ['train','val']:
        if file_hash(fine/f'{split}.npy')!=info['source_array_sha256'][split]:
            raise ValueError('Fine array differs from coarse source')
        a=np.load(fine/f'{split}.npy',mmap_mode='r');b=np.load(coarse/f'{split}.npy',mmap_mode='r')
        count=sum(r['split']==split for r in m['snapshots'])*len(m['slice_indices'])
        if a.shape!=(count,len(norm['mean']),n,n) or b.shape!=(count,len(norm['mean']),c,c):
            raise ValueError('Array shape/count mismatch')
        arrays[split]=(a,b)
    return m,cm,norm,cn,arrays


def check(fine,coarse,out):
    torch.set_num_threads(1)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    m,cm,norm,cn,arrays=load_pair(fine,coarse);c=cm['shape'][0];n=m['shape'][0]
    mu=np.array(norm['mean'])[:,None,None];sd=np.array(norm['std'])[:,None,None]
    cmu=np.array(cn['mean'])[:,None,None];csd=np.array(cn['std'])[:,None,None]
    errors=[]
    for split,(a,b) in arrays.items():
        for i in sorted(set([0,len(a)//2,len(a)-1])):
            raw=np.asarray(a[i],dtype='float64')*sd+mu
            ctx=upsample(np.asarray(b[i],dtype='float64')*csd+cmu,n)
            if not np.isfinite(raw).all() or not np.isfinite(ctx).all():raise ValueError('Nonfinite source field')
            exact=low(torch.from_numpy(raw),c).numpy()
            residual=raw-exact
            row=dict(split=split,index=i,stored_context_max_error=float(abs(ctx-exact).max()),roundtrip_max_error=float(abs(upsample(downsample(raw,c),n)-exact).max()),residual_low_max=float(abs(low(torch.from_numpy(residual),c).numpy()).max()),reconstruction_max_error=float(abs(exact+residual-raw).max()))
            if row['stored_context_max_error']>1e-5 or max(row[k] for k in ['roundtrip_max_error','residual_low_max','reconstruction_max_error'])>1e-10:
                raise ValueError(f'Filtering check failed: {row}')
            errors.append(row)
    report=dict(status='passed',fine_manifest_sha256=digest(m),coarse_manifest_sha256=digest(cm),checked_planes=errors,warning='Geometry checks only; no independence or generation quality claim. Test data not opened.')
    write(Path(out),report);return report


@torch.no_grad()
def generate(model,context,noise,coarse_size,steps):
    x=high(noise,coarse_size)
    for i in range(steps):
        x=high(x+high(model(x,i/steps,context),coarse_size)/steps,coarse_size)
    return context+x


def spectrum(a):
    # Batch mean, channel sum, physical fluctuation shell energies.
    n=a.shape[-1];k=np.fft.fftfreq(n)*n
    bins=np.rint(np.hypot(k[:,None],k[None,:])).astype(int)
    a=a-a.mean((-2,-1),keepdims=True)
    power=.5*(abs(np.fft.fft2(a))**2).sum(1).mean(0)/n**4
    return np.bincount(bins.ravel(),weights=power.ravel())


def train(a):
    seed(a.seed);out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    check(a.data,a.coarse,out/'geometry-check.json')
    m,cm,norm,cn,arrays=load_pair(a.data,a.coarse);c=cm['shape'][0];channels=len(norm['mean'])
    config=dict(channels=channels,width=a.width,normalization='group',padding='circular',context_channels=channels)
    model=FlowNet(**config).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    history=[];source_hashes={p:file_hash(p) for p in ['model.py','refinement.py']}
    def save(name):
        torch.save(dict(model=model.state_dict(),config=config,normalization=norm,manifest=m,coarse_manifest=cm,args=vars(a),history=history,source_hashes=source_hashes,kind='real_coarse_conditional_residual'),out/name)
    # Fix validation planes, noise and time across every report.
    val=torch.as_tensor(np.array(arrays['val'][0][:min(a.batch,len(arrays['val'][0]))]),device=a.device)
    vc=low(val,c);vt=high(val,c)
    vn=high(torch.randn(val.shape,device=a.device,generator=torch.Generator(device=a.device).manual_seed(9182)),c)
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    sync(a.device);start=time.perf_counter()
    for step in range(1,a.updates+1):
        model.train();idx=np.random.randint(len(arrays['train'][0]),size=a.batch)
        real=torch.as_tensor(np.array(arrays['train'][0][idx]),device=a.device)
        context=low(real,c);target=high(real,c);noise=high(torch.randn_like(real),c)
        t=torch.rand(a.batch,1,1,1,device=a.device)
        pred=high(model((1-t)*noise+t*target,t,context),c)
        loss=(pred-(target-noise)).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step==1 or step%100==0 or step==a.updates:
            model.eval()
            with torch.no_grad():vl=(high(model((vn+vt)/2,.5,vc),c)-(vt-vn)).square().mean().item()
            row=dict(step=step,train_loss=loss.item(),val_loss_t05=vl);history.append(row);print(row,flush=True)
            write(out/'partial.json',dict(history=history,args=vars(a)))
        if step%1000==0:save(f'checkpoint-{step}.pt')
    sync(a.device);elapsed=time.perf_counter()-start;save('checkpoint.pt')
    write(out/'training.json',dict(args=vars(a),config=config,history=history,seconds=elapsed,parameters=sum(p.numel() for p in model.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,source_hashes=source_hashes,warning='One seed, real coarse conditioning; not end-to-end generated context.'))


def evaluate(a):
    seed(4242);out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    m,cm,norm,cn,arrays=load_pair(a.data,a.coarse)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if ck['kind']!='real_coarse_conditional_residual' or ck['manifest']!=m or ck['coarse_manifest']!=cm or ck['normalization']!=norm:raise ValueError('Checkpoint provenance mismatch')
    model=FlowNet(**ck['config']).to(a.device);model.load_state_dict(ck['model']);model.eval()
    c=cm['shape'][0];n=m['shape'][0];val=arrays['val'][0]
    if a.samples>len(val):raise ValueError('Samples exceed distinct validation contexts')
    ids=np.linspace(0,len(val)-1,a.samples,dtype=int)
    mu=np.array(norm['mean'])[None,:,None,None];sd=np.array(norm['std'])[None,:,None,None]
    refs=[];gens=[];resrefs=[];resgens=[];rows=[];views=None
    for j,idx in enumerate(ids):
        real=torch.as_tensor(np.array(val[idx:idx+1]),device=a.device);ctx=low(real,c)
        noise=torch.randn(real.shape,device=a.device,generator=torch.Generator(device=a.device).manual_seed(4242+j))
        sync(a.device);start=time.perf_counter();pred=generate(model,ctx,noise,c,a.steps);sync(a.device)
        if not torch.isfinite(pred).all():raise ValueError('Nonfinite generated field')
        raw=real.cpu().numpy()*sd+mu;gen=pred.cpu().numpy()*sd+mu
        rc=ctx.cpu().numpy()*sd+mu
        r=raw-rc;g=gen-rc
        error=float(abs(low(pred,c)-ctx).max().item())
        if error>1e-4:raise ValueError('Refiner changed supplied low modes')
        rows.append(dict(seed=4242+j,validation_index=int(idx),seconds=time.perf_counter()-start,max_normalized_low_mode_error=error))
        refs.append(spectrum(raw));gens.append(spectrum(gen));resrefs.append(spectrum(r));resgens.append(spectrum(g))
        if views is None:views=[raw[0,0],rc[0,0],gen[0,0],r[0,0],g[0,0]];np.savez(out/'example.npz',real=raw,context=rc,generated=gen)
        write(out/'partial.json',dict(samples=rows))
    rr=np.mean(resrefs,0);rg=np.mean(resgens,0);rf=np.mean(refs,0);gf=np.mean(gens,0)
    report=dict(kind='oracle_real_coarse_refinement',model_config=ck['config'],training_args=ck['args'],samples=rows,fine_manifest_sha256=digest(m),coarse_manifest_sha256=digest(cm),checkpoint_sha256=file_hash(a.checkpoint),args=vars(a),reference_spectrum=rf.tolist(),generated_spectrum=gf.tolist(),reference_residual_spectrum=rr.tolist(),generated_residual_spectrum=rg.tolist(),residual_energy_ratio=float(rg.sum()/max(rr.sum(),1e-30)),residual_spectrum_relative_l1=float(abs(rg-rr).sum()/max(rr.sum(),1e-30)),high_shell_start=n//4,high_band_ratio=float(rg[n//4:].sum()/max(rr[n//4:].sum(),1e-30)),warning='Low modes, including plane means, are supplied by real validation fields and preserved by construction. They do not measure learned generation. Residual spectra use the same matched subset of real contexts. One training seed; no generated coarse conditioning; provisional split; test unopened.')
    write(out/'evaluation.json',report)
    (out/'summary.md').write_text(f'# Real-coarse refinement diagnostic\n\n{report["warning"]}\n\nResidual energy / real: {report["residual_energy_ratio"]:.3f}\n\nResidual spectrum relative L1: {report["residual_spectrum_relative_l1"]:.3f}\n\nHigh band / real: {report["high_band_ratio"]:.3f}\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(18,4),layout='constrained')
    for i,(v,title) in enumerate(zip(views,['Real u','Supplied real coarse u','Refined u','Real residual u','Generated residual u'])):
        lim=max(abs(views[k]).max() for k in (range(3) if i<3 else [3,4]))
        im=axes[i].imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim);axes[i].set_title(title);axes[i].axis('off')
        fig.colorbar(im,ax=axes[i],shrink=.65)
    fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(layout='constrained');ax.loglog(rr,label='Real residual');ax.loglog(rg,label='Generated residual');ax.set(xlabel='Planar radial shell',ylabel='Residual fluctuation energy');ax.legend();fig.savefig(out/'residual-spectrum.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['check','train','evaluate']);p.add_argument('--data',required=True);p.add_argument('--coarse',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--updates',type=int,default=5000);p.add_argument('--batch',type=int,default=8);p.add_argument('--width',type=int,default=32);p.add_argument('--seed',type=int,default=42);p.add_argument('--checkpoint');p.add_argument('--samples',type=int,default=8);p.add_argument('--steps',type=int,default=64)
    a=p.parse_args()
    if min(a.updates,a.batch,a.samples,a.steps)<1:p.error('Counts must be positive')
    if a.command=='check':check(a.data,a.coarse,a.out)
    elif a.command=='train':train(a)
    else:
        if not a.checkpoint:p.error('Evaluation requires --checkpoint')
        evaluate(a)


if __name__=='__main__':main()
