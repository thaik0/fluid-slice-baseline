import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from data import prepare, digest
from model import FlowNet, sample
from metrics import metrics


def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2))
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def train(a):
    seed(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    data=Path(a.data)
    norm=json.loads((data/'normalization.json').read_text())
    m=json.loads((data/'manifest.json').read_text())
    if digest(m)!=norm['manifest_sha256']: raise ValueError('Manifest/normalization mismatch')
    x=np.load(data/'train.npy',mmap_mode='r'); val=np.load(data/'val.npy',mmap_mode='r')
    if a.patch%4 or a.patch<4 or a.patch>min(x.shape[-2:]): raise ValueError('Invalid training patch')
    model=FlowNet(x.shape[1],a.width).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    rng=np.random.default_rng(a.seed)
    def batch(arr,rng):
        crops=[]
        for _ in range(a.batch):
            field=arr[int(rng.integers(len(arr)))]; h,w=field.shape[-2:]
            y,z=rng.integers(h),rng.integers(w)
            crops.append(field[:,(y+np.arange(a.patch))%h][:,:,(z+np.arange(a.patch))%w])
        return torch.tensor(np.stack(crops),device=a.device)
    history=[]; start=time.perf_counter()
    if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        model.train(); target=batch(x,rng); noise=torch.randn_like(target); t=torch.rand(a.batch,device=a.device)
        state=(1-t[:,None,None,None])*noise+t[:,None,None,None]*target
        loss=(model(state,t)-(target-noise)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step==0 or (step+1)%100==0 or step+1==a.steps:
            # Fixed validation crops/noise, separate RNG, no training RNG perturbation.
            model.eval()
            with torch.random.fork_rng(devices=[torch.device(a.device).index or 0] if a.device.startswith('cuda') else []), torch.no_grad():
                torch.manual_seed(9182)
                v=batch(val,np.random.default_rng(9182)); z=torch.randn_like(v); tv=torch.full((a.batch,),.5,device=a.device)
                vl=(model((v+z)/2,tv)-(v-z)).square().mean().item()
            record=dict(step=step+1,train_loss=loss.item(),val_loss_t05=vl)
            history.append(record); print(record,flush=True)
    sync(a.device)
    config=dict(channels=x.shape[1],width=a.width)
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    ckpt=dict(model=model.state_dict(),optimizer=opt.state_dict(),config=config,normalization=norm,manifest=m,args=vars(a),history=history,source_hashes=source_hashes)
    torch.save(ckpt,out/'checkpoint.pt')
    write(out/'training.json',dict(parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),args=vars(a),history=history,source_hashes=source_hashes))


def evaluate(a):
    seed(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False) # trusted own checkpoint only
    model=FlowNet(**ck['config']).to(a.device); model.load_state_dict(ck['model']); model.eval()
    norm=ck['normalization']; data=Path(a.data)
    if json.loads((data/'normalization.json').read_text())!=norm: raise ValueError('Normalization differs from checkpoint')
    m=json.loads((data/'manifest.json').read_text())
    if digest(m)!=norm['manifest_sha256']: raise ValueError('Reference manifest differs')
    h,w=m['shape']; p=ck['args']['patch']
    mean=np.array(norm['mean'])[None,:,None,None]; std=np.array(norm['std'])[None,:,None,None]
    ref=np.load(data/f'{a.split}.npy',mmap_mode='r')
    report=dict(reference_split=a.split,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),manifest_sha256=digest(m),reference=metrics(ref*std+mean,p,m['spacing']),samples=[])
    for s in range(a.seed,a.seed+a.samples):
        g=torch.Generator(device=a.device).manual_seed(s)
        noise=torch.randn((1,ck['config']['channels'],h,w),generator=g,device=a.device)
        for shifted in [False,True]:
            if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            sync(a.device); start=time.perf_counter()
            pred,shifts=sample(model,noise,a.steps,p,shifted,s+100000,a.batch)
            sync(a.device); elapsed=time.perf_counter()-start
            raw=pred.cpu().numpy()*std+mean
            mode='shifted' if shifted else 'fixed'
            np.save(out/f'{mode}-{s}.npy',raw)
            report['samples'].append(dict(seed=s,mode=mode,seconds=elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,shifts=shifts,metrics=metrics(raw,p,m['spacing'])))
    report['reference_planes'] = [metrics(ref[i:i+1]*std+mean,p,m['spacing']) for i in range(len(ref))]
    report['paired_shifted_minus_fixed'] = []
    for fixed,shifted in zip(report['samples'][::2], report['samples'][1::2]):
        fm,sm=fixed['metrics'],shifted['metrics']
        report['paired_shifted_minus_fixed'].append(dict(seed=fixed['seed'],energy=sm['energy']-fm['energy'],low_band_energy=sm['low_band_energy']-fm['low_band_energy'],high_band_energy=sm['high_band_energy']-fm['high_band_energy'],seam_ratio=(np.array(sm['max_phase_seam_ratio'])-fm['max_phase_seam_ratio']).tolist()))
    report['sampling'] = dict(steps=a.steps,patch=p,batch=a.batch,device=a.device,torch=torch.__version__,numpy=np.__version__)
    write(out/'evaluation.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    fields=[ref[0]*std[0]+mean[0]]+[np.load(out/f'{mode}-{a.seed}.npy')[0] for mode in ['fixed','shifted']]
    limit=max(float(abs(f[0]).max()) for f in fields)
    for ax,f,title in zip(axes,fields,['Real slice','Fixed','Shifted']):
        im=ax.imshow(f[0],vmin=-limit,vmax=limit,cmap='RdBu_r'); ax.set_title(title); ax.axis('off')
    fig.colorbar(im,ax=axes,label='u (dataset units)'); fig.savefig(out/'comparison.png',dpi=140); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--manifest',required=True); prep.add_argument('--root',required=True); prep.add_argument('--out',required=True)
    tr=sub.add_parser('train'); ev=sub.add_parser('evaluate')
    for q in [tr,ev]:
        q.add_argument('--data',required=True); q.add_argument('--out',required=True); q.add_argument('--seed',type=int,default=42); q.add_argument('--steps',type=int,default=1000); q.add_argument('--batch',type=int,default=8); q.add_argument('--device',default='cpu')
    tr.add_argument('--width',type=int,default=32); tr.add_argument('--patch',type=int,default=64); tr.add_argument('--lr',type=float,default=1e-4)
    ev.add_argument('--checkpoint',required=True); ev.add_argument('--samples',type=int,default=8); ev.add_argument('--split',choices=['val','test'],default='val')
    a=p.parse_args()
    if a.command != 'prepare' and (a.steps < 1 or a.batch < 1): p.error('steps and batch must be positive')
    if a.command == 'evaluate' and a.samples < 1: p.error('samples must be positive')
    if a.command=='prepare': prepare(a.manifest,a.root,a.out)
    elif a.command=='train': train(a)
    else: evaluate(a)
if __name__=='__main__': main()
