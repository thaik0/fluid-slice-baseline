"""Descriptive per-plane mean distributions, without asserting independent slices."""
import numpy as np
import torch


def distribution(values):
    a=np.asarray(values,dtype='float64')
    if a.ndim!=2 or len(a)<1 or not np.isfinite(a).all():raise ValueError('Need finite sample-by-channel means')
    centered=a-a.mean(0)
    return dict(count=len(a),mean=a.mean(0).tolist(),std=a.std(0).tolist(),covariance=(centered.T@centered/len(a)).tolist(),quantiles_05_50_95=np.quantile(a,[.05,.5,.95],axis=0).tolist(),mean_flow_energy=float(.5*np.mean(np.sum(a*a,axis=1))))


def wasserstein_1d(a,b):
    a=np.sort(a);b=np.sort(b);grid=np.unique(np.concatenate([a,b]))
    return float(np.sum(abs(np.searchsorted(a,grid[:-1],side='right')/len(a)-np.searchsorted(b,grid[:-1],side='right')/len(b))*np.diff(grid)))


@torch.no_grad()
def collect_mean_ensemble(model,ck,val,manifest,args,existing,integrator):
    n=args.mean_samples; channels=ck['config']['channels']; h,w=manifest['shape'];patch=ck['args']['patch'];norm=ck['normalization']
    mu=np.array(norm['mean']);sd=np.array(norm['std']);planes=len(manifest['slice_indices'])
    snapshots=[r for r in manifest['snapshots'] if r['split']=='val']
    if len(val)!=len(snapshots)*planes:raise ValueError('Reference array does not match snapshot-plane mapping')
    real=val.mean((-2,-1),dtype='float64')*sd+mu
    reuse={s['seed']:s['metrics']['mean'] for s in existing if s['arm']=='fixed-euler-64'}
    generated=[]; seeds=list(range(args.seed,args.seed+n))
    for s in seeds:
        if s in reuse: avg=reuse[s]
        else:
            noise=torch.randn((1,channels,h,w),device=args.device,generator=torch.Generator(device=args.device).manual_seed(s))
            pred,_=integrator(model,noise,64,patch,'fixed',[(0,0)]*32,'euler',args.batch,norm)
            avg=pred.double().mean((0,2,3)).cpu().numpy()*sd+mu
        generated.append(np.asarray(avg).tolist())
        if len(generated)%8==0:print(f'Mean ensemble {len(generated)}/{n}',flush=True)
    generated=np.array(generated)
    w1=np.array([wasserstein_1d(generated[:,c],real[:,c]) for c in range(channels)])
    labels=[dict(snapshot_time=r['time'],slice_index=z) for r in snapshots for z in manifest['slice_indices']]
    return dict(sampler='fixed-euler-64',channels=manifest['channels'],generated_seeds=seeds,generated_means=generated.tolist(),reference_plane_means=real.tolist(),reference_labels=labels,generated=distribution(generated),reference=distribution(real),component_wasserstein_dataset_units=w1.tolist(),component_wasserstein_training_std=(w1/sd).tolist(),warning='Reference planes within snapshots and nearby times are correlated. These are descriptive distribution comparisons, not confidence intervals or zero-mean constraints.')
