"""Fresh validation contexts and a matched conditioning ablation; saved weights only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from data import digest
from model import FlowNet
from prepare_coarse import file_hash
from refinement import check,load_pair,low,high,spectrum
from run import seed,sync,write


WARNING=('Real coarse modes are supplied; one training seed and provisional temporal split. '
         'New plane indices/noises are not independent snapshots or a reserved test set. '
         'Shuffling measures sensitivity and teacher-forced context usefulness, not conditional distribution fidelity.')


def select_contexts(count,samples,shuffle_samples):
    """Exclude the prior eight generated contexts and fixed t=.5 validation planes."""
    previous=set(np.linspace(0,count-1,8,dtype=int).tolist())
    excluded=previous | set(range(min(8,count)))
    eligible=[i for i in range(count) if i not in excluded]
    if not 2<=shuffle_samples<=samples<=len(eligible):raise ValueError('Not enough unused validation planes or invalid subset sizes')
    ids=np.array(eligible)[np.linspace(0,len(eligible)-1,samples,dtype=int)].tolist()
    subset=np.linspace(0,samples-1,shuffle_samples,dtype=int).tolist()
    donors=np.roll(np.array(subset),shuffle_samples//2).tolist()
    return ids,sorted(excluded),dict(zip(subset,donors))


def plan(data,coarse,run,samples=32,shuffle_samples=8,steps=64):
    if steps<1:raise ValueError('Positive sampling steps required')
    data=Path(data);coarse=Path(coarse);path=Path(run)/'train/checkpoint-3000.pt'
    m=json.loads((data/'manifest.json').read_text());cm=json.loads((coarse/'manifest.json').read_text());norm=json.loads((data/'normalization.json').read_text())
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['kind']!='real_coarse_conditional_residual' or ck['history'][-1]['step']!=3000:raise ValueError('Expected saved 3000-update residual checkpoint')
    if ck['manifest']!=m or ck['coarse_manifest']!=cm or ck['normalization']!=norm:raise ValueError('Checkpoint/data mismatch')
    snapshots=[r for r in m['snapshots'] if r['split']=='val'];planes=len(m['slice_indices'])
    ids,excluded,donors=select_contexts(len(snapshots)*planes,samples,shuffle_samples)
    rows=[]
    for j,idx in enumerate(ids):
        rows.append(dict(validation_index=idx,noise_seed=6242+j,snapshot_row=idx//planes,time=snapshots[idx//planes]['time'],slice_index=m['slice_indices'][idx%planes],donor_validation_index=ids[donors[j]] if j in donors else None))
    return dict(status='passed',checkpoint=str(path),checkpoint_sha256=file_hash(path),training_update=3000,training_seed=ck['args']['seed'],model_config=ck['config'],fine_manifest_sha256=digest(m),coarse_manifest_sha256=digest(cm),samples=samples,shuffle_samples=shuffle_samples,steps=steps,excluded_validation_indices=excluded,contexts=rows,warning=WARNING)


@torch.no_grad()
def conditioned_sample(model,carrier,conditioning,noise,c,steps):
    """Hold output low modes fixed; intervene only on the model's context input."""
    x=high(noise,c)
    for i in range(steps):x=high(x+high(model(x,i/steps,conditioning),c)/steps,c)
    return carrier+x


def stats(reference,generated,c):
    rr=np.mean(reference,axis=0);rg=np.mean(generated,axis=0)
    if not np.isfinite(rr).all() or not np.isfinite(rg).all() or rr.sum()<=0:raise ValueError('Invalid residual spectra')
    edges=[c//2,2*c//3,c,3*c//2,len(rr)]
    return dict(reference_residual_spectrum=rr.tolist(),generated_residual_spectrum=rg.tolist(),residual_energy_ratio=float(rg.sum()/rr.sum()),residual_spectrum_relative_l1=float(abs(rg-rr).sum()/rr.sum()),bands=[dict(first_shell=lo,last_shell=hi-1,energy_ratio=float(rg[lo:hi].sum()/max(rr[lo:hi].sum(),1e-30))) for lo,hi in zip(edges[:-1],edges[1:])])


def evaluate(a,p):
    seed(6242);out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);write(out/'plan.json',p)
    m,cm,norm,_,arrays=load_pair(a.data,a.coarse);val=arrays['val'][0];c=cm['shape'][0]
    ck=torch.load(p['checkpoint'],map_location=a.device,weights_only=False)
    model=FlowNet(**ck['config']).to(a.device);model.load_state_dict(ck['model']);model.eval()
    sd=np.array(norm['std'])[None,:,None,None]
    refs=[];gens=[];subsetrefs=[];subsetgens=[];shuffled=[];rows=[];views=None
    for row in p['contexts']:
        row=dict(row);i=row['validation_index'];real=torch.as_tensor(np.array(val[i:i+1]),device=a.device)
        ctx=low(real,c);target=high(real,c)
        noise=torch.randn(real.shape,device=a.device,generator=torch.Generator(device=a.device).manual_seed(row['noise_seed']))
        sync(a.device);start=time.perf_counter();pred=conditioned_sample(model,ctx,ctx,noise,c,p['steps']);sync(a.device)
        row['sampling_seconds']=time.perf_counter()-start
        if not torch.isfinite(pred).all():raise ValueError('Nonfinite generated field')
        row['max_normalized_low_mode_error']=float(abs(low(pred,c)-ctx).max().item())
        if row['max_normalized_low_mode_error']>1e-4:raise ValueError('Supplied low modes changed')
        rr=target.cpu().numpy()*sd;rg=(pred-ctx).cpu().numpy()*sd
        rs=spectrum(rr);gs=spectrum(rg);refs.append(rs);gens.append(gs)
        row['reference_residual_spectrum']=rs.tolist();row['generated_residual_spectrum']=gs.tolist()
        donor=row['donor_validation_index']
        if donor is not None:
            other=torch.as_tensor(np.array(val[donor:donor+1]),device=a.device);wrong=low(other,c)
            sync(a.device);start=time.perf_counter();alt=conditioned_sample(model,ctx,wrong,noise,c,p['steps']);sync(a.device)
            row['shuffled_sampling_seconds']=time.perf_counter()-start
            if not torch.isfinite(alt).all():raise ValueError('Nonfinite ablation field')
            error=float(abs(low(alt,c)-ctx).max().item());row['shuffled_max_normalized_low_mode_error']=error
            if error>1e-4:raise ValueError('Ablation changed the carrier low modes')
            alt_r=(alt-ctx).cpu().numpy()*sd;subsetrefs.append(rs);subsetgens.append(gs);shuffled.append(spectrum(alt_r))
            row['shuffled_residual_spectrum']=shuffled[-1].tolist()
            row['correct_residual_mean_square_physical']=float(np.mean(rg**2))
            row['shuffled_minus_correct_mean_square_physical']=float(np.mean((alt_r-rg)**2))
            # The same held-out residual/noise state tests correct vs mismatched context.
            with torch.no_grad():
                hn=high(noise,c);state=(hn+target)/2;velocity=target-hn
                good=high(model(state,.5,ctx),c);bad=high(model(state,.5,wrong),c)
                if not torch.isfinite(good).all() or not torch.isfinite(bad).all():raise ValueError('Nonfinite context probe')
                row['teacher_forced_t05_correct_mse']=float((good-velocity).square().mean().item())
                row['teacher_forced_t05_shuffled_mse']=float((bad-velocity).square().mean().item())
            if views is None:views=[rr[0,0],rg[0,0],alt_r[0,0]]
        rows.append(row);write(out/'partial.json',dict(plan=p,samples=rows))
        print(f'Fresh context {len(rows)}/{len(p["contexts"])}',flush=True)
    selected=[r for r in rows if r['donor_validation_index'] is not None]
    good_mse=np.array([r['teacher_forced_t05_correct_mse'] for r in selected]);bad_mse=np.array([r['teacher_forced_t05_shuffled_mse'] for r in selected])
    response=np.sqrt(sum(r['shuffled_minus_correct_mean_square_physical'] for r in selected)/max(sum(r['correct_residual_mean_square_physical'] for r in selected),1e-30))
    report=dict(plan=p,samples=rows,all_fresh=stats(refs,gens,c),matched_subset_correct=stats(subsetrefs,subsetgens,c),matched_subset_shuffled=stats(subsetrefs,shuffled,c),conditioning=dict(residual_change_rms_over_correct_rms=float(response),teacher_forced_t05_correct_mse=float(good_mse.mean()),teacher_forced_t05_shuffled_mse=float(bad_mse.mean()),teacher_forced_t05_shuffled_over_correct=float(bad_mse.mean()/max(good_mse.mean(),1e-30)),contexts_with_lower_correct_mse=int((good_mse<bad_mse).sum()),contexts=len(selected)),evaluation_source_hashes={f:file_hash(f) for f in ['confirm_refinement.py','refinement.py','model.py']},warning=WARNING)
    write(out/'confirmation.json',report)
    lines=['# Fresh-context refinement confirmation','',WARNING,'','| Evaluation | Residual / real | Spectrum relative L1 |','|---|---|---|']
    for label,key in [(f'{len(rows)} fresh contexts','all_fresh'),('Matched subset: correct context','matched_subset_correct'),('Matched subset: shuffled context','matched_subset_shuffled')]:
        s=report[key];lines.append(f'| {label} | {s["residual_energy_ratio"]:.3f} | {s["residual_spectrum_relative_l1"]:.3f} |')
    lines+=['',f'Conditioning response RMS / correct residual RMS: {response:.3f}',f'Teacher-forced shuffled / correct MSE: {report["conditioning"]["teacher_forced_t05_shuffled_over_correct"]:.3f}','','Output low modes and initial noise are identical in each correct/shuffled pair. Only model conditioning changes. Spectrum agreement alone does not establish useful conditioning.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,keys in zip(axes,[['all_fresh'],['matched_subset_correct','matched_subset_shuffled']]):
        ref=np.array(report[keys[0]]['reference_residual_spectrum']);k=np.arange(len(ref));mask=k>=c//2
        ax.semilogy(k[mask],ref[mask],color='black',label='Matched real residual')
        for key in keys:ax.semilogy(k[mask],np.array(report[key]['generated_residual_spectrum'])[mask],label=key.replace('_',' '))
        ax.set(xlabel='Planar radial shell',ylabel='Residual energy');ax.legend()
    fig.savefig(out/'spectra.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained');lim=max(abs(v).max() for v in views)
    for ax,v,title in zip(axes,views,['Real residual u','Correct-context residual u','Shuffled-context residual u']):
        im=ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_title(title);ax.axis('off')
    fig.colorbar(im,ax=list(axes),shrink=.7);fig.savefig(out/'conditioning.png',dpi=140);plt.close(fig)
    print((out/'summary.md').read_text(),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['preflight','evaluate']);ap.add_argument('--data',required=True);ap.add_argument('--coarse',required=True);ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--samples',type=int,default=32);ap.add_argument('--shuffle-samples',type=int,default=8);ap.add_argument('--steps',type=int,default=64)
    a=ap.parse_args();p=plan(a.data,a.coarse,a.run,a.samples,a.shuffle_samples,a.steps)
    if a.command=='preflight':
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);geometry=check(a.data,a.coarse,out.with_suffix('.geometry.json'));p['geometry_status']=geometry['status'];write(out,p);print(json.dumps(p,indent=2))
    else:evaluate(a,p)


if __name__=='__main__':main()
