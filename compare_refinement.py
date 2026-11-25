"""Matched checkpoint/sampling comparison; evaluates existing weights only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data import digest
from prepare_coarse import file_hash
from refinement import check,evaluate
from run import write

UPDATES=(1000,3000,5000)
SAMPLING_STEPS=(64,128)


def preflight(data,coarse,run,samples):
    if samples<1:raise ValueError('Positive sample count required')
    run=Path(run)
    m=json.loads((Path(data)/'manifest.json').read_text())
    cm=json.loads((Path(coarse)/'manifest.json').read_text())
    norm=json.loads((Path(data)/'normalization.json').read_text())
    count=sum(r['split']=='val' for r in m['snapshots'])*len(m['slice_indices'])
    if samples>count:raise ValueError('Not enough distinct validation contexts')
    paths={};common=None
    for update in UPDATES:
        path=run/'train'/f'checkpoint-{update}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['kind']!='real_coarse_conditional_residual' or ck['history'][-1]['step']!=update:
            raise ValueError('Checkpoint update/kind mismatch')
        if ck['manifest']!=m or ck['coarse_manifest']!=cm or ck['normalization']!=norm:
            raise ValueError('Checkpoint/data mismatch')
        signature={k:ck[k] for k in ['config','args','source_hashes']}
        if common is None:common=signature
        elif common!=signature:raise ValueError('Checkpoints are not from the same training configuration')
        paths[str(update)]=dict(path=str(path),sha256=file_hash(path))
    return dict(status='passed',fine_manifest_sha256=digest(m),coarse_manifest_sha256=digest(cm),checkpoints=paths,samples=samples,training_seed=common['args']['seed'],model_config=common['config'],sampling_steps=list(SAMPLING_STEPS),warning='Evaluation only; real coarse contexts, one training seed, provisional split. No test data or new training.')


def summarize(reports,coarse_size):
    rows=[];reference=None;keys=set()
    for update,steps,d in reports:
        key=(update,steps)
        if key in keys:raise ValueError('Duplicate comparison arm')
        keys.add(key)
        if steps!=d['args']['steps']:raise ValueError('Sampling-step mismatch')
        pairing=[(s['validation_index'],s['seed']) for s in d['samples']]
        signature=(pairing,d['fine_manifest_sha256'],d['coarse_manifest_sha256'],d['model_config'])
        rr=np.asarray(d['reference_residual_spectrum']);rg=np.asarray(d['generated_residual_spectrum'])
        if reference is None:reference=(signature,rr)
        elif signature!=reference[0] or not np.allclose(rr,reference[1],rtol=1e-12,atol=1e-16):
            raise ValueError('Unmatched contexts/noise, model or reference spectra')
        if len(rr)!=len(rg) or not np.isfinite(rg).all() or rr.sum()<=0:raise ValueError('Invalid spectrum')
        error=max(s['max_normalized_low_mode_error'] for s in d['samples'])
        if error>1e-4:raise ValueError('Supplied coarse modes changed')
        edges=[coarse_size//2,2*coarse_size//3,coarse_size,3*coarse_size//2,len(rr)]
        bands=[dict(first_shell=lo,last_shell=hi-1,energy_ratio=float(rg[lo:hi].sum()/max(rr[lo:hi].sum(),1e-30))) for lo,hi in zip(edges[:-1],edges[1:])]
        rows.append(dict(training_update=update,sampling_steps=steps,residual_energy_ratio=float(rg.sum()/rr.sum()),residual_spectrum_relative_l1=float(abs(rg-rr).sum()/rr.sum()),bands=bands,max_normalized_low_mode_error=error,sampling_seconds=sum(s['seconds'] for s in d['samples']),checkpoint_sha256=d['checkpoint_sha256']))
    if keys!={(u,s) for u in UPDATES for s in SAMPLING_STEPS}:raise ValueError('Incomplete six-arm comparison')
    changes=[]
    for update in UPDATES:
        by_steps={s:d for u,s,d in reports if u==update}
        g_start=np.asarray(by_steps[SAMPLING_STEPS[0]]['generated_residual_spectrum'])
        g_end=np.asarray(by_steps[SAMPLING_STEPS[-1]]['generated_residual_spectrum'])
        changes.append(dict(training_update=update,from_steps=SAMPLING_STEPS[0],to_steps=SAMPLING_STEPS[-1],spectrum_change_relative_to_real_residual=float(abs(g_end-g_start).sum()/reference[1].sum())))
    return dict(rows=rows,sampling_changes=changes,matched_contexts_and_noise=reference[0][0],warning='Real coarse modes are supplied. Compare residuals only; this cannot establish end-to-end generation or conditional diversity. No automatic choice of more training.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','evaluate']);p.add_argument('--data',required=True);p.add_argument('--coarse',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--samples',type=int,default=8)
    a=p.parse_args();plan=preflight(a.data,a.coarse,a.run,a.samples)
    out=Path(a.out)
    if a.command=='preflight':
        out.parent.mkdir(parents=True,exist_ok=True)
        geometry=check(a.data,a.coarse,out.with_suffix('.geometry.json'))
        plan['geometry_status']=geometry['status'];write(out,plan);print(json.dumps(plan,indent=2));return
    if out.exists() and any(out.iterdir()):raise ValueError('Use fresh comparison directory')
    out.mkdir(parents=True,exist_ok=True);write(out/'plan.json',plan)
    reports=[]
    for update in UPDATES:
        for steps in SAMPLING_STEPS:
            dest=out/f'update-{update}-euler-{steps}'
            args=argparse.Namespace(command='evaluate',data=a.data,coarse=a.coarse,checkpoint=plan['checkpoints'][str(update)]['path'],out=str(dest),device=a.device,samples=a.samples,steps=steps)
            print(f'Evaluating saved update {update}, Euler {steps}',flush=True)
            evaluate(args)
            report=json.loads((dest/'evaluation.json').read_text());reports.append((update,steps,report))
            write(out/'progress.json',dict(completed=[dict(training_update=u,sampling_steps=s) for u,s,_ in reports]))
    coarse_size=json.loads((Path(a.coarse)/'manifest.json').read_text())['shape'][0]
    summary=summarize(reports,coarse_size);summary['plan']=plan;write(out/'comparison.json',summary)
    labels=[f'{b["first_shell"]}–{b["last_shell"]}' for b in summary['rows'][0]['bands']]
    lines=['# Saved refinement checkpoint comparison','',summary['warning'],'','All six arms use the same contexts/noise seeds; results are descriptive.','','| Updates | Euler steps | Residual / real | Residual spectrum L1 | '+' | '.join(labels)+' |','|---|---|---|---|---|---|---|---|']
    for r in summary['rows']:
        lines.append(f'| {r["training_update"]} | {r["sampling_steps"]} | {r["residual_energy_ratio"]:.3f} | {r["residual_spectrum_relative_l1"]:.3f} | '+' | '.join(f'{b["energy_ratio"]:.3f}' for b in r['bands'])+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for steps in SAMPLING_STEPS:
        selected=[r for r in summary['rows'] if r['sampling_steps']==steps]
        for ax,key in zip(axes,['residual_energy_ratio','residual_spectrum_relative_l1']):ax.plot([r['training_update'] for r in selected],[r[key] for r in selected],'o-',label=f'Euler{steps}')
    axes[0].axhline(1,color='black',ls='--');axes[0].set(ylabel='Residual energy / real');axes[1].set(ylabel='Residual spectrum relative L1')
    for ax in axes:ax.set(xlabel='Training update');ax.legend()
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print((out/'summary.md').read_text(),flush=True)


if __name__=='__main__':main()
