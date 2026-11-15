"""Summarize a complete whole-plane factorial with within-training-seed contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np

TREATMENTS=[('group','zeros'),('group','circular'),('none','zeros'),('none','circular')]
KEYS=['energy_ratio','fluctuation_ratio','low_ratio','high_ratio','spectrum_l1','mean_flow_energy_ratio']


def collect(root):
    runs={}; manifests=set(); reference_hashes=set(); settings=set(); sampling=set()
    for path in sorted(Path(root).glob('*/diagnostics/diagnostics.json')):
        d=json.loads(path.read_text()); t=json.loads((path.parent.parent/'train/training.json').read_text()); a=t['args']
        key=(a['seed'],a['model_norm'],a['padding'])
        if key in runs:raise ValueError('Duplicate training seed/treatment')
        settings.add((a['width'],a['steps'],a['batch'],a['patch'],a['lr']))
        manifests.add(d['manifest_sha256']);reference_hashes.add(json.dumps(d['reference'],sort_keys=True))
        sampling.add(tuple((s['arm'],s['seed']) for s in d['samples']))
        if a['patch']!=384:raise ValueError('Expected 384-square whole-plane experiment')
        ref=d['reference']; arms={}
        for arm in ['fixed-euler-64','late-euler-64','all-euler-64']:
            ss=[s['metrics'] for s in d['samples'] if s['arm']==arm]
            if not ss:raise ValueError(f'Missing arm {arm}')
            r=d['summary'][arm]
            arms[arm]={k:r[k]['mean'] for k in KEYS if k!='mean_flow_energy_ratio'}
            arms[arm]['mean_flow_energy_ratio']=float(np.mean([m['energy']-m['fluctuation_energy'] for m in ss])/(ref['energy']-ref['fluctuation_energy']))
            arms[arm]['component_mean']=np.mean([m['mean'] for m in ss],axis=0).tolist()
        runs[key]=dict(seed=key[0],normalization=key[1],padding=key[2],parameters=t['parameters'],seconds=t['seconds'],peak_cuda_bytes=t['peak_cuda_bytes'],arms=arms)
    expected={(s,n,p) for s in [42,43,44] for n,p in TREATMENTS}
    if set(runs)!=expected:raise ValueError(f'Incomplete/unexpected factorial. Missing: {sorted(expected-set(runs))}; extra: {sorted(set(runs)-expected)}')
    if len(manifests)!=1 or len(reference_hashes)!=1 or len(settings)!=1 or len(sampling)!=1:raise ValueError('Data/reference/training/sampling settings differ')
    contrasts=[]; aggregate=[]
    for arm in ['fixed-euler-64','late-euler-64','all-euler-64']:
        for seed in [42,43,44]:
            a,b,c,d=[runs[seed,n,p]['arms'][arm] for n,p in TREATMENTS]
            # Main effects averaged over the other factor; interaction is
            # difference of padding effects. Raw contrasts are descriptive.
            contrasts.append(dict(seed=seed,arm=arm,effects={k:dict(remove_groupnorm=((c[k]-a[k])+(d[k]-b[k]))/2,periodic_padding=((b[k]-a[k])+(d[k]-c[k]))/2,interaction=d[k]-c[k]-b[k]+a[k]) for k in KEYS}))
        for n,p in TREATMENTS:
            aggregate.append(dict(arm=arm,normalization=n,padding=p,statistics={k:dict(mean=float(np.mean([runs[s,n,p]['arms'][arm][k] for s in [42,43,44]])),min=float(min(runs[s,n,p]['arms'][arm][k] for s in [42,43,44])),max=float(max(runs[s,n,p]['arms'][arm][k] for s in [42,43,44]))) for k in KEYS}))
    report=dict(manifest_sha256=manifests.pop(),runs=list(runs.values()),aggregate=aggregate,within_seed_effects=contrasts,primary_sampler='fixed-euler-64',warning='Three training seeds; ranges are not confidence intervals. Mean-flow energy is descriptive, not a zero-mean constraint.')
    json.dumps(report,allow_nan=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args()
    r=collect(a.root);root=Path(a.root)
    (root/'factorial-summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    lines=['# Whole-plane normalization / boundary factorial','','Primary sampler: fixed Euler64. Three training seeds; ranges are descriptive.','', '| Model normalization | Padding | Low / real | High / real | Spectrum error | Mean-flow energy / real |','|---|---|---:|---:|---:|---:|']
    for row in r['aggregate']:
        if row['arm']!='fixed-euler-64':continue
        vals=[row['statistics'][k]['mean'] for k in ['low_ratio','high_ratio','spectrum_l1','mean_flow_energy_ratio']]
        lines.append('| '+row['normalization']+' | '+row['padding']+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    text='\n'.join(lines)+'\n';(root/'factorial-summary.md').write_text(text); print(text)
if __name__=='__main__':main()
