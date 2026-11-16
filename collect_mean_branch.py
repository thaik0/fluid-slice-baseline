"""Collect a complete three-seed mean-branch comparison; validation only."""
import argparse
import json
from pathlib import Path
import numpy as np


def collect(root):
    runs={};settings=set();manifests=set();references=set();sampling=set();gen_seeds=set()
    for path in sorted(Path(root).glob('*/diagnostics/diagnostics.json')):
        d=json.loads(path.read_text());t=json.loads((path.parent.parent/'train/training.json').read_text());a=t['args'];m=d['mean_distribution'];ref=d['reference']
        key=(a['seed'],a['mean_mode'])
        if key in runs:raise ValueError('Duplicate seed/treatment')
        c=d['model_config']
        if c['normalization']!='group' or c['padding']!='circular' or c['mean_mode']!=a['mean_mode']:raise ValueError('Unexpected model configuration')
        if a['patch']!=384 or m['sampler']!='fixed-euler-64' or len(m['generated_seeds'])!=64:raise ValueError('Unexpected window or ensemble')
        settings.add((a['width'],a['steps'],a['batch'],a['patch'],a['lr']))
        manifests.add(d['manifest_sha256']);references.add(json.dumps(dict(metrics=ref,means=m['reference_plane_means'],labels=m['reference_labels']),sort_keys=True))
        sampling.add(tuple((s['arm'],s['seed']) for s in d['samples']));gen_seeds.add(tuple(m['generated_seeds']))
        s=d['summary']['fixed-euler-64']
        runs[key]=dict(seed=key[0],mean_mode=key[1],parameters=t['parameters'],seconds=t['seconds'],generation_seeds=m['generated_seeds'],generated_mean_distribution=m['generated'],reference_mean_distribution=m['reference'],wasserstein_training_std=m['component_wasserstein_training_std'],mean_flow_energy_ratio=m['generated']['mean_flow_energy']/m['reference']['mean_flow_energy'],spectral_screen={k:s[k]['mean'] for k in ['energy_ratio','fluctuation_ratio','low_ratio','high_ratio','spectrum_l1']})
    expected={(s,m) for s in [42,43,44] for m in ['implicit','learned']}
    if set(runs)!=expected:raise ValueError(f'Incomplete comparison. Missing: {sorted(expected-set(runs))}')
    if any(len(v)!=1 for v in [settings,manifests,references,sampling,gen_seeds]):raise ValueError('Experiment settings, data, references or sampling differ')
    paired=[]
    for seed in [42,43,44]:
        c=runs[seed,'implicit'];l=runs[seed,'learned']
        paired.append(dict(seed=seed,wasserstein_training_std_delta=(np.array(l['wasserstein_training_std'])-c['wasserstein_training_std']).tolist(),mean_flow_energy_ratio_delta=l['mean_flow_energy_ratio']-c['mean_flow_energy_ratio'],spectral_delta={k:l['spectral_screen'][k]-c['spectral_screen'][k] for k in c['spectral_screen']}))
    report=dict(manifest_sha256=manifests.pop(),runs=list(runs.values()),paired_learned_minus_implicit=paired,warning='Mean distributions use 64 generation seeds; spectral screen uses four. Three training seeds; real planes are correlated. No confidence intervals or winner selection.')
    json.dumps(report,allow_nan=False);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args();r=collect(a.root);out=Path(a.root)
    (out/'mean-summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    lines=['# Learned mean branch comparison','','Spectral screen: four matched generation seeds. Mean distributions: 64 matched generation seeds.','', '| Training seed | Mean mode | Low / real | High / real | Spectrum error | Mean-flow energy / real | Mean-distribution W1 / training std (u,v,w) |','|---|---|---:|---:|---:|---:|---|']
    for row in sorted(r['runs'],key=lambda x:(x['seed'],x['mean_mode'])):
        s=row['spectral_screen'];w=', '.join(f'{v:.3f}' for v in row['wasserstein_training_std'])
        lines.append(f"| {row['seed']} | {row['mean_mode']} | {s['low_ratio']:.3f} | {s['high_ratio']:.3f} | {s['spectrum_l1']:.3f} | {row['mean_flow_energy_ratio']:.3f} | {w} |")
    text='\n'.join(lines)+'\n';(out/'mean-summary.md').write_text(text);print(text)
if __name__=='__main__':main()
