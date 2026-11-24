"""Combine the existing seed42 coarse pilot with completed seed43/44 repeats."""
import argparse
import json
from pathlib import Path
import numpy as np


def collect(pilot,replication):
    runs=[];manifests=set();settings=set();configs=set();ref=None
    for seed,path in [(42,Path(pilot)),(43,Path(replication)/'seed-43'),(44,Path(replication)/'seed-44')]:
        t=json.loads((path/'train/training.json').read_text());d=json.loads((path/'eval/coarse-evaluation.json').read_text());a=t['args']
        if a['seed']!=seed:raise ValueError('Unexpected training seed')
        c=d['model_config']
        if c.get('normalization')!='group' or c.get('padding')!='circular' or c.get('mean_mode','implicit')!='implicit':raise ValueError('Unexpected model variant')
        if a['patch']!=96 or a['steps']!=5000 or a['width']!=32 or d['args']['steps']!=64:raise ValueError('Unexpected coarse training/sampling settings')
        if [s['seed'] for s in d['samples']]!=list(range(4242,4250)):raise ValueError('Need eight matched sampling seeds')
        manifests.add(d['manifest_sha256']);settings.add((a['batch'],a['lr'],a['steps'],a['width'],a['patch'],t['parameters'],d['args']['steps']))
        configs.add(json.dumps(c,sort_keys=True))
        r=np.array(d['reference']['spectrum'])[1:17]
        if ref is None:ref=r
        elif not np.allclose(ref,r,rtol=1e-10,atol=1e-14):raise ValueError('Reference shell energies differ')
        spec=np.array([s['metrics']['spectrum'][1:17] for s in d['samples']]);ratio=spec.mean(0)/ref
        means=np.array([s['metrics']['mean'] for s in d['samples']])
        runs.append(dict(training_seed=seed,parameters=t['parameters'],training_seconds=t['seconds'],training_peak_cuda_bytes=t['peak_cuda_bytes'],shell_ratio_1_16=ratio.tolist(),shells1_4_ratio=float(spec.mean(0)[:4].sum()/ref[:4].sum()),relative_l1_shells1_16=float(abs(spec.mean(0)-ref).sum()/ref.sum()),component_mean=means.mean(0).tolist(),component_mean_std_eight_samples=means.std(0).tolist(),final_validation_t05=t['history'][-1]['val_loss_t05']))
    if len(manifests)!=1 or len(settings)!=1 or len(configs)!=1:raise ValueError('Data or experiment configuration differs')
    ratios=np.array([r['shell_ratio_1_16'] for r in runs])
    return dict(manifest_sha256=manifests.pop(),runs=runs,shells1_16_across_training_seeds=dict(mean=ratios.mean(0).tolist(),min=ratios.min(0).tolist(),max=ratios.max(0).tolist()),warning='Three training seeds. Per-model eight generation seeds are matched, not additional training replicates. Ranges are descriptive; mean-flow bias is reported separately. Common physical low shells only.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True);p.add_argument('--replication',required=True);a=p.parse_args()
    report=collect(a.pilot,a.replication);out=Path(a.replication)
    (out/'coarse-replication-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Coarse whole-domain replication','','Fixed Euler64, eight matched generation seeds per trained model.','', '| Training seed | Shell1 / real | Shells1–4 / real | Relative L1, shells1–16 | Training seconds |','|---|---:|---:|---:|---:|']
    for r in report['runs']:lines.append(f"| {r['training_seed']} | {r['shell_ratio_1_16'][0]:.3f} | {r['shells1_4_ratio']:.3f} | {r['relative_l1_shells1_16']:.3f} | {r['training_seconds']:.1f} |")
    text='\n'.join(lines)+'\n';(out/'coarse-replication-summary.md').write_text(text);print(text)
if __name__=='__main__':main()
