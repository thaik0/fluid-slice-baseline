"""Collect completed screen arms without choosing a winner or touching test data."""
import argparse
import json
from pathlib import Path


def collect(root,arm):
    rows=[]; manifests=set()
    for path in sorted(Path(root).glob('*/diagnostics/diagnostics.json')):
        d=json.loads(path.read_text()); t=json.loads((path.parent.parent/'train/training.json').read_text())
        manifests.add(d['manifest_sha256'])
        if arm not in d['summary']: raise ValueError(f'Missing arm {arm}')
        r=d['summary'][arm]
        rows.append(dict(name=path.parent.parent.name,parameters=t['parameters'],training_seconds=t['seconds'],patch=t['args']['patch'],updates=t['args']['steps'],training_seed=t['args']['seed'],validation_t05=t['history'][-1]['val_loss_t05'],**{k:r[k]['mean'] for k in ['energy_ratio','fluctuation_ratio','low_ratio','high_ratio','spectrum_l1','seam_yx','mean']}))
    if not rows: raise ValueError('No completed screen reports found')
    if len(manifests)!=1: raise ValueError('Cannot compare different manifests')
    return dict(sampling_arm=arm,manifest_sha256=manifests.pop(),runs=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root'); p.add_argument('--arm',default='fixed-euler-64')
    a=p.parse_args(); report=collect(a.root,a.arm); root=Path(a.root)
    (root/'screen-summary.json').write_text(json.dumps(report,indent=2))
    lines=['# Training screen','',f'Common sampling control: {a.arm}. Validation only; training-seed repeats remain exploratory.','', '| Run | Parameters | Window | Updates | Train seconds | Low / real | High / real | Spectrum L1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['runs']:
        lines.append(f"| {r['name']} | {r['parameters']} | {r['patch']} | {r['updates']} | {r['training_seconds']:.1f} | {r['low_ratio']:.3f} | {r['high_ratio']:.3f} | {r['spectrum_l1']:.3f} |")
    text='\n'.join(lines)+'\n'; (root/'screen-summary.md').write_text(text); print(text)
if __name__=='__main__': main()
