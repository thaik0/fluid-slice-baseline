"""Build an explicitly provisional snapshot manifest from an inventory (no data copied)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from data import validate


def build(inventory, windows, counts, gap, spacing, planes, channels):
    records = {}
    layout = None
    total = 0
    for f in sorted(inventory, key=lambda x:x['file']):
        if len(f['time']) != f['sizes']['time']:
            raise ValueError('Time length differs from inventory dimension')
        if not np.isfinite(f['time']).all() or np.any(np.diff(f['time']) <= 0):
            raise ValueError('Times must be finite and strictly increasing in each file')
        if any(f['variables'].get(c) != ['time','x','y','z'] for c in channels):
            raise ValueError('Expected explicit time,x,y,z storage dimensions')
        sizes = tuple(f['sizes'][d] for d in ['z','y','x'])
        if layout is not None and layout != sizes:
            raise ValueError('Inconsistent spatial dimensions')
        layout = sizes
        for i,t in enumerate(f['time']):
            total += 1
            # Exact coordinate equality only; near matches are not silently merged.
            records.setdefault(t, dict(file=f['file'],time_index=i,time=float(t)))
    if not records or min(planes)<0 or max(planes)>=layout[0]:
        raise ValueError('No inventory or invalid z plane')
    windows = {s:tuple(map(float,windows[s])) for s in ['train','val','test']}
    for s,(lo,hi) in windows.items():
        if not np.isfinite([lo,hi]).all() or lo>=hi or counts[s]<1:
            raise ValueError('Need increasing finite bounds and positive counts')
    if windows['val'][0]-windows['train'][1] < gap or windows['test'][0]-windows['val'][1] < gap:
        raise ValueError('Requested chronological windows violate the guard gap')
    times = sorted(records)
    selected=[]; summaries={}
    for split,(lo,hi) in windows.items():
        available=np.array([t for t in times if lo<=t<=hi])
        n=counts[split]
        if len(available)<n: raise ValueError(f'Insufficient snapshots in {split} window')
        # Choose equal physical-time bins, not equally spaced filename/array indices.
        # One actual snapshot nearest its bin center, without replacement.
        edges=np.linspace(lo,hi,n+1); chosen=[]
        for i in range(n):
            candidates=available[(available>=edges[i]) & ((available<=edges[i+1]) if i==n-1 else (available<edges[i+1]))]
            if not len(candidates): raise ValueError(f'Empty time bin in {split}; reduce count or change bounds')
            t=float(candidates[np.argmin(abs(candidates-(edges[i]+edges[i+1])/2))])
            chosen.append(t); selected.append(dict(records[t],split=split))
        summaries[split]=dict(snapshots=n,planes=n*len(planes),first_time=min(chosen),last_time=max(chosen),requested_window=[lo,hi])
    m=dict(status='provisional_engineering_split_not_independence_validated',axes=['z','y','x'],dims=['time','z','y','x'],channels=channels,shape=list(layout[1:]),slice_indices=planes,spacing=spacing,verify_spacing=True,minimum_time_gap=gap,snapshots=selected,selection=dict(method='nearest_center_of_equal_physical_time_bins',duplicate_policy='exact_time_keep_lexicographically_first_filename',units='unspecified_dataset_units',available_records=total,unique_times=len(records),duplicate_time_records_omitted=total-len(records),summaries=summaries,warning='Coordinate duplicates are not verified identical fields. Gap is provisional and does not establish statistical independence. Unselected records are excluded.'))
    return validate(m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True); p.add_argument('--out',required=True)
    for s in ['train','val','test']:
        p.add_argument(f'--{s}-window',nargs=2,type=float,required=True)
        p.add_argument(f'--{s}-count',type=int,required=True)
    p.add_argument('--gap',type=float,required=True); p.add_argument('--spacing',type=float,required=True)
    p.add_argument('--planes',nargs='+',type=int,default=[0,96,192,288])
    p.add_argument('--channels',nargs='+',default=['u','v','w'])
    a=p.parse_args(); path=Path(a.inventory)
    m=build(json.loads(path.read_text()),{s:getattr(a,s+'_window') for s in ['train','val','test']},{s:getattr(a,s+'_count') for s in ['train','val','test']},a.gap,a.spacing,a.planes,a.channels)
    m['selection']['inventory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(m,indent=2))
    print(json.dumps(m['selection'],indent=2)); print(f'Wrote {out}; this is a provisional split.')
if __name__=='__main__': main()
