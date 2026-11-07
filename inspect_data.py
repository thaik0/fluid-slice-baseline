"""Inventory only. Never assigns splits or guesses which time coordinates overlap."""
import argparse
import json
from pathlib import Path
import xarray as xr
p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--pattern',default='fluid_*.nc'); p.add_argument('--time-dim',default='time'); p.add_argument('--out',required=True)
a=p.parse_args(); records=[]
for path in sorted(Path(a.root).glob(a.pattern)):
    with xr.open_dataset(path, decode_times=False) as ds:
        records.append(dict(file=path.name,sizes=dict(ds.sizes),variables={c:list(ds[c].dims) for c in ['u','v','w'] if c in ds},time=ds[a.time_dim].values.tolist(),time_attributes=dict(ds[a.time_dim].attrs)))
if not records: raise ValueError('No files found')
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(records,indent=2,default=str))
print(f'Inventoried {len(records)} files. Confirm axis mapping, time units and duplicate times before assigning splits.')
