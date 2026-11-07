"""Synthetic plumbing test, not synthetic turbulence or a quality experiment."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import xarray as xr
root=Path('runs/smoke'); root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(12)
a=rng.normal(size=(3,2,32,32)).astype('float32')
xr.Dataset({c:(('time','z','y','x'),a+i*.1) for i,c in enumerate(['u','v','w'])},coords={'time':[0.,10.,20.]}).to_netcdf(root/'synthetic.nc')
m=dict(status='synthetic_smoke_only',axes=['z','y','x'],dims=['time','z','y','x'],channels=['u','v','w'],shape=[32,32],slice_indices=[0,1],spacing=1.,minimum_time_gap=5.,snapshots=[dict(file='synthetic.nc',time_index=i,time=i*10.,split=s) for i,s in enumerate(['train','val','test'])])
(root/'manifest.json').write_text(json.dumps(m,indent=2))
commands=[['prepare','--manifest',str(root/'manifest.json'),'--root',str(root),'--out',str(root/'data')],['train','--data',str(root/'data'),'--out',str(root/'train'),'--steps','3','--width','8','--patch','8','--batch','2'],['evaluate','--data',str(root/'data'),'--checkpoint',str(root/'train/checkpoint.pt'),'--out',str(root/'eval'),'--steps','4','--samples','2','--batch','4']]
for command in commands: subprocess.run([sys.executable,'run.py',*command],check=True)
