"""Explicit coordinates and snapshot-level split; never infer time from filenames."""
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def validate(m):
    if m['axes'] != ['z', 'y', 'x']:
        raise ValueError('Map physical axes explicitly to z,y,x in manifest axes/dims')
    if m['channels'] not in [['u', 'v'], ['u', 'v', 'w']]:
        raise ValueError('xy baseline requires u,v or u,v,w')
    if len(m['dims']) != 4 or len(set(m['dims'])) != 4:
        raise ValueError('Four distinct physical dimension names required')
    if len(m['shape']) != 2 or min(m['shape']) < 4 or m['spacing'] <= 0:
        raise ValueError('Invalid plane shape or spacing')
    if not m['slice_indices'] or len(set(m['slice_indices'])) != len(m['slice_indices']) or min(m['slice_indices']) < 0:
        raise ValueError('Need unique nonnegative plane indices')
    if not np.isfinite([m['spacing'], m['minimum_time_gap']]).all():
        raise ValueError('Spacing and gap must be finite')
    seen, times = set(), {}
    for r in m['snapshots']:
        key = (r['file'], r['time_index'])
        if key in seen or r['time'] in times:
            raise ValueError('Duplicate snapshot or physical time (including overlapping files)')
        seen.add(key)
        times[r['time']] = r['split']
        if r['split'] not in ['train', 'val', 'test', 'excluded']:
            raise ValueError('Unknown split')
    groups = {s: [t for t, v in times.items() if v == s] for s in ['train','val','test']}
    if not all(groups.values()):
        raise ValueError('Need nonempty train, val and test snapshot groups')
    gap = m['minimum_time_gap']
    if gap <= 0:
        raise ValueError('Set a positive provisional physical-time guard gap')
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        if min(abs(x-y) for x in groups[a] for y in groups[b]) < gap:
            raise ValueError('Split violates physical-time guard gap')
    return m


def load_manifest(path):
    return validate(json.loads(Path(path).read_text()))


def slices(m, root, split):
    td, zd, yd, xd = m['dims']
    for r in m['snapshots']:
        if r['split'] != split:
            continue
        p = Path(r['file'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Use relative dataset filenames')
        with xr.open_dataset(Path(root)/p, decode_times=False) as ds:
            if m.get('verify_spacing', False):
                for dim in [zd, yd, xd]:
                    if dim not in ds.coords:
                        raise ValueError(f'Missing physical coordinate {dim}')
                    coord = np.asarray(ds[dim].values,dtype='float64')
                    if coord.ndim != 1 or not np.isfinite(coord).all() or not np.allclose(np.diff(coord),m['spacing'],rtol=1e-3,atol=1e-7):
                        raise ValueError(f'Nonuniform or mismatched spacing along {dim}')
            actual = float(ds[td].values[r['time_index']])
            if not np.isclose(actual, r['time'], rtol=0, atol=1e-9):
                raise ValueError('Manifest time differs from file coordinate')
            if any(z >= ds.sizes[zd] for z in m['slice_indices']):
                raise ValueError('Slice index outside volume')
            if any(ds[c].dims != ds[m['channels'][0]].dims for c in m['channels']):
                raise ValueError('Components have different dimension layouts')
            for z in m['slice_indices']:
                a = np.stack([ds[c].isel({td:r['time_index'], zd:z}).transpose(yd,xd).values for c in m['channels']]).astype('float32')
                if a.shape != (len(m['channels']), *m['shape']) or not np.isfinite(a).all():
                    raise ValueError('Invalid slice shape or nonfinite data')
                yield a


def prepare(manifest, root, out):
    m = load_manifest(manifest)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # Two-pass, float64 moments; no validation/test data influence normalization.
    count = 0; mean = np.zeros(len(m['channels'])); m2 = mean.copy()
    for i,a in enumerate(slices(m,root,'train')):
        if i % 8 == 0: print(f'Training statistics: plane {i+1}',flush=True)
        b = a.astype('float64').reshape(len(mean), -1)
        n = b.shape[1]; mu = b.mean(1); delta = mu-mean
        m2 += ((b-mu[:,None])**2).sum(1) + delta**2*count*n/(count+n)
        mean += delta*n/(count+n); count += n
    std = np.sqrt(m2/count)
    if np.any(std <= 0): raise ValueError('Constant training channel')
    norm = dict(mean=mean.tolist(), std=std.tolist(), count=count, manifest_sha256=digest(m), channels=m['channels'])
    (out/'normalization.json').write_text(json.dumps(norm,indent=2))
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    # Disk-backed arrays; no full-volume cache.
    split_statistics = {}
    for split in ['train','val','test']:
        sums = np.zeros(len(mean)); squares = sums.copy(); pixels = 0
        wrap = np.zeros(2); interior = np.zeros(2)
        n = sum(r['split']==split for r in m['snapshots'])*len(m['slice_indices'])
        arr = np.lib.format.open_memmap(out/f'{split}.npy', mode='w+', dtype='float32', shape=(n,len(mean),*m['shape']))
        for i,a in enumerate(slices(m,root,split)):
            normalized = (a-mean[:,None,None])/std[:,None,None]
            arr[i] = normalized
            sums += normalized.sum((1,2)); squares += (normalized**2).sum((1,2)); pixels += a.shape[1]*a.shape[2]
            for j,axis in enumerate([-2,-1]):
                wrap[j] += np.mean((np.take(a,0,axis=axis).astype('float64')-np.take(a,-1,axis=axis))**2)
                interior[j] += np.mean(np.diff(a.astype('float64'),axis=axis)**2)
            if i % 8 == 0: print(f'Export {split}: plane {i+1}/{n}',flush=True)
        arr.flush()
        split_statistics[split] = dict(normalized_mean=(sums/pixels).tolist(),normalized_std=np.sqrt(np.maximum(squares/pixels-(sums/pixels)**2,0)).tolist(),physical_wrap_to_neighbor_squared_jump_yx=(wrap/np.maximum(interior,1e-30)).tolist())
    summary = dict(split_statistics=split_statistics,status=m.get('status'), shape=m['shape'], channels=m['channels'], mean=mean.tolist(), std=std.tolist(), counts={s:sum(r['split']==s for r in m['snapshots'])*len(m['slice_indices']) for s in ['train','val','test']}, manifest_sha256=digest(m))
    (out/'preparation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
