import json
from pathlib import Path
import pytest
from collect_factorial import collect,TREATMENTS


def fixture(root):
    for seed in [42,43,44]:
        for i,(norm,padding) in enumerate(TREATMENTS):
            run=root/f'{norm}-{padding}-{seed}';(run/'train').mkdir(parents=True);(run/'diagnostics').mkdir()
            # Known interaction: [1,2,3,8] -> 8-3-2+1 = 4.
            value=[1.,2.,3.,8.][i]
            m=dict(energy=3.,fluctuation_energy=2.,mean=[0.,0.,0.])
            summary={a:{k:dict(mean=value) for k in ['energy_ratio','fluctuation_ratio','low_ratio','high_ratio','spectrum_l1']} for a in ['fixed-euler-64','late-euler-64','all-euler-64']}
            d=dict(reference=m,manifest_sha256='same',summary=summary,samples=[dict(seed=4242,arm=a,metrics=m) for a in summary])
            t=dict(args=dict(seed=seed,model_norm=norm,padding=padding,width=32,steps=5000,batch=8,patch=384,lr=.0001),parameters=1,seconds=1.,peak_cuda_bytes=1)
            (run/'train/training.json').write_text(json.dumps(t));(run/'diagnostics/diagnostics.json').write_text(json.dumps(d))


def test_factorial_requires_complete_matched_design_and_correct_interaction(tmp_path):
    fixture(tmp_path); r=collect(tmp_path)
    assert len(r['runs'])==12 and len(r['within_seed_effects'])==9
    assert r['within_seed_effects'][0]['effects']['low_ratio']['interaction']==4.
    path=next(tmp_path.glob('*/diagnostics/diagnostics.json'));d=json.loads(path.read_text());d['manifest_sha256']='different';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='differ'):collect(tmp_path)
    path.unlink()
    with pytest.raises(ValueError,match='Incomplete'):collect(tmp_path)
