import json
import pytest
from collect_mean_branch import collect


def fixture(root):
    for seed in [42,43,44]:
        for mode in ['implicit','learned']:
            path=root/f'{mode}-{seed}';(path/'train').mkdir(parents=True);(path/'diagnostics').mkdir()
            a=dict(seed=seed,mean_mode=mode,width=32,steps=5000,batch=8,patch=384,lr=.0001)
            summary={'fixed-euler-64':{k:dict(mean=1.) for k in ['energy_ratio','fluctuation_ratio','low_ratio','high_ratio','spectrum_l1']}}
            dist=dict(mean_flow_energy=1.,mean=[0.,0.,0.],std=[1.,1.,1.])
            m=dict(sampler='fixed-euler-64',generated_seeds=list(range(4242,4306)),generated=dist,reference=dist,component_wasserstein_training_std=[0.,0.,0.],reference_plane_means=[[0.,0.,0.]],reference_labels=[dict(snapshot_time=0,slice_index=0)])
            d=dict(model_config=dict(normalization='group',padding='circular',mean_mode=mode),reference=dict(energy=1.),manifest_sha256='same',mean_distribution=m,summary=summary,samples=[dict(seed=4242,arm='fixed-euler-64')])
            (path/'train/training.json').write_text(json.dumps(dict(args=a,parameters=1,seconds=1.)))
            (path/'diagnostics/diagnostics.json').write_text(json.dumps(d))


def test_complete_mean_comparison_and_mismatch_rejection(tmp_path):
    fixture(tmp_path);r=collect(tmp_path)
    assert len(r['runs'])==6 and len(r['paired_learned_minus_implicit'])==3
    assert r['paired_learned_minus_implicit'][0]['wasserstein_training_std_delta']==[0.,0.,0.]
    path=tmp_path/'learned-42/diagnostics/diagnostics.json';d=json.loads(path.read_text());d['mean_distribution']['generated_seeds'][0]=123;path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='differ'):collect(tmp_path)
    path.unlink()
    with pytest.raises(ValueError,match='Incomplete'):collect(tmp_path)
