"""Synthetic-only checks of frozen test reporting and one-time safeguards."""
import numpy as np
import pandas as pd
import pytest
from scripts import stage5d_test_evaluate as e


def test_paired_brier_direction_and_sample_sd():
    s=e.summary([-.1,0,.1],lower=True)
    assert s['sd']==pytest.approx(.1)
    assert (s['improved'],s['worsened'],s['tied'])==(1,1,1)
    assert e.summary([-.1,-.2],lower=True)['improved']==2


def test_pairing_rejects_changed_denominators():
    rows=[]
    for strategy in ['object_stratified','host_group_stratified']:
        for repeat in range(10):
            for track in ['primary','qc_pass']:
                rows.append(dict(strategy=strategy,repeat_id=repeat,partition='test',track=track,
                    sample_count=10,positive_count=4,negative_count=6,assigned_count=10,qc_excluded_count=0,
                    **{m:.6 for m in e.PAIRED}))
    a=pd.DataFrame(rows)
    assert len(e.paired(a,a))==200
    b=a.copy();b.loc[0,'positive_count']=5
    with pytest.raises(RuntimeError,match='denominator'):e.paired(a,b)


def test_test_scope_excludes_train_and_validation():
    objects=pd.DataFrame(dict(source_row_index=[0,1,2],kepid=[1,2,3],kepoi_name=['a','b','c'],label=[0,1,0],qc_pass=[True,False,True]))
    assignment=objects.assign(partition=['train','validation','test'],fold_id=[0,14,17],strategy='object_stratified',repeat_id=0)
    mask,frame=e.test_frame(objects,assignment,dict(strategy='object_stratified',repeat_id=0,seed=1729))
    assert np.array_equal(mask,[False,False,True])
    assert frame.kepoi_name.tolist()==['c'] and frame.partition.tolist()==['test']


def test_gate_refuses_existing_evaluation_before_access(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'OUT',tmp_path)
    (tmp_path/'pre_test_integrity.json').write_text('{}')
    monkeypatch.setattr(e.r,'read',lambda *a:pytest.fail('Read inputs after duplicate gate'))
    with pytest.raises(RuntimeError,match='already exists'):e.gate()


def test_model_plan_rejects_seed_or_missing_identity():
    config=dict(strategies=['object_stratified','host_group_stratified'],seeds=list(range(10)))
    items=[dict(strategy=s,repeat_id=i,seed=i) for s in config['strategies'] for i in range(10)]
    with pytest.raises(RuntimeError,match='exactly 20'):e.validate_plan(items[:-1],config,items)
    bad=[dict(x) for x in items];bad[0]['seed']=999
    with pytest.raises(RuntimeError,match='matrix/seed'):e.validate_plan(bad,config,items)


def test_no_scientific_fit_or_probe_entrypoint():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(e.__file__).read_text(encoding='utf-8'))
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls.intersection({'fit','fit_probe','train_cnn','probe_run','execute'})
