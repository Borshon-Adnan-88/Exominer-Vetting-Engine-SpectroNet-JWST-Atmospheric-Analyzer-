import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts import stage5a_run_diagnostics as d


def test_flags():
    row=d.probability_stats(np.array([.4,.4+1e-8,.4]))
    assert row['few_values'] and row['range_below_1e7'] and row['range_below_1e6']
    assert row['diagnostic_class']=='near_constant'
    assert d.probability_stats(np.linspace(0,1,10))['diagnostic_class']=='neither'
    assert d.probability_stats(np.array([0.,1.]))['recoverable_logit_count']==0


def test_sample_training_only_and_order_independent():
    f=pd.DataFrame({'kepoi_name':['c','a','b','d'],'partition':['test','train','train','validation']})
    selected=f.iloc[d.sample_indices(f,1)].kepoi_name.tolist()
    other=f.iloc[::-1].reset_index(drop=True)
    assert other.iloc[d.sample_indices(other,1)].kepoi_name.tolist()==selected
    assert set(f.iloc[d.sample_indices(f)].partition)=={'train'}


def test_receipt_validates_hash_and_context(tmp_path):
    data={name:[] for name in d.TABLES}; context={'test':1}
    assert d.receipt(tmp_path,context) is None
    d.atomic(tmp_path/'diagnostics.json',data)
    d.atomic(tmp_path/'complete.json',{'context':context,'sha256':{'diagnostics.json':d.sha256(tmp_path/'diagnostics.json')}})
    assert d.receipt(tmp_path,context)==data
    with pytest.raises(RuntimeError): d.receipt(tmp_path,{'test':2})
    (tmp_path/'diagnostics.json').write_text('{}')
    with pytest.raises(RuntimeError): d.receipt(tmp_path,context)


def test_status_atomic_and_log(tmp_path):
    d.status(tmp_path,3,'run','probe')
    first=json.loads((tmp_path/'status.json').read_text())
    d.status(tmp_path,4,'next','complete')
    last=json.loads((tmp_path/'status.json').read_text())
    assert first['completed_models']==3 and last['completed_models']==4
    assert last['last_update']>=first['last_update']
    assert len((tmp_path/'stage5a.log').read_text().splitlines())==2
    assert not list(tmp_path.glob('*.tmp'))


def test_readonly_hashes_and_corruption(tmp_path):
    p=tmp_path/'certified'; p.write_bytes(b'original')
    expected={str(p):d.sha256(p)}
    d.verify_hashes(expected); assert p.read_bytes()==b'original'
    p.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError): d.verify_hashes(expected)
    assert p.read_bytes()==b'corrupt'


def test_manifest_escape(tmp_path):
    p=tmp_path/'manifest'; p.write_text('abc  ../outside\n')
    with pytest.raises(RuntimeError): d.manifest_entries(p,tmp_path)


def test_batch_sensitivity_and_probe_weights_unchanged():
    from kepler_cnn.model import build,predict,weight_hash
    model=build(1729); before=weight_hash(model)
    rng=np.random.default_rng(1)
    views={'global_flux':rng.normal(size=(65,2000,1)).astype('float32'),
           'local_flux':rng.normal(size=(65,200,1)).astype('float32')}
    source={k:v.copy() for k,v in views.items()}
    prob=predict(model,views,64)
    for batch in [32,64,256]:
        actual=predict(model,views,batch)
        result=d.sensitivity(np.arange(65)%2,prob,actual)
        assert result['maximum_probability_difference']==float(np.max(np.abs(actual-prob)))
        assert result['changed_labels']==int(np.sum((actual>=.5)!=(prob>=.5)))
        if batch==64: assert np.array_equal(actual,prob)
    probes=d.probes(model,views)
    assert probes['global_conv1'].shape==(65,2000,16)
    assert probes['global_flatten'].shape==(65,2560)
    assert probes['local_conv1'].shape==(65,200,16)
    assert probes['local_flatten'].shape==(65,1600)
    assert probes['hidden'].shape==(65,32) and probes['pre_sigmoid_logit'].shape==(65,1)
    assert before==weight_hash(model)
    assert all(np.array_equal(source[k],v) for k,v in views.items())


def test_sensitivity_detects_labels_and_ranks():
    value=d.sensitivity(np.array([0,1,0,1]),np.array([.1,.9,.3,.7]),np.array([.9,.1,.3,.7]))
    assert value['changed_labels']==2 and value['changed_average_ranks']==2
    assert value['roc_auc_change']<0

@pytest.fixture(scope='module', autouse=True)
def certified_artifacts_unchanged():
    paths=[p for root in ('data/stage4b','data/stage5') for p in Path(root).rglob('*') if p.is_file()]
    before={str(p):d.sha256(p) for p in paths}
    yield
    d.verify_hashes(before)
