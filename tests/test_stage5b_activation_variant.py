import inspect,json
from pathlib import Path
import numpy as np
import pytest
from scripts import stage5b_run_activation_variant as r


def test_exact_single_change_and_policy():
    import tensorflow as tf
    from kepler_cnn import model as old
    from kepler_cnn_variant import model as new
    a=old.build(1729); b=new.build(1729)
    assert a.count_params()==b.count_params()==138561
    assert old.weight_hash(a)==old.weight_hash(b)==old.weight_hash(new.build(1729))
    assert old.weight_hash(b)!=old.weight_hash(new.build(2718))
    assert inspect.getsource(old.fit)==inspect.getsource(new.fit)
    assert a.optimizer.get_config()==b.optimizer.get_config()
    assert b.get_layer('hidden_leaky_relu').negative_slope==.01
    for layer in a.layers:
        left=layer.get_config(); right=b.get_layer(layer.name).get_config()
        if layer.name=='hidden': assert left.pop('activation')=='relu' and right.pop('activation')=='linear'
        assert left==right
    config=json.loads(r.CONFIG.read_text()); parent=json.loads(Path('configs/kepler_stage5_cnn_v1.json').read_text())
    assert config['training']==parent['training'] and config['seeds']==parent['seeds']
    assert config['frozen_sha256']==parent['frozen_sha256'] and config['inputs']==parent['inputs']
    assert config['parent_config_sha256']==r.sha256('configs/kepler_stage5_cnn_v1.json')


def test_collapse():
    s=r.stability(np.array([.4,.4]),np.zeros((3,32)))
    assert s['unstable'] and s['severe'] and s['few_values'] and s['hidden_entirely_zero']
    s=r.stability(np.linspace(.1,.9,20),np.ones((3,32)))
    assert not s['unstable']
    with pytest.raises(RuntimeError): r.stability(np.array([np.nan]),np.zeros((3,32)))


def test_partial_and_resume(tmp_path):
    assert r.verified_receipt(tmp_path,'training','context') is None
    (tmp_path/'started.json').write_text('{}')
    with pytest.raises(RuntimeError): r.verified_receipt(tmp_path,'training','context')
    for name in ('model.keras','history.csv','train_validation_predictions.csv'): (tmp_path/name).write_text('test')
    r.atomic(tmp_path/'training_manifest.json',{'test':True})
    r.seal(tmp_path,'training','context',['model.keras','history.csv','train_validation_predictions.csv','training_manifest.json'])
    assert r.verified_receipt(tmp_path,'training','context')=={'test':True}
    with pytest.raises(RuntimeError): r.verified_receipt(tmp_path,'training','other')
    (tmp_path/'model.keras').write_text('bad')
    with pytest.raises(RuntimeError): r.verified_receipt(tmp_path,'training','context')


def test_validation_gate(tmp_path):
    records=[dict(strategy=s,repeat_id=i,unstable=False,validation_bce_minus_stage5=-.01) for s in ['object_stratified','host_group_stratified'] for i in range(10)]
    with pytest.raises(RuntimeError): r.freeze_gate(records[:-1],tmp_path/'gate.json','ctx')
    h=r.freeze_gate(records,tmp_path/'gate.json','ctx')
    assert h==r.freeze_gate(records,tmp_path/'gate.json','ctx')
    records[0]['unstable']=True
    with pytest.raises(RuntimeError): r.freeze_gate(records,tmp_path/'gate.json','ctx')


def test_save_reload_inference_only(tmp_path):
    import tensorflow as tf
    from kepler_cnn_variant.model import build
    from kepler_cnn.model import predict,weight_hash
    model=build(1729); before=weight_hash(model)
    rng=np.random.default_rng(8)
    x={'global_flux':rng.normal(size=(8,2000,1)).astype('float32'),'local_flux':rng.normal(size=(8,200,1)).astype('float32')}
    expected=predict(model,x); model.save(tmp_path/'model.keras')
    loaded=tf.keras.models.load_model(tmp_path/'model.keras',compile=False)
    assert np.array_equal(expected,predict(loaded,x))
    assert before==weight_hash(model)==weight_hash(loaded)
    assert 'hidden_leaky_relu' in [l.name for l in loaded.layers]


@pytest.fixture(scope='module',autouse=True)
def protected_artifacts_unchanged():
    before={str(p):r.sha256(p) for root in ('data/stage4b','data/stage5','data/stage5a','kepler_cnn') for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    yield
    r.verify_hashes(before)
