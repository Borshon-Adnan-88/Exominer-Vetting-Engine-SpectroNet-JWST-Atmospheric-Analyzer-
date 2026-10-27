"""No scientific fits: architecture, leakage boundaries, alignment and receipts."""
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import stage5d_run_fusion_width as r


def history():
    return pd.DataFrame(dict(epoch=range(1,12),training_loss=np.linspace(.5,.3,11),
        validation_loss=np.linspace(.4,.6,11),training_roc_auc=np.linspace(.5,.9,11),
        validation_roc_auc=np.linspace(.6,.95,11),epoch_seconds=np.ones(11)))


def outcome():
    return dict(best_epoch=1,best_validation_loss=.4,stopping_epoch=11,epoch_cap_reached=False)


def frames():
    objects=pd.DataFrame(dict(source_row_index=range(6),kepid=range(100,106),
        kepoi_name=['K'+str(i) for i in range(6)],label=[0,1,0,1,0,1],qc_pass=[True]*6))
    assigned=objects.copy()
    assigned['partition']=['train','train','validation','validation','test','test']
    assigned['fold_id']=[0,1,14,15,17,18]
    return objects,assigned


def test_architecture_only_width_changes_and_deterministic_initialization():
    from kepler_cnn_width128.model import build,checked_initialization,branch_hash
    from kepler_cnn_variant.model import build as build32
    from kepler_cnn.model import weight_hash
    old=build32(1729); wide,evidence=checked_initialization(1729)
    assert old.count_params()==138561 and wide.count_params()==538113
    assert evidence['branch_sha256']==branch_hash(old)==branch_hash(wide)
    assert weight_hash(old)!=weight_hash(wide)
    assert weight_hash(build(2718))!=evidence['initial_weight_sha256']
    assert old.optimizer.get_config()==wide.optimizer.get_config()
    assert wide.loss==old.loss=='binary_crossentropy'
    for layer in old.layers:
        a=layer.get_config();b=wide.get_layer(layer.name).get_config()
        if layer.name=='hidden':
            assert a.pop('units')==32 and b.pop('units')==128
        assert a==b
    for name,width in [('global_flatten',2560),('local_flatten',1600),('fusion',4160),('hidden_leaky_relu',128)]:
        assert wide.get_layer(name).output.shape[-1]==width
    # AST normalization ensures the isolated builder has no other scientific edit.
    left=inspect.getsource(build32)
    right=inspect.getsource(build).replace('Dense(128,','Dense(32,').replace('== 538113','== 138561')
    assert ast.dump(ast.parse(left))==ast.dump(ast.parse(right))


def test_checkpoint_is_minimum_bce_not_maximum_auc():
    from kepler_cnn.model import fit
    best=r.selected_checkpoint(history(),outcome())
    assert best.epoch==1 and history().validation_roc_auc.idxmax()==10
    bad=outcome();bad['best_epoch']=11
    with pytest.raises(RuntimeError,match='minimum validation BCE'):
        r.selected_checkpoint(history(),bad)
    # The runner invokes this existing, immutable training function.
    tree=ast.parse(inspect.getsource(fit))
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='EarlyStopping')
    values={k.arg:ast.literal_eval(k.value) for k in call.keywords if isinstance(k.value,ast.Constant)}
    assert values==dict(monitor='val_loss',mode='min',min_delta=0,restore_best_weights=True)
    assert inspect.signature(fit).parameters['patience'].default==10
    assert inspect.signature(fit).parameters['epochs'].default==100


def test_untrained_wide_model_serialization_without_fit(tmp_path):
    import tensorflow as tf
    from kepler_cnn_width128.model import build
    from kepler_cnn.model import predict,weight_hash
    model=build(1729);before=weight_hash(model)
    rng=np.random.default_rng(42)
    inputs={'global_flux':rng.normal(size=(8,2000,1)).astype('float32'),
            'local_flux':rng.normal(size=(8,200,1)).astype('float32')}
    expected=predict(model,inputs)
    model.save(tmp_path/'model.keras')
    loaded=tf.keras.models.load_model(tmp_path/'model.keras',compile=False)
    assert loaded.count_params()==538113
    assert np.array_equal(expected,predict(loaded,inputs))
    assert before==weight_hash(loaded)==weight_hash(model)


def test_feature_boundary_never_materializes_held_out_rows():
    objects,assigned=frames()
    class GuardedCache:
        def __getitem__(self,mask):
            assert np.array_equal(mask,[True,True,True,True,False,False])
            return np.zeros((4,2200),dtype=np.float64)
    scope=r.load_scope(GuardedCache(),objects,assigned)
    assert len(scope.frame)==4
    assert len(scope.select('train')['global_flux'])==2
    assert len(scope.select('validation')['local_flux'])==2
    with pytest.raises(RuntimeError,match='Forbidden'):
        scope.select('test')
    scope.frame.loc[0,'partition']='test'
    with pytest.raises(RuntimeError,match='Forbidden'):
        scope.select('validation')


def test_baseline_alignment_rejects_seed_partition_or_identifier_changes():
    objects,assigned=frames();_,frame=r.scope_frame(objects,assigned)
    who=r.identity('object_stratified',0,1729)
    record={**who,**outcome(),'run_path':r.key_for(who)}
    pred=frame.assign(**who)
    assert r.baseline_alignment(who,frame,record,pred,history())['validation_roc_auc']==.6
    for field,value in [('seed',2718),('repeat_id',1),('strategy','host_group_stratified')]:
        bad={**record,field:value}
        with pytest.raises(RuntimeError):r.baseline_alignment(who,frame,bad,pred,history())
    for column,value in [('kepoi_name','other'),('label',1),('partition','test')]:
        bad=pred.copy();bad.loc[0,column]=value
        with pytest.raises(RuntimeError):r.baseline_alignment(who,frame,record,bad,history())


def test_raw_reuse_rejects_incompatible_protocol_or_identifiers():
    config=r.read(r.CONFIG);parent=r.read(config['parent_config']);protocol=r.read(config['probe_protocol'])
    who=r.identity('object_stratified',0,1729)
    pred=pd.read_csv(r.PARENT/r.key_for(who)/'train_validation_predictions.csv',float_precision='round_trip')
    frame=pred[r.IDENTITY+['qc_pass','partition','fold_id']]
    receipt=r.read(r.PROBE_ROOT/r.key_for(who)/'complete.json')
    with np.load(r.PROBE_ROOT/r.key_for(who)/'validation_predictions.npz',allow_pickle=False) as source:
        saved={k:source[k] for k in source.files}
    assert set(r.raw_compatibility(who,frame,protocol,receipt,saved,config,parent))==set(r.RAW)
    altered=json.loads(json.dumps(protocol));altered['parameters']['C']=2.
    with pytest.raises(RuntimeError):r.raw_compatibility(who,frame,altered,receipt,saved,config,parent)
    saved['kepoi_name']=saved['kepoi_name'][::-1]
    with pytest.raises(RuntimeError):r.raw_compatibility(who,frame,protocol,receipt,saved,config,parent)


def test_resume_skips_completed_and_stops_on_partial_corrupt_or_mismatched(tmp_path):
    who=r.identity('object_stratified',0,1729);binding={'scope_sha256':'ids'}
    def producer(directory):
        np.save(directory/'probability.npy',np.array([.2,.7]))
        return who
    assert r.once(tmp_path,'ctx',who,binding,r.PROBE_FILES,producer)==who
    def forbidden(directory):raise AssertionError('A completed phase must not fit again')
    assert r.once(tmp_path,'ctx',who,binding,r.PROBE_FILES,forbidden)==who
    with pytest.raises(RuntimeError):r.once(tmp_path,'changed',who,binding,r.PROBE_FILES,forbidden)
    with pytest.raises(RuntimeError):r.once(tmp_path,'ctx',who,{'scope_sha256':'wrong'},r.PROBE_FILES,forbidden)
    (tmp_path/'probability.npy').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError):r.once(tmp_path,'ctx',who,binding,r.PROBE_FILES,forbidden)
    partial=tmp_path/'partial';partial.mkdir();(partial/'started.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Partial'):r.once(partial,'ctx',who,binding,r.PROBE_FILES,forbidden)


def test_cli_has_no_implicit_execution_or_held_out_option(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Training entry point reached')
    monkeypatch.setattr(r,'execute',forbidden);monkeypatch.setattr(r,'preflight',forbidden)
    for args in [[],['--test'],['--execute','--preflight']]:
        with pytest.raises(SystemExit):r.main(args)
    tree=ast.parse(Path(r.__file__).read_text())
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls.intersection({'evaluate_predictions','load_features','run'})
    source=inspect.getsource(r.main)
    assert source.index('if args.preflight:')<source.index('execute(data)')


def test_preflight_mode_never_calls_fit_or_probe_fit(monkeypatch,tmp_path):
    import kepler_cnn.model as model
    import kepler_cnn_width128.model as wide
    import scripts.stage5c_linear_probes as probes
    def forbidden(*args,**kwargs):raise AssertionError('Fit invoked in preflight')
    monkeypatch.setattr(model,'fit',forbidden);monkeypatch.setattr(probes,'fit_probe',forbidden)
    monkeypatch.setattr(r,'execute',forbidden);monkeypatch.setattr(model,'configure',lambda:None)
    monkeypatch.setattr(wide,'checked_initialization',lambda seed:(None,{'seed':seed}))
    monkeypatch.setattr(r,'preflight',lambda:dict(config={'seeds':[1729]},context='ctx',references={'one':{}},environment={}))
    monkeypatch.setattr(r,'META',tmp_path)
    r.main(['--preflight'])
    evidence=r.read(tmp_path/'preflight.json')
    assert evidence['scientific_fits_executed']==evidence['linear_probe_fits_executed']==0


def test_training_orchestration_with_no_training_stub(monkeypatch,tmp_path):
    import kepler_cnn.model as shared
    import tensorflow as tf
    objects,assigned=frames()
    matrix=np.zeros((6,2200),dtype=np.float64)
    matrix[-2:]=np.nan  # Held-out feature sentinels must never reach inference.
    scope=r.load_scope(matrix,objects,assigned)
    who=r.identity('object_stratified',0,1729)
    calls=[]
    def no_training(model,training,y,validation,vy,seed):
        calls.append(seed)
        assert len(y)==len(vy)==2
        assert np.array_equal(y,[0,1]) and np.array_equal(vy,[0,1])
        p=shared.predict(model,validation)
        loss=float(tf.keras.losses.BinaryCrossentropy()(vy[:,None],p[:,None]).numpy())
        h=history();h['validation_loss']=loss+np.arange(11)*.01
        h['validation_roc_auc']=r.metrics(vy,p)['roc_auc']
        return h.to_dict('records'),{**outcome(),'best_validation_loss':loss}
    monkeypatch.setattr(shared,'fit',no_training)
    result=r.train_cnn(tmp_path,who,scope,{'validation_roc_auc':.6})
    assert calls==[1729] and result['parameters']==538113 and result['best_epoch']==1
    saved=pd.read_csv(tmp_path/'train_validation_predictions.csv')
    assert set(saved.partition)=={'train','validation'} and len(saved)==4
    assert set(saved.kepoi_name)==set(scope.frame.kepoi_name)


def test_probe_orchestration_fits_only_four_and_resumes_without_inference(monkeypatch,tmp_path):
    import tensorflow as tf
    import scripts.stage5c_linear_probes as original
    from kepler_cnn_width128.model import build
    from kepler_cnn.model import weight_hash
    objects,assigned=frames()
    scope=r.load_scope(np.zeros((6,2200),dtype=np.float64),objects,assigned)
    who=r.identity('object_stratified',0,1729)
    model=build(1729);cnn={'final_weight_sha256':weight_hash(model)}
    (tmp_path/'cnn').mkdir();model.save(tmp_path/'cnn'/'model.keras')
    calls=[]
    def no_probe_fit(features,labels,training,parameters):
        calls.append(features.shape[1])
        assert np.array_equal(training,[True,True,False,False])
        return np.array([.3,.7]),1,None
    monkeypatch.setattr(original,'fit_probe',no_probe_fit)
    raw={name:dict(probability=np.array([.3,.7]),row={**who,'probe':name}) for name in r.RAW}
    rows=r.probe_run(tmp_path,who,scope,cnn,{},raw,{'probe_parameters':{}},'ctx')
    assert calls==[2560,1600,4160,128] and len(rows)==6
    assert sum(row['reused'] for row in rows)==2
    def forbidden(*args,**kwargs):raise AssertionError('Completed probes were recomputed')
    monkeypatch.setattr(original,'fit_probe',forbidden)
    monkeypatch.setattr(tf.keras.models,'load_model',forbidden)
    assert r.probe_run(tmp_path,who,scope,cnn,{},raw,{'probe_parameters':{}},'ctx')==rows
