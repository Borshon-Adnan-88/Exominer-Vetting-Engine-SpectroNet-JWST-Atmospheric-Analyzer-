"""Local-only Stage 5B, with a persisted all-run validation gate before test evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_DETERMINISTIC_OPS='1',TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='2')
import argparse
import json
import hashlib
from pathlib import Path
import threading
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from kepler_baselines.inputs import require,sha256,read_input_tables,load_features,IDENTITY
from kepler_baselines.runner import write_csv
from kepler_cnn.artifacts import views,align,check_prediction_identity
from scripts.stage5a_run_diagnostics import atomic,verify_hashes,manifest_entries,sample_indices,probability_stats

META=Path('data/stage5b')
ROOT=Path('data/stage5b_runtime/activation_v1')
CONFIG=Path('configs/kepler_stage5b_activation_v1.json')
FREEZE=Path('configs/kepler_stage5b_freeze_v1.json')


def verified_receipt(directory,phase,context):
    receipt=directory/(phase+'_complete.json')
    if not receipt.exists():
        partial=[directory/n for n in (['started.json','model.keras','history.csv','train_validation_predictions.csv','training_manifest.json'] if phase=='training' else ['predictions.csv','evaluation_manifest.json']) if (directory/n).exists()]
        require(not partial,'Partial run detected; review required, no automatic refit: '+str(directory))
        return None
    value=json.loads(receipt.read_text())
    expected={'model.keras','history.csv','train_validation_predictions.csv','training_manifest.json'} if phase=='training' else {'predictions.csv','evaluation_manifest.json'}
    require(value['context']==context and set(value['sha256'])==expected,'Completion context/schema mismatch')
    verify_hashes({str(directory/n):h for n,h in value['sha256'].items()})
    return json.loads((directory/('training_manifest.json' if phase=='training' else 'evaluation_manifest.json')).read_text())


def seal(directory,phase,context,names):
    atomic(directory/(phase+'_complete.json'),dict(context=context,sha256={n:sha256(directory/n) for n in names}))


def stability(probability,hidden):
    finite=bool(np.isfinite(probability).all() and np.isfinite(hidden).all())
    require(finite,'Nonfinite predictions/activations: unstable, stop for review')
    p=probability_stats(probability)
    zero=bool(np.all(hidden==0))
    return dict(finite=True,validation_unique_count=p['unique_probability_count'],validation_range=p['probability_range'],
        validation_std=p['probability_std'],severe=p['range_below_1e7'],few_values=p['few_values'],hidden_entirely_zero=zero,
        hidden_zero_fraction=float(np.mean(hidden==0)),hidden_std=float(np.std(hidden.astype(float))),
        unstable=bool(p['range_below_1e7'] or p['few_values'] or zero))


def freeze_gate(records,path,context):
    require(len(records)==20 and len({(r['strategy'],r['repeat_id']) for r in records})==20,'Validation gate requires all 20 fits')
    for strategy in ('object_stratified','host_group_stratified'):
        require({r['repeat_id'] for r in records if r['strategy']==strategy}==set(range(10)),'Validation gate run matrix mismatch')
    frame=pd.DataFrame(records)
    result=dict(context=context,models=20,records=records,stability_success=bool(not frame.unstable.any()),
        validation_bce_difference_by_strategy=frame.groupby('strategy').validation_bce_minus_stage5.mean().to_dict(),
        test_evaluation_started=False,criteria='No severe/few-value validation outputs, finite parameters/predictions, no fully zero training probe hidden representation')
    if path.exists(): require(json.loads(path.read_text())==result,'Frozen validation decision changed')
    else: atomic(path,result)
    return sha256(path)


def locally_available(path):
    # Refuse cloud placeholders before any canonical file read.
    if os.name=='nt':
        import ctypes
        attrs=ctypes.windll.kernel32.GetFileAttributesW(str(Path(path).resolve()))
        require(attrs!=-1 and not attrs & (0x1000|0x40000|0x400000),'File unavailable offline; defer without download: '+str(path))
    require(Path(path).is_file(),'Missing local input: '+str(path))


class Progress:
    def __init__(self):
        self.lock=threading.Lock(); self.done=threading.Event()
        self.value=dict(status='running',completed_models=0,total_models=20,current_model=None,current_phase='hash_validation',pid=os.getpid())
        self.update(); self.thread=threading.Thread(target=self.heartbeat,daemon=True); self.thread.start()
    def update(self,**values):
        with self.lock:
            self.value.update(values); self.value['last_update']=datetime.now(timezone.utc).isoformat()
            atomic(META/'status.json',self.value)
            with (META/'stage5b.log').open('a',encoding='utf-8') as f: f.write(json.dumps(self.value)+'\n')
    def heartbeat(self):
        while not self.done.wait(10): self.update()
    def close(self): self.done.set(); self.thread.join()


def prediction_table(objects,aligned,probability,mask,strategy,repeat,seed):
    prediction=objects.loc[mask,IDENTITY+['qc_pass']].copy()
    prediction['model']='dual_view_cnn_leaky_relu_001'; prediction['strategy']=strategy; prediction['repeat_id']=repeat; prediction['seed']=seed
    prediction['partition']=aligned.loc[mask,'partition']; prediction['fold_id']=aligned.loc[mask,'fold_id']
    prediction['probability']=probability.astype(float); prediction['prediction']=(probability>=.5).astype(int)
    prediction['host_seen_in_training']=objects.loc[mask,'kepid'].isin(set(objects.loc[aligned.partition.eq('train'),'kepid']))
    return prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path(r'G:\My Drive\Exominer Research Data\kepler_dr25'))
    args=parser.parse_args(); META.mkdir(parents=True,exist_ok=True); ROOT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(META/'runner.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    progress=Progress(); start=time.perf_counter(); completed=0
    try:
        frozen=json.loads(FREEZE.read_text()); context=sha256(FREEZE)
        for path in frozen['sha256']: locally_available(path)
        verify_hashes(frozen['sha256'])
        config=json.loads(CONFIG.read_text())
        import tensorflow as tf
        from kepler_cnn.model import configure,fit,predict,weight_hash,verify_probability
        from kepler_cnn_variant.model import build
        from kepler_cnn.runner import context as parent_context
        parent_runs=json.loads(Path('data/stage5/run_manifest.json').read_text())['runs']
        require(parent_context()==parent_runs[0]['context'],'Pinned environment mismatch')
        configure()
        objects,index,assignments=read_input_tables(config)
        progress.update(current_phase='canonical_local_hash_validation')
        for relative in objects.relative_path: locally_available(args.dataset/relative)
        inputs=views(load_features(args.dataset,objects,index))
        labels=objects.label.to_numpy()
        probe_records=pd.read_csv('data/stage5a/run_diagnostics.csv')
        records=[]; initial_hashes={}
        for repeat,seed in enumerate(config['seeds']):
            for strategy in config['strategies']:
                key=f'{strategy}/repeat-{repeat:02d}'; directory=ROOT/key
                progress.update(current_model=key,current_phase='training',completed_models=completed)
                # Recheck frozen code and metadata before every fit, without reloading canonical data.
                verify_hashes(frozen['sha256'])
                record=verified_receipt(directory,'training',context)
                aligned=align(objects,assignments,strategy,repeat)
                train=aligned.partition.eq('train').to_numpy(); val=aligned.partition.eq('validation').to_numpy(); non_test=train|val
                if record is None:
                    directory.mkdir(parents=True,exist_ok=True)
                    atomic(directory/'started.json',dict(context=context,run=key))
                    model=build(seed); initial=weight_hash(model)
                    parent=next(r for r in parent_runs if r['run_path']==key)
                    require(initial==parent['initial_weight_sha256'],'Variant initial parameters differ from certified Stage 5')
                    train_inputs={k:a[train] for k,a in inputs.items()}; val_inputs={k:a[val] for k,a in inputs.items()}
                    begin=time.perf_counter()
                    history,outcome=fit(model,train_inputs,labels[train],val_inputs,labels[val],seed)
                    vp=predict(model,val_inputs)
                    sample_frame=objects[['kepoi_name']].assign(partition=aligned.partition)
                    sample=sample_indices(sample_frame)
                    old=probe_records[(probe_records.strategy==strategy)&(probe_records.repeat_id==repeat)].iloc[0]
                    require(objects.iloc[sample].kepoi_name.tolist()==json.loads(old.sample_kois),'Diagnostic training sample changed')
                    before=weight_hash(model)
                    probe=tf.keras.Model(model.inputs,model.get_layer('hidden_leaky_relu').output)
                    hidden=probe({k:a[sample] for k,a in inputs.items()},training=False).numpy()
                    stable=stability(vp,hidden)
                    require(all(np.isfinite(w).all() for w in model.get_weights()),'Nonfinite weights: unstable')
                    loss=tf.keras.losses.BinaryCrossentropy()
                    bce=float(loss(labels[val,None],vp[:,None]).numpy())
                    parent_bce=float(parent['outcome']['best_validation_loss'])
                    probability=predict(model,{k:a[non_test] for k,a in inputs.items()})
                    model.save(directory/'model.keras')
                    reload=tf.keras.models.load_model(directory/'model.keras',compile=False)
                    actual=predict(reload,{k:a[non_test] for k,a in inputs.items()}); verify_probability(probability,actual)
                    require(np.array_equal(probability,actual),'Train/validation reload not bit-identical')
                    require(before==weight_hash(model)==weight_hash(reload),'Weights changed during serialization/probing')
                    write_csv(directory/'train_validation_predictions.csv',prediction_table(objects,aligned,probability,non_test,strategy,repeat,seed))
                    write_csv(directory/'history.csv',pd.DataFrame(history))
                    record=dict(strategy=strategy,repeat_id=repeat,seed=seed,run_path=key,context=context,
                        initial_weight_sha256=initial,final_weight_sha256=before,parameters=model.count_params(),
                        fit_and_artifact_seconds=time.perf_counter()-begin,validation_bce=bce,stage5_validation_bce=parent_bce,
                        validation_bce_minus_stage5=bce-parent_bce,**stable,**outcome)
                    atomic(directory/'training_manifest.json',record)
                    seal(directory,'training',context,['model.keras','history.csv','train_validation_predictions.csv','training_manifest.json'])
                    del model,reload,probe; tf.keras.backend.clear_session()
                require(record['strategy']==strategy and record['repeat_id']==repeat and record['seed']==seed,'Resume identity mismatch')
                require(repeat not in initial_hashes or initial_hashes[repeat]==record['initial_weight_sha256'],'Paired initial weights mismatch')
                initial_hashes[repeat]=record['initial_weight_sha256']; records.append(record); completed+=1
                atomic(META/'run_manifest.json',dict(status='training_test_sealed',completed_models=completed,runs=records))
                progress.update(completed_models=completed,current_phase='trained_test_sealed')
        progress.update(current_phase='freeze_validation_decision')
        gate=freeze_gate(records,META/'validation_stability_summary.json',context)
        atomic(META/'validation_gate.sha256.json',dict(sha256=gate,models=20))
        from kepler_baselines.metrics import evaluate_predictions,paired_and_summary,METRICS
        metrics=[]; bins=[]; hosts=[]
        for record in records:
            key=record['run_path']; directory=ROOT/key
            require(sha256(META/'validation_stability_summary.json')==gate,'Validation gate changed')
            progress.update(current_model=key,current_phase='test_evaluation_after_frozen_gate')
            verified_receipt(directory,'training',context)
            evaluated=verified_receipt(directory,'evaluation',context)
            strategy=record['strategy']; repeat=record['repeat_id']; aligned=align(objects,assignments,strategy,repeat)
            if evaluated is None:
                model=tf.keras.models.load_model(directory/'model.keras',compile=False); before=weight_hash(model)
                probability=predict(model,inputs); verify_probability(probability,predict(model,inputs))
                require(weight_hash(model)==before==record['final_weight_sha256'],'Evaluation changed weights')
                prediction=prediction_table(objects,aligned,probability,np.ones(len(objects),bool),strategy,repeat,record['seed'])
                check_prediction_identity(prediction,objects,aligned)
                write_csv(directory/'predictions.csv',prediction)
                saved=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
                require(np.array_equal(saved.probability.to_numpy(),probability),'CSV probability roundtrip differs')
                atomic(directory/'evaluation_manifest.json',dict(context=context,validation_gate_sha256=gate,run_path=key,probabilities=len(prediction),weights_unchanged=True))
                seal(directory,'evaluation',context,['predictions.csv','evaluation_manifest.json'])
                del model; tf.keras.backend.clear_session()
            else: require(evaluated['validation_gate_sha256']==gate and evaluated['run_path']==key,'Evaluation gate/identity mismatch')
            prediction=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
            check_prediction_identity(prediction,objects,aligned)
            m,b,h=evaluate_predictions(prediction,'dual_view_cnn_leaky_relu_001',strategy,repeat)
            metrics.extend(m); bins.extend(b); hosts.extend(h)
        frame=pd.DataFrame(metrics); paired,summary,differences=paired_and_summary(frame)
        for name,table in [('metrics',frame),('calibration_bins',pd.DataFrame(bins)),('host_diagnostic',pd.DataFrame(hosts)),('paired_differences',paired),('repeat_summary',summary),('paired_summary',differences)]: write_csv(META/(name+'.csv'),table)
        comparisons=[]; keys=['strategy','repeat_id','partition','track']
        for source,filename in [('stage5_cnn','data/stage5/metrics.csv'),('stage4b','data/stage4b/metrics.csv')]:
            old=pd.read_csv(filename)
            for family,base in old.groupby('model'):
                joined=frame.merge(base,on=keys,suffixes=('_variant','_reference'),validate='one_to_one'); require(len(joined)==120,'Comparison matrix incomplete')
                for count in ['sample_count','positive_count','negative_count','assigned_count','qc_excluded_count']: require(joined[count+'_variant'].equals(joined[count+'_reference']),'Comparison denominator mismatch')
                for row in joined.to_dict('records'):
                    for metric in METRICS: comparisons.append({**{k:row[k] for k in keys},'reference':source+':'+family,'metric':metric,'variant_minus_reference':row[metric+'_variant']-row[metric+'_reference']})
        write_csv(META/'reference_comparisons.csv',pd.DataFrame(comparisons))
        verify_hashes(frozen['sha256'])
        atomic(META/'run_manifest.json',dict(status='complete',completed_models=20,runs=records,validation_gate_sha256=gate))
        atomic(META/'provenance.json',dict(context=context,config_sha256=sha256(CONFIG),parent_config_sha256=config['parent_config_sha256'],stage5a_report_sha256=config['stage5a_report_sha256'],source_diff_sha256=config['source_diff_sha256'],scientific_fits=20,runtime_seconds=time.perf_counter()-start,certified_inputs_unchanged=True,independent_external_confirmation=False))
        for root in (ROOT,META):
            files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('checksums.sha256','status.json','stage5b.log','runner.lock','stdout.log','stderr.log')]
            (root/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(root).as_posix()}\n' for p in files),encoding='utf-8')
        progress.update(status='complete',current_model=None,current_phase='finished',runtime_seconds=time.perf_counter()-start)
    except Exception as exc:
        progress.update(status='stopped_for_review',current_phase='stopped_for_review',error=str(exc),nonfinite_failure='non-finite' in str(exc).lower() or 'nonfinite' in str(exc).lower()); raise
    finally: progress.close(); lock.close()

if __name__=='__main__': main()
