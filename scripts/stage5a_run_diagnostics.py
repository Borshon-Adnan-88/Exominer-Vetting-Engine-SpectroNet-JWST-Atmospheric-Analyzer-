"""Read-only, resumable Stage 5A diagnostics. Run with the pinned Stage 5 Python."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[_key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_DETERMINISTIC_OPS='1', TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2')
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from kepler_baselines.inputs import require, sha256, read_input_tables

OUT = Path('data/stage5a')
ROOT = Path('data/stage5_runtime/cnn_v1')
POLICY = dict(sample_size=128, sample='smallest SHA256(KOI) among training rows', batch_sizes=[32,64,256],
    near_constant_range=1e-7, small_range=1e-6, few_values=3,
    loss_improvement=0.01, divergence_gap_increase=0.05, plateau_last_epochs=5, plateau_loss_range=0.001)
TABLES = ['run_diagnostics','layer_weight_diagnostics','activation_diagnostics','input_diagnostics','numerical_sensitivity','training_histories']

def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(path)

def status(root, completed=0, current=None, phase='validation', state='running', **extra):
    value=dict(status=state,completed_models=completed,total_models=20,current_model=current,current_phase=phase,
               last_update=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    atomic(root/'status.json',value)
    with (root/'stage5a.log').open('a',encoding='utf-8') as f:
        f.write(f"{value['last_update']} {state} {completed}/20 {current or '-'} {phase}\n"); f.flush()

def verify_hashes(entries):
    for p,h in entries.items():
        require(Path(p).is_file() and sha256(p)==h,f'Certified artifact mismatch: {p}')

def manifest_entries(path, root):
    result={}
    for line in Path(path).read_text().splitlines():
        h,n=line.split('  ',1); p=(root/n).resolve()
        require(p.is_relative_to(root.resolve()) and str(p) not in result,'Unsafe/duplicate manifest entry')
        result[str(p)]=h
    return result

def frozen():
    # The approved commit is the trust anchor, not mutable on-disk provenance.
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','4a2b193','--','configs','data/stage4b','data/stage5','data/stage5_infrastructure','kepler_baselines','kepler_cnn','scripts/stage5_run_cnn.py'],text=True).splitlines()
    hashes={p:hashlib.sha256(subprocess.check_output(['git','show','4a2b193:'+p])).hexdigest() for p in names}
    verify_hashes(hashes)
    config=json.loads(Path('configs/kepler_stage5_cnn_v1.json').read_text())
    hashes.update(config['frozen_sha256'])
    hashes.update(manifest_entries(Path('data/stage5/runtime_checksums.sha256'),ROOT))
    for name in ('prediction_checksums.sha256','model_checksums.sha256'):
        hashes.update(manifest_entries(Path('data/stage4b')/name,Path(config['baseline_runtime'])))
    verify_hashes(hashes)
    return config,hashes

def sample_indices(frame, count=128):
    positions=[i for i in range(len(frame)) if frame.iloc[i].partition=='train']
    return np.array(sorted(positions,key=lambda i:hashlib.sha256(str(frame.iloc[i].kepoi_name).encode()).hexdigest())[:count],dtype=int)

def stats(a):
    a=np.asarray(a,dtype=np.float64); finite=a[np.isfinite(a)]
    return dict(min=float(finite.min()) if finite.size else None,max=float(finite.max()) if finite.size else None,
        mean=float(finite.mean()) if finite.size else None,std=float(finite.std()) if finite.size else None,
        zero_fraction=float(np.mean(a==0)),nonfinite_count=int((~np.isfinite(a)).sum()),norm=float(np.linalg.norm(finite)))

def probability_stats(p):
    p=np.asarray(p,dtype=float); require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probability')
    spread=float(np.ptp(p)); unique=len(np.unique(p)); inside=(p>0)&(p<1)
    logits=np.log(p[inside])-np.log1p(-p[inside])
    _,counts=np.unique(p,return_counts=True); f=counts/len(p)
    return dict(unique_probability_count=unique,probability_min=float(p.min()),probability_max=float(p.max()),
        probability_range=spread,probability_mean=float(p.mean()),probability_std=float(p.std()),
        logit_mean=float(logits.mean()) if len(logits) else None,logit_std=float(logits.std()) if len(logits) else None,
        recoverable_logit_count=int(inside.sum()),fraction_above_half=float(np.mean(p>.5)),
        fraction_below_half=float(np.mean(p<.5)),fraction_equal_half=float(np.mean(p==.5)),
        discrete_output_entropy_bits=float(-np.sum(f*np.log2(f))),
        range_below_1e7=spread<1e-7,range_below_1e6=spread<1e-6,few_values=unique<=3,
        diagnostic_class='near_constant' if spread<1e-7 else 'few_values' if unique<=3 else 'neither')

def sensitivity(y, reference, actual):
    return dict(maximum_probability_difference=float(np.max(np.abs(actual-reference))),
        changed_labels=int(np.sum((actual>=.5)!=(reference>=.5))),
        roc_auc_change=float(roc_auc_score(y,actual)-roc_auc_score(y,reference)),
        changed_average_ranks=int(np.sum(rankdata(actual)!=rankdata(reference))))

def receipt(directory, context):
    p=directory/'complete.json'
    if not p.exists(): return None
    value=json.loads(p.read_text()); require(value['context']==context,'Diagnostic resume context mismatch')
    require(set(value['sha256'])=={'diagnostics.json'},'Incomplete diagnostic receipt')
    verify_hashes({str(directory/n):h for n,h in value['sha256'].items()})
    result=json.loads((directory/'diagnostics.json').read_text())
    require(set(result)==set(TABLES),'Diagnostic schema mismatch')
    return result

def probes(model, views):
    import tensorflow as tf
    names=['global_conv1','global_flatten','local_conv1','local_flatten','hidden']
    probe=tf.keras.Model(model.inputs,[model.get_layer(n).output for n in names])
    values=probe(views,training=False)
    result={n:v.numpy() for n,v in zip(names,values)}
    output=model.get_layer('confirmed_probability')
    result['pre_sigmoid_logit']=(tf.matmul(values[-1],output.kernel)+output.bias).numpy()
    return result

def inspect_model(model, inputs, prediction, history, record, fill):
    from kepler_cnn.model import predict,weight_hash
    before=weight_hash(model); positions=sample_indices(prediction,POLICY['sample_size'])
    sampled={k:v[positions] for k,v in inputs.items()}
    base=dict(strategy=record['identity']['strategy'],repeat_id=record['identity']['repeat_id'])
    row={**base,**record['outcome'],**probability_stats(prediction.probability.to_numpy())}
    train=history.training_loss.to_numpy(); val=history.validation_loss.to_numpy()
    row.update(loss_improved_by_one_percent=bool(val.min()<=val[0]*.99),
        plateau_last_five=bool(len(val)>=5 and np.ptp(val[-5:])<.001),
        train_validation_divergence=bool((val[-1]-train[-1])-(val[0]-train[0])>.05 and train[-1]<train[0]),
        final_training_auc=float(history.training_roc_auc.iloc[-1]),final_validation_auc=float(history.validation_roc_auc.iloc[-1]),
        learning_interpretation='Descriptive flags only; saved endpoints cannot establish when or why collapse arose',
        sample_kois=json.dumps(prediction.iloc[positions].kepoi_name.tolist()))
    tables={name:[] for name in TABLES}; tables['run_diagnostics']=[row]
    tables['training_histories']=[{**base,**r} for r in history.to_dict('records')]
    for branch,a in sampled.items():
        variance=np.var(a.astype(float),axis=(1,2))
        tables['input_diagnostics'].append({**base,'branch':branch,'sample_count':len(a),**stats(a),
            'unobserved_bin_fraction':float(fill[branch][positions].mean()),
            **{f'object_variance_q{q}':float(np.percentile(variance,q)) for q in (0,25,50,75,100)}})
    for layer in model.layers:
        for i,w in enumerate(layer.weights):
            tables['layer_weight_diagnostics'].append({**base,'layer':layer.name,'tensor':w.name,'tensor_index':i,
                'role':'bias' if 'bias' in w.name else 'weight','shape':str(tuple(w.shape)),**stats(w.numpy())})
        if not layer.weights:
            tables['layer_weight_diagnostics'].append({**base,'layer':layer.name,'role':'no_weights','tensor':'','tensor_index':-1})
    for name,a in probes(model,sampled).items():
        tables['activation_diagnostics'].append({**base,'layer':name,'shape':str(a.shape),**stats(a),
            'between_object_variance':float(np.var(a.astype(float),axis=0).mean())})
    reference=prediction.probability.to_numpy(); y=prediction.label.to_numpy()
    for batch in POLICY['batch_sizes']:
        actual=predict(model,inputs,batch)
        if batch==64: require(np.array_equal(actual,reference),'Authoritative batch-64 replay differs; stop for review')
        for partition in ('all','train','validation','test'):
            m=np.ones(len(y),bool) if partition=='all' else prediction.partition.eq(partition).to_numpy()
            tables['numerical_sensitivity'].append({**base,'batch_size':batch,'partition':partition,
                **sensitivity(y[m],reference[m],actual[m])})
    require(before==weight_hash(model),'Diagnostics changed weights')
    row['weights_unchanged']=True
    return tables

def canonical(dataset,objects,index):
    from kepler_scale.arrays import KEYS,logical_hash
    inputs={b+'_flux':np.empty((len(objects),n,1),np.float32) for b,n in [('global',2000),('local',200)]}
    fill={k:np.empty(v.shape[:2],bool) for k,v in inputs.items()}
    for i,obj in enumerate(objects.itertuples()):
        path=(dataset/obj.relative_path).resolve(); require(path.is_relative_to(dataset.resolve()),'Unsafe NPZ path')
        payload=path.read_bytes(); require(hashlib.sha256(payload).hexdigest()==obj.npz_sha256,'Canonical NPZ mismatch: '+obj.kepoi_name)
        with np.load(io.BytesIO(payload),allow_pickle=False) as source:
            require(set(source.files)==set(KEYS),'Unexpected canonical arrays')
            for key in KEYS:
                a=source[key]; require(logical_hash(a)==index.loc[obj.kepoi_name,key+'_sha256'],'Logical hash mismatch')
                require(np.isfinite(a).all(),'Nonfinite canonical input')
            for key in inputs:
                inputs[key][i,:,0]=source[key]
                fill[key][i]=~source[key.replace('_flux','_observed_mask')]
        if (i+1)%500==0: status(OUT,phase=f'canonical_hash_validation_{i+1}_of_{len(objects)}')
    return inputs,fill

def finish(results,context):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tables={name:pd.DataFrame([r for result in results for r in result[name]]) for name in TABLES}
    for name,table in tables.items(): table.to_csv(OUT/(name+'.csv'),index=False)
    runs=tables['run_diagnostics']; acts=tables['activation_diagnostics']
    summary=dict(status='complete',models=len(runs),policy=POLICY,context=context,
        near_constant=int(runs.range_below_1e7.sum()),few_values=int(runs.few_values.sum()),
        range_below_1e6=int(runs.range_below_1e6.sum()),
        class_counts=runs.diagnostic_class.value_counts().to_dict(),
        interpretation='Exploratory descriptive diagnostics; no causal mechanism established; no runs excluded or corrected.')
    # Associations are predeclared and descriptive, including overlapping flags.
    joined=acts.merge(runs[['strategy','repeat_id','diagnostic_class']],on=['strategy','repeat_id'],validate='many_to_one')
    summary['activation_group_means']=joined.groupby(['diagnostic_class','layer'])[['zero_fraction','between_object_variance']].mean().reset_index().to_dict('records')
    weights=tables['layer_weight_diagnostics'].merge(runs[['strategy','repeat_id','diagnostic_class']],on=['strategy','repeat_id'])
    summary['weight_group_means']=weights.dropna(subset=['norm']).groupby(['diagnostic_class','layer','role'])[['norm','std']].mean().reset_index().to_dict('records')
    atomic(OUT/'diagnostic_summary.json',summary)
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    for name,column in [('collapse_overview','unique_probability_count'),('probability_spread','probability_range')]:
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(np.arange(len(runs)),runs[column]); ax.set(xlabel='Run order (all 20)',ylabel=column)
        fig.tight_layout(); fig.savefig(figures/(name+'.png')); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for layer,rows in joined.groupby('layer'): ax.plot(np.arange(len(rows)),rows.between_object_variance,label=layer)
    ax.set_yscale('symlog',linthresh=1e-12); ax.legend(fontsize=7); ax.set(xlabel='Run order',ylabel='Between-object activation variance')
    fig.tight_layout(); fig.savefig(figures/'activation_variance.png'); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    histories=tables['training_histories']
    for ax,key in zip(axes.flat,['training_loss','validation_loss','training_roc_auc','validation_roc_auc']):
        for _,h in histories.groupby(['strategy','repeat_id']): ax.plot(h.epoch,h[key],alpha=.5)
        ax.set(xlabel='Epoch',ylabel=key)
    fig.tight_layout(); fig.savefig(figures/'training_history_comparison.png'); plt.close(fig)
    report=Path('docs/kepler_stage5a_failure_analysis.md')
    text='# Stage 5A failure analysis\n\nDiagnostics only; all 20 certified runs retained. No fitting or weight updates.\n\n'
    text+=f"Observation: {summary['few_values']} runs have at most three probability values; {summary['near_constant']} have range below 1e-7.\n\n"
    text+='Evidence: [run diagnostics](../data/stage5a/run_diagnostics.csv), [layer weights](../data/stage5a/layer_weight_diagnostics.csv), [activations](../data/stage5a/activation_diagnostics.csv), [training-only inputs](../data/stage5a/input_diagnostics.csv), [batch sensitivity](../data/stage5a/numerical_sensitivity.csv), and [histories](../data/stage5a/training_histories.csv).\n\n'
    text+='Hypotheses under examination: low-variance or zero ReLU representations, output saturation, weight-scale differences, and numerical ranking sensitivity. Group summaries are descriptive associations, not causal evidence. Discrete output entropy describes diversity, not predictive uncertainty. Logits recovered from probabilities exclude exact 0/1; internal logits are measured separately. Zero flux and unobserved bins are reported separately.\n\n'
    text+='Unresolved mechanism: endpoint probes cannot reconstruct training trajectories or establish why a representation developed. Saved optimizer slots reflect stopping epoch while restored weights reflect the selected epoch. No corrective experiment is authorized. Original Stage 5 predictions and metrics remain authoritative; alternative batches are diagnostics only.\n\n'
    text+='Predeclared policy and group summaries:\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    report.write_text(text,encoding='utf-8')
    atomic(OUT/'report_integrity.json',{'path':str(report),'sha256':sha256(report)})
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('checksums.sha256','status.json','stage5a.log','runner.lock') and not p.name.endswith('.tmp')]
    (OUT/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')
    verify_hashes(manifest_entries(OUT/'checksums.sha256',OUT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path(r'G:\My Drive\Exominer Research Data\kepler_dr25'))
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    # OS releases the lock after a crash; simultaneous invocations cannot write together.
    import msvcrt
    lock=(OUT/'runner.lock').open('a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    completed=0; current=None; started=time.perf_counter()
    try:
        status(OUT)
        config,anchors=frozen()
        context=dict(source_sha256=sha256(__file__),policy=POLICY,certified_commit='4a2b193',
                     anchors_sha256=hashlib.sha256(json.dumps(anchors,sort_keys=True).encode()).hexdigest())
        atomic(OUT/'provenance.json',dict(context=context,anchors=anchors,scientific_fits=0))
        objects,index,assignments=read_input_tables(config)
        records=json.loads(Path('data/stage5/run_manifest.json').read_text())['runs']
        require(len(records)==20 and len({r['run_path'] for r in records})==20,'Incomplete run matrix')
        prior=[receipt(OUT/'runs'/r['run_path'],context) for r in records]
        inputs=fill=None
        if any(r is None for r in prior):
            inputs,fill=canonical(args.dataset,objects,index)
            import tensorflow as tf
            from kepler_cnn.model import configure
            from kepler_cnn.runner import context as scientific_context
            require(scientific_context()==records[0]['context'],'Pinned scientific environment changed')
            configure()
        from kepler_cnn.artifacts import check_prediction_identity,align
        results=[]
        for record,cached in zip(records,prior):
            current=record['run_path']; directory=OUT/'runs'/current
            if cached is None:
                status(OUT,completed,current,'activation_and_numerical_diagnostics')
                pred=pd.read_csv(ROOT/current/'predictions.csv',float_precision='round_trip')
                check_prediction_identity(pred,objects,align(objects,assignments,record['identity']['strategy'],record['identity']['repeat_id']))
                history=pd.read_csv(ROOT/current/'history.csv',float_precision='round_trip')
                model=tf.keras.models.load_model(ROOT/current/'model.keras',compile=False)
                cached=inspect_model(model,inputs,pred,history,record,fill)
                verify_hashes({str(ROOT/current/n):h for n,h in record['sha256'].items()})
                atomic(directory/'diagnostics.json',cached)
                atomic(directory/'complete.json',dict(context=context,sha256={'diagnostics.json':sha256(directory/'diagnostics.json')}))
                require(receipt(directory,context) is not None,'Incomplete diagnostics')
                del model; tf.keras.backend.clear_session()
            results.append(cached); completed+=1; status(OUT,completed,current,'model_complete')
        verify_hashes(anchors)
        require(sha256(__file__)==context['source_sha256'],'Diagnostic code changed during execution')
        finish(results,context)
        status(OUT,completed,None,'finished','complete',runtime_seconds=time.perf_counter()-started)
    except Exception as exc:
        status(OUT,completed,current,'stopped_for_review','stopped_for_review',error=str(exc)); raise
    finally:
        lock.close()

if __name__=='__main__': main()
