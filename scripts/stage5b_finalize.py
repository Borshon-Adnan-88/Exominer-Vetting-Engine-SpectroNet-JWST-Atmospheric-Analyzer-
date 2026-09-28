"""Finalize frozen Stage 5B artifacts; never fit a model or read raw archives.

Run --analyze once in the pinned environment, then --certify after offline tests.
The separately sealed replay is reused on subsequent reporting invocations.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from scripts import stage5b_run_activation_variant as runner
from scripts.stage5a_run_diagnostics import manifest_entries, verify_hashes, sample_indices, stats, probability_stats
from kepler_baselines.inputs import require, sha256, read_input_tables
from kepler_baselines.metrics import evaluate_predictions, paired_and_summary, METRICS
from kepler_baselines.runner import write_csv
from kepler_cnn.artifacts import align, check_prediction_identity, views
import numpy as np
import pandas as pd

META, ROOT, CONFIG, FREEZE = runner.META, runner.ROOT, runner.CONFIG, runner.FREEZE
REPORT = Path('docs/kepler_stage5b_activation_ablation.md')
PARENT = '4a2b193'
KEYS = ['strategy', 'repeat_id']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    from scripts.stage5b_resume_existing import resilient_atomic
    resilient_atomic(path, value)


def csv(name, frame):
    write_csv(META / (name + '.csv'), frame)


def equal_table(actual, saved):
    require(set(actual.columns) == set(saved.columns) and len(actual) == len(saved), 'Table schema/count differs')
    for name in actual:
        a, b = actual[name].reset_index(drop=True), saved[name].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a):
            require(np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True), 'Table numeric mismatch: ' + name)
        else:
            require(a.fillna('').equals(b.fillna('')), 'Table identity mismatch: ' + name)


def load_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def check_policy():
    config, parent = read(CONFIG), read('configs/kepler_stage5_cnn_v1.json')
    expected = json.loads(json.dumps(parent))
    expected['architecture']['fusion_activation'] = dict(dense_activation='linear', negative_slope=.01)
    expected['experiment_id'] = config['experiment_id']
    require(all(config[k] == v for k, v in expected.items()), 'Unexpected scientific configuration difference')
    require(set(config) - set(parent) == {'parent_config_sha256', 'stage5a_report_sha256', 'source_diff_sha256', 'stability'}, 'Unexpected configuration additions')
    source = Path('kepler_cnn/model.py').read_text()
    variant = Path('kepler_cnn_variant/model.py').read_text()
    normalized = variant.replace('Dense(32, activation="linear"', 'Dense(32, activation="relu"').replace(
        '    x = tf.keras.layers.LeakyReLU(negative_slope=0.01, name="hidden_leaky_relu")(x)\n', '')
    require(ast.dump(ast.parse(source)) == ast.dump(ast.parse(normalized)), 'Unexpected scientific source difference')
    verify_hashes(read(FREEZE)['sha256'])
    require(sha256('configs/kepler_stage5_cnn_v1.json') == config['parent_config_sha256'], 'Parent configuration changed')
    require(sha256('docs/kepler_stage5a_failure_analysis.md') == config['stage5a_report_sha256'], 'Stage 5A report changed')
    return config


def audit():
    config = check_policy()
    # Verify historical receipt files before replacing any compact derived metadata.
    verify_hashes(manifest_entries(META / 'checksums.sha256', META))
    verify_hashes(manifest_entries(ROOT / 'checksums.sha256', ROOT))
    verify_hashes(read('data/stage5a/provenance.json')['anchors'])
    verify_hashes(manifest_entries(Path('data/stage5a/checksums.sha256'), Path('data/stage5a')))
    # Respect historical Git checkout line endings, but never normalize frozen hashes.
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PARENT, '--',
        'configs', 'data', 'docs', 'kepler_catalog', 'kepler_splits', 'kepler_lightcurves',
        'kepler_scale', 'kepler_baselines', 'kepler_cnn'], text=True).splitlines()
    import hashlib
    anchors = {}; checkout_line_endings = []
    for p in tracked:
        committed = subprocess.check_output(['git', 'show', PARENT + ':' + p])
        local = Path(p).read_bytes()
        if committed != local:
            require(committed.replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n'), 'Parent scientific content changed: '+p)
            checkout_line_endings.append(p)
        anchors[p] = hashlib.sha256(local).hexdigest()
    protected = {**anchors, **read(FREEZE)['sha256']}
    save(META/'protected_input_hashes.json', protected)
    records = read(META / 'run_manifest.json')['runs']
    expected = {(s, i) for s in config['strategies'] for i in range(10)}
    require(len(records) == 20 and {(r['strategy'], r['repeat_id']) for r in records} == expected, 'Missing/duplicate run identity')
    require(len(list(ROOT.rglob('model.keras'))) == 20 and len(list(ROOT.rglob('predictions.csv'))) == 20, 'Duplicate scientific outputs')
    context = sha256(FREEZE)
    gate = read(META / 'validation_stability_summary.json')
    gate_hash = sha256(META / 'validation_stability_summary.json')
    require(gate['records'] == records and gate['models'] == 20 and gate['context'] == context, 'Validation gate mismatch')
    require(gate['test_evaluation_started'] is False, 'Gate was not sealed before test evaluation')
    require(read(META / 'validation_gate.sha256.json')['sha256'] == gate_hash, 'Gate receipt mismatch')
    events = [json.loads(line) for line in (META / 'stage5b.log').read_text().splitlines()]
    gate_events = [e for e in events if e['current_phase'] == 'freeze_validation_decision']
    test_events = [e for e in events if e['current_phase'] == 'test_evaluation_after_frozen_gate']
    require(gate_events and test_events and gate_events[0]['last_update'] < test_events[0]['last_update'], 'Gate/test log ordering mismatch')
    objects, _, assignments = read_input_tables(config)
    parent = {r['run_path']: r for r in read('data/stage5/run_manifest.json')['runs']}
    rows, bins, hosts, outcomes, histories, diversity, batch_context = [], [], [], [], [], [], []
    for r in records:
        d = ROOT / r['run_path']; s, i = r['strategy'], r['repeat_id']
        require(runner.verified_receipt(d, 'training', context) == r, 'Aggregate run disagreement')
        evaluation = runner.verified_receipt(d, 'evaluation', context)
        require(evaluation['validation_gate_sha256'] == gate_hash and evaluation['run_path'] == r['run_path'], 'Evaluation receipt mismatch')
        require(r['seed'] == config['seeds'][i] and r['parameters'] == 138561, 'Run seed/parameter mismatch')
        require(r['initial_weight_sha256'] == parent[r['run_path']]['initial_weight_sha256'], 'Initial parameters changed')
        pred = load_csv(d / 'predictions.csv')
        check_prediction_identity(pred, objects, align(objects, assignments, s, i))
        for k, v in dict(strategy=s, repeat_id=i, seed=r['seed'], model='dual_view_cnn_leaky_relu_001').items():
            require(pred[k].eq(v).all(), 'Prediction run identity mismatch')
        sealed = load_csv(d / 'train_validation_predictions.csv')
        non_test = pred.loc[pred.partition.ne('test')].reset_index(drop=True)
        equal_table(non_test.drop(columns='probability'), sealed.drop(columns='probability'))
        # These are different inference cohorts/batch compositions, not a replay.
        # Keep pre-gate values authoritative; quantify drift without a new threshold.
        batch_context.append(dict(strategy=s,repeat_id=i,
            max_probability_difference=float(np.max(np.abs(non_test.probability-sealed.probability))),
            changed_labels=int((non_test.prediction!=sealed.prediction).sum())))
        a, b, h = evaluate_predictions(pred, 'dual_view_cnn_leaky_relu_001', s, i)
        rows.extend(a); bins.extend(b); hosts.extend(h)
        history = load_csv(d / 'history.csv'); best = history.loc[history.validation_loss.idxmin()]
        require(np.isfinite(history.to_numpy()).all(), 'Nonfinite history')
        require(history.epoch.tolist() == list(range(1, len(history)+1)), 'Epoch sequence mismatch')
        require(best.epoch == r['best_epoch'] and abs(best.validation_loss-r['best_validation_loss']) <= 1e-12, 'Best epoch mismatch')
        require(len(history) == r['stopping_epoch'] == min(r['best_epoch'] + 10, 100), 'Stopping rule mismatch')
        require(r['epoch_cap_reached'] == (len(history) == 100), 'Epoch cap mismatch')
        outcomes.append({**r, 'validation_roc_auc': float(best.validation_roc_auc), 'hidden_variance': r['hidden_std']**2,
                         'epoch_seconds_sum': float(history.epoch_seconds.sum())})
        histories.extend({**dict(strategy=s, repeat_id=i), **v} for v in history.to_dict('records'))
        diversity.append(dict(strategy=s, repeat_id=i, **probability_stats(pred.probability)))
    frame = pd.DataFrame(rows)
    for name, value in [('metrics', frame), ('calibration_bins', pd.DataFrame(bins)), ('host_diagnostic', pd.DataFrame(hosts))]:
        equal_table(value, load_csv(META / (name+'.csv')))
    for name, value in zip(['paired_differences', 'repeat_summary', 'paired_summary'], paired_and_summary(frame)):
        equal_table(value, load_csv(META / (name+'.csv')))
    for path in (META / 'recovery').glob('*/recovery.json'):
        verify_hashes(read(path)['protected'])
    csv('run_outcomes', pd.DataFrame(outcomes)); csv('training_histories', pd.DataFrame(histories)); csv('output_diversity', pd.DataFrame(diversity))
    csv('inference_cohort_comparison',pd.DataFrame(batch_context))
    (META / 'runtime_checksums.sha256').write_bytes((ROOT / 'checksums.sha256').read_bytes())
    result = dict(status='passed', models=20, prediction_rows=len(objects)*20, metric_rows=len(rows),
        calibration_rows=len(bins), no_duplicate_outputs=True, protected_initial_fits=9,
        parent_commit=subprocess.check_output(['git', 'rev-parse', PARENT], text=True).strip(),
        parent_scientific_files_verified=len(anchors), frozen_files_verified=len(read(FREEZE)['sha256']),
        historical_git_checkout_line_ending_only_differences=checkout_line_endings,
        gate_sha256=gate_hash, gate_time=gate_events[0]['last_update'], first_test_time=test_events[0]['last_update'],
        fit_and_artifact_seconds=sum(r['fit_and_artifact_seconds'] for r in records),
        completed_epoch_seconds=sum(v['epoch_seconds_sum'] for v in outcomes),
        resumed_process_seconds=read(META / 'status.json')['runtime_seconds'],
        total_including_interrupted_work_seconds=None,
        timing_limitation='Completed-fit duration includes inference/serialization; interrupted unfinished fit time is not durably recorded. Process resume duration is not total experiment duration.',
        stage4a_arrays='Existing certified canonical/cache equivalence and pinned cache hash reused; no archive or 6637-file audit repeated.',
        scientific_fits_executed=0)
    save(META / 'run_integrity.json', result)
    return config, records, objects, assignments


def replay(config, records, objects, assignments):
    receipt = META / 'replay_complete.json'
    if receipt.exists():
        saved = read(receipt)
        require(saved['context'] == sha256(FREEZE), 'Replay context changed')
        verify_hashes(saved['sha256'])
        require(read(META/'prediction_verification.json')['status'] == 'passed', 'Replay did not pass')
        return
    import tensorflow as tf
    from kepler_cnn.model import configure, predict, verify_probability, weight_hash
    from kepler_cnn.runner import context
    require(context() == read('data/stage5/run_manifest.json')['runs'][0]['context'], 'Replay environment mismatch')
    configure()
    cache = Path(config['baseline_runtime']) / 'features.npy'
    require(sha256(cache) == config['feature_cache_sha256'], 'Feature cache changed')
    require(read('data/stage5/input_validation.json')['canonical_equals_existing_cache'], 'Cache equivalence uncertified')
    require(objects.equals(load_csv('data/stage4b/objects.csv')), 'Cache KOI order changed')
    inputs = views(np.load(cache, mmap_mode='r', allow_pickle=False))
    old_probes = load_csv('data/stage5a/run_diagnostics.csv').set_index(KEYS)
    activations, verifications = [], []; started = time.perf_counter()
    for r in records:
        d = ROOT/r['run_path']; s, i = r['strategy'], r['repeat_id']
        pred = load_csv(d/'predictions.csv')
        check_prediction_identity(pred, objects, align(objects, assignments, s, i))
        model = tf.keras.models.load_model(d/'model.keras', compile=False)
        before = weight_hash(model)
        require(before == r['final_weight_sha256'] and model.count_params() == 138561, 'Loaded model mismatch')
        actual = predict(model, inputs)
        verify_probability(pred.probability.to_numpy(), actual)
        replayed = pred.copy(); replayed['probability'] = actual.astype(float); replayed['prediction'] = (actual >= .5).astype(int)
        a, b, _ = evaluate_predictions(replayed, 'dual_view_cnn_leaky_relu_001', s, i)
        saved_metrics = load_csv(META/'metrics.csv'); saved_bins = load_csv(META/'calibration_bins.csv')
        equal_table(pd.DataFrame(a), saved_metrics[(saved_metrics.strategy==s)&(saved_metrics.repeat_id==i)])
        equal_table(pd.DataFrame(b), saved_bins[(saved_bins.strategy==s)&(saved_bins.repeat_id==i)])
        positions = sample_indices(pred)
        require(pred.iloc[positions].kepoi_name.tolist() == json.loads(old_probes.loc[(s,i),'sample_kois']), 'Training probe changed')
        names = ['global_flatten', 'local_flatten', 'hidden_leaky_relu']
        probe = tf.keras.Model(model.inputs, [model.get_layer(n).output for n in names])
        values = probe({k:v[positions] for k,v in inputs.items()}, training=False)
        output = model.get_layer('confirmed_probability')
        logits = (tf.matmul(values[-1],output.kernel)+output.bias).numpy()
        for name, value in zip(['global_flatten','local_flatten','hidden','pre_sigmoid_logit'], [v.numpy() for v in values]+[logits]):
            activations.append(dict(strategy=s,repeat_id=i,layer=name,shape=str(value.shape),**stats(value),
                between_object_variance=float(np.var(value.astype(float),axis=0).mean())))
        require(weight_hash(model) == before, 'Inference changed model weights')
        verifications.append(dict(strategy=s,repeat_id=i,rows=len(pred),maximum_probability_difference=float(np.max(np.abs(actual-pred.probability))),
            labels_equal=bool(np.array_equal(actual>=.5,pred.prediction)),metrics_recomputed=True,weights_unchanged=True,sample_kois=pred.iloc[positions].kepoi_name.tolist()))
        del model, probe; tf.keras.backend.clear_session()
        print('Replay/probe verified: '+r['run_path'], flush=True)
    csv('activation_diagnostics',pd.DataFrame(activations))
    save(META/'prediction_verification.json',dict(status='passed',models=20,rows=20*len(objects),fresh_process_pid=os.getpid(),
        python=sys.executable,environment=context(),atol=1e-7,rtol=0,threshold=.5,
        max_probability_difference=max(v['maximum_probability_difference'] for v in verifications),
        records=verifications,runtime_seconds=time.perf_counter()-started,source_sha256=sha256(__file__),scientific_fits_executed=0))
    save(receipt,dict(context=sha256(FREEZE),sha256={str(META/n):sha256(META/n) for n in ['prediction_verification.json','activation_diagnostics.csv']}))


def paired_reference(frame, reference, family):
    keys = KEYS+['partition','track']
    ref = reference[reference.model.eq(family)]
    joined = frame.merge(ref,on=keys,validate='one_to_one',suffixes=('_variant','_reference'))
    require(len(joined)==120,'Incomplete reference pairing')
    for field in ['sample_count','positive_count','negative_count','assigned_count','qc_excluded_count']:
        require(joined[field+'_variant'].equals(joined[field+'_reference']),'Reference denominators differ')
    rows=[]
    for r in joined.to_dict('records'):
        for m in METRICS:
            rows.append({**{k:r[k] for k in keys},'reference':family,'metric':m,
                'variant':r[m+'_variant'],'reference_value':r[m+'_reference'],
                'difference':r[m+'_variant']-r[m+'_reference'],'lower_is_better':m=='brier'})
    return pd.DataFrame(rows)


def summarize_pairs(frame, groups, column='difference'):
    result=frame.groupby(groups)[column].agg(['count','mean','median','std','min','max']).reset_index()
    counts=frame.assign(higher=frame[column]>0,lower=frame[column]<0,tied=frame[column]==0).groupby(groups)[['higher','lower','tied']].sum().reset_index()
    return result.merge(counts,on=groups,validate='one_to_one')


def table(frame):
    def cell(v):
        return f'{v:.7g}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def primary(frame):
    return frame[frame.partition.eq('test') & frame.track.eq('primary')]


def analyze_tables():
    runs=load_csv(META/'run_outcomes.csv'); frame=load_csv(META/'metrics.csv')
    comparisons=pd.concat([paired_reference(frame,load_csv('data/stage5/metrics.csv'),'dual_view_cnn'),
        *[paired_reference(frame,load_csv('data/stage4b/metrics.csv'),f) for f in ['random_forest','logistic_regression']]],ignore_index=True)
    csv('stage5_comparisons',comparisons[comparisons.reference.eq('dual_view_cnn')])
    csv('baseline_comparisons',comparisons[comparisons.reference.ne('dual_view_cnn')])
    old=load_csv(META/'reference_comparisons.csv').copy()
    mapped=comparisons.copy(); mapped['reference']=mapped.reference.map({'dual_view_cnn':'stage5_cnn:dual_view_cnn','random_forest':'stage4b:random_forest','logistic_regression':'stage4b:logistic_regression'})
    keys=KEYS+['partition','track','reference','metric']
    joined=old.merge(mapped,on=keys,validate='one_to_one')
    require(len(joined)==len(old)==len(comparisons) and np.allclose(joined.variant_minus_reference,joined.difference,rtol=0,atol=1e-12),'Runner reference comparisons differ')
    csv('comparison_summary',summarize_pairs(comparisons,['reference','strategy','partition','track','metric']))
    splits=load_csv(META/'paired_differences.csv')
    csv('split_comparison_summary',summarize_pairs(splits,['partition','track','metric'],'object_minus_grouped'))
    fields=['validation_range','validation_std','hidden_zero_fraction','hidden_variance','best_epoch','stopping_epoch','validation_bce','validation_roc_auc']
    summary=runs[fields].agg(['mean','median','std','min','max']).T.reset_index(names='measurement')
    csv('stability_summary',summary)
    old_runs=load_csv('data/stage5a/run_diagnostics.csv'); old_acts=load_csv('data/stage5a/activation_diagnostics.csv')
    old_runs['group']=np.where(old_runs.few_values,'Stage 5 collapsed','Stage 5 non-collapsed')
    old_acts=old_acts.merge(old_runs[KEYS+['group']],on=KEYS,validate='many_to_one')
    new_acts=load_csv(META/'activation_diagnostics.csv'); new_acts['group']='Stage 5B overall'
    acts=pd.concat([old_acts,new_acts],ignore_index=True)
    acts=acts[acts.layer.isin(['global_flatten','local_flatten','hidden','pre_sigmoid_logit'])]
    groups=acts.groupby(['group','layer'])[['mean','std','zero_fraction','between_object_variance']].mean().reset_index()
    groups['runs']=acts.groupby(['group','layer']).size().to_numpy()
    csv('activation_group_comparison',groups)
    diversity=load_csv(META/'output_diversity.csv'); diversity['group']='Stage 5B overall'
    csv('diversity_group_comparison',pd.concat([old_runs,diversity]).groupby('group')[['probability_range','probability_std','unique_probability_count']].agg(['mean','min','max']).pipe(lambda x:x.set_axis(['_'.join(c) for c in x.columns],axis=1)).reset_index())
    validation=load_csv('data/stage5/metrics.csv'); validation=validation[validation.partition.eq('validation')&validation.track.eq('primary')]
    matched=runs.merge(validation[KEYS+['roc_auc']],on=KEYS,validate='one_to_one')
    matched['validation_auc_difference']=matched.validation_roc_auc-matched.roc_auc
    csv('validation_comparisons',matched[KEYS+['validation_bce','stage5_validation_bce','validation_bce_minus_stage5','validation_roc_auc','roc_auc','validation_auc_difference']])
    # Include summaries of counts and prevalence as well as conventional metrics.
    fields=list(METRICS)+['tn','fp','fn','tp','sample_count','positive_count','negative_count','positive_prevalence','assigned_count','qc_excluded_count']
    long=frame[frame.partition.eq('test')].melt(id_vars=KEYS+['track'],value_vars=fields,var_name='metric',value_name='value')
    summary=long.groupby(['strategy','track','metric']).value.agg(['count','mean','median','std','min','max']).reset_index()
    summary['range']=summary['max']-summary['min']; csv('test_descriptive_summary',summary)
    csv('test_metrics',frame[frame.partition.eq('test')])
    q=frame[frame.partition.eq('test')].pivot(index=KEYS,columns='track',values=list(METRICS)+['assigned_count','sample_count','qc_excluded_count','positive_count','negative_count','positive_prevalence'])
    q.columns=['_'.join(c) for c in q.columns]; q=q.reset_index()
    for m in METRICS:q[m+'_qc_minus_primary']=q[m+'_qc_pass']-q[m+'_primary']
    csv('qc_sensitivity',q)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    out=META/'figures'; out.mkdir(exist_ok=True)
    def finish(fig,name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=150); plt.close(fig)
    runs=load_csv(META/'run_outcomes.csv'); old=load_csv('data/stage5a/run_diagnostics.csv')
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3)
    ax.bar(x-.18,[old.range_below_1e7.sum(),old.few_values.sum(),7],.36,label='Stage 5')
    ax.bar(x+.18,[runs.severe.sum(),runs.few_values.sum(),runs.hidden_entirely_zero.sum()],.36,label='Stage 5B')
    ax.set(xticks=x,xticklabels=['Range < 1e-7','At most 3 values','Zero hidden probe'],ylabel='Runs / 20',title='Frozen collapse criteria; probability flags use validation for Stage 5B')
    ax.legend(); finish(fig,'collapse_stability')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,s in zip(axes,['object_stratified','host_group_stratified']):
        grid=np.linspace(0,1,201); curves=[]
        for i in range(10):
            pred=load_csv(ROOT/s/f'repeat-{i:02d}'/'predictions.csv'); pred=pred[pred.partition.eq('test')]
            fpr,tpr,_=roc_curve(pred.label,pred.probability); ax.plot(fpr,tpr,alpha=.2,color='C0'); curves.append(np.interp(grid,fpr,tpr))
        ax.plot(grid,np.mean(curves,axis=0),label='Mean interpolated curve'); ax.plot([0,1],[0,1],'k--',alpha=.4)
        ax.set(title=s,xlabel='False positive rate',ylabel='True positive rate'); ax.legend(fontsize=7)
    finish(fig,'test_roc')
    comp=primary(load_csv(META/'comparison_summary.csv'))
    for ref,filename in [('dual_view_cnn','paired_stage5_roc_auc'),('random_forest','paired_random_forest_roc_auc')]:
        source=load_csv(META/('stage5_comparisons.csv' if ref=='dual_view_cnn' else 'baseline_comparisons.csv'))
        source=primary(source); source=source[source.reference.eq(ref)&source.metric.eq('roc_auc')]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,(s,rows) in zip(axes,source.groupby('strategy')):
            for r in rows.itertuples():ax.plot([0,1],[r.reference_value,r.variant],'o-',alpha=.7)
            ax.set(xticks=[0,1],xticklabels=[ref,'Stage 5B'],ylabel='Test ROC-AUC',title=s)
        finish(fig,filename)
    split=primary(load_csv(META/'paired_differences.csv')); split=split[split.metric.eq('roc_auc')]
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(split.repeat_id,split.object_minus_grouped); ax.axhline(0,color='black',lw=.7)
    ax.set(xlabel='Repeat ID',ylabel='Object minus grouped test ROC-AUC'); finish(fig,'object_vs_grouped')
    acts=load_csv(META/'activation_group_comparison.csv'); fig,ax=plt.subplots(figsize=(9,4))
    for g,rows in acts.groupby('group'):ax.plot(rows.layer,rows.between_object_variance,'o-',label=g)
    ax.set_yscale('symlog',linthresh=1e-12); ax.set(ylabel='Mean between-object variance'); ax.legend(fontsize=8); finish(fig,'activation_variance')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label,path,color in [('Stage 5','data/stage5a/training_histories.csv','C1'),('Stage 5B',META/'training_histories.csv','C0')]:
        h=load_csv(path)
        for ax,column in zip(axes,['validation_loss','validation_roc_auc']):
            for _,v in h.groupby(KEYS):ax.plot(v.epoch,v[column],color=color,alpha=.12)
            avg=h.groupby('epoch')[column].mean(); ax.plot(avg.index,avg,label=label,color=color)
            ax.set(xlabel='Epoch (surviving runs only)',ylabel=column); ax.legend()
    finish(fig,'learning_histories')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,s in zip(axes,['object_stratified','host_group_stratified']):
        for label,path,family in [('Stage 5',Path('data/stage5/calibration_bins.csv'),'dual_view_cnn'),('Stage 5B',META/'calibration_bins.csv','dual_view_cnn_leaky_relu_001'),('Random Forest',Path('data/stage4b/calibration_bins.csv'),'random_forest')]:
            bins=primary(load_csv(path)); bins=bins[(bins.strategy==s)&(bins.model==family)]
            for _,b in bins.groupby('repeat_id'):ax.plot(b.mean_probability,b.observed_fraction,alpha=.1)
            b=bins.groupby('bin_id')[['mean_probability','observed_fraction']].mean(); ax.plot(b.mean_probability,b.observed_fraction,'o-',label=label)
        ax.plot([0,1],[0,1],'k--',alpha=.4); ax.set(title=s,xlabel='Mean predicted probability',ylabel='Observed fraction'); ax.legend(fontsize=7)
    finish(fig,'calibration')


def report(certified=False):
    runs=load_csv(META/'run_outcomes.csv'); integrity=read(META/'run_integrity.json')
    summary=load_csv(META/'test_descriptive_summary.csv'); comps=primary(load_csv(META/'comparison_summary.csv'))
    splits=primary(load_csv(META/'paired_differences.csv')); split_summary=primary(load_csv(META/'split_comparison_summary.csv'))
    acts=load_csv(META/'activation_group_comparison.csv'); replayed=read(META/'prediction_verification.json')
    old=load_csv('data/stage5a/run_diagnostics.csv'); diversity=load_csv(META/'output_diversity.csv')
    all_fixed=not bool(runs.unstable.any())
    rf=comps[(comps.reference=='random_forest')&(comps.metric=='roc_auc')]
    finding='B' if all_fixed and (rf['mean']<0).all() else 'A' if all_fixed else 'C' if runs.unstable.sum()<7 else 'D'
    text='# Stage 5B: LeakyReLU fusion activation ablation\n\n'
    text+='Status: **'+('certified' if certified else 'analysis complete; certification pending offline tests')+'**. All 20 intended fits retained. No scientific fits executed during finalization.\n\n'
    text+='## 1. Objective\n\nTest whether replacing the fusion ReLU materially reduces the zero-representation collapse diagnosed in Stage 5A. This is one controlled activation ablation, not a search using test results.\n\n'
    text+='## 2. Frozen change\n\n`Dense(32, ReLU)` became `Dense(32, linear) -> LeakyReLU(negative_slope=0.01)`. AST comparison after reversing these two edits is identical to Stage 5. Configuration comparison preserves convolutional branches, filters (16/32), kernels (5), pooling (global 5/local 2), Flatten widths (2560/1600), Dense width 32, dropout 0.2, 138,561 parameters, Adam at 0.001, BCE, batch 64, maximum 100 epochs, patience 10, restore-best weights, no class/sample weighting, global clipnorm 1, threshold 0.5, seeds, splits, arrays, cohort and QC policy. Matched initialization hashes agree with Stage 5. Experiment identity and provenance/stability metadata are the other configuration differences.\n\n'
    text+='## 3. Run integrity and timing\n\n'
    text+=f"Exactly 2 strategies x 10 repeats; 20 models and prediction files, 132,740 prediction rows, 120 metric rows and 1,200 calibration rows. Completion receipts, runtime hashes, histories, identities and denominators passed. The nine pre-interruption fits remain byte-identical; no duplicate scientific outputs were created.\n\nCompleted fits total **{integrity['fit_and_artifact_seconds']:.3f} seconds** ({integrity['fit_and_artifact_seconds']/3600:.3f} hours), including per-fit evaluation/serialization. Recorded completed epoch time totals {integrity['completed_epoch_seconds']:.3f} seconds. The last resumed process took {integrity['resumed_process_seconds']:.3f} seconds; it is not the experiment total. Exact total including the discarded, unfinished fit cannot be reconstructed from durable histories, so no complete all-attempt wall time is claimed.\n\n"
    text+='## 4. Validation gate before test evaluation\n\n'
    text+=f"Frozen gate SHA256 `{integrity['gate_sha256']}`. Log gate event: {integrity['gate_time']}; first test event: {integrity['first_test_time']}. Each evaluation receipt binds to that gate. Stability and validation conclusions below use the pre-test records; subsequent test performance did not change the model or rules.\n\n"
    text+='## 5. Collapse and validation stability\n\n'
    text+=f"Stage 5B: **{int(runs.severe.sum())}/20 severe**, **{int(runs.few_values.sum())}/20 at most three values**, **{int(runs.hidden_entirely_zero.sum())}/20 fully zero fusion probes**, **{int((~runs.finite).sum())}/20 nonfinite runs**, and {int(runs.epoch_cap_reached.sum())}/20 epoch caps. Stage 5 had 4 severe and 7 low-diversity runs; all seven had fully zero fusion probes. These counts overlap and must not be added.\n\n"
    text+=f"LeakyReLU {'eliminated the observed original fusion-layer collapse in all 20 runs' if all_fixed else 'did not eliminate all frozen instability flags'}. Stage 5A probability flags used the full cohort; the pre-test Stage 5B gate used validation only. A directly comparable full-cohort check gives {int(diversity.range_below_1e7.sum())} severe and {int(diversity.few_values.sum())} low-diversity Stage 5B runs, versus {int(old.range_below_1e7.sum())} and {int(old.few_values.sum())} for Stage 5.\n\n"
    text+=table(load_csv(META/'stability_summary.csv'))+'\nValidation BCE is the restored best-checkpoint BCE. Validation ROC-AUC is read from that same selected epoch, not the stopping epoch. Hidden variance here is the variance over all sampled objects and units; between-object variance appears next. Sample standard deviations across runs use ddof=1.\n\n'
    batch=load_csv(META/'inference_cohort_comparison.csv')
    text+=f"Pre-gate train/validation-only inference and post-gate full-cohort inference use different batch compositions. Their maximum probability difference is {batch.max_probability_difference.max():.9g}, with {int(batch.changed_labels.sum())} changed labels. These are not identical-context replay calls; no new equivalence tolerance is introduced. The frozen pre-gate records remain authoritative, while the fresh replay below uses the original full-cohort order and batch size and must satisfy the existing 1e-7 tolerance.\n\n"
    val=load_csv(META/'validation_comparisons.csv')
    text+=table(val.groupby('strategy')[['validation_bce','stage5_validation_bce','validation_bce_minus_stage5','validation_roc_auc','roc_auc','validation_auc_difference']].mean().reset_index())+'\n![Collapse](../data/stage5b/figures/collapse_stability.png)\n\n'
    text+='## 6. Activation findings\n\nSame 128 training KOIs per run, selected by smallest SHA256(KOI), verified against the Stage 5A lists. All activations are inference-only with dropout disabled. Branch rows are Flatten representations; hidden is post-ReLU for Stage 5 and post-LeakyReLU for Stage 5B. Logits are measured before sigmoid, not reconstructed from rounded probabilities. `std` is over all probe entries; `between_object_variance` is mean feature-wise variance across objects, both computed in float64 with ddof=0. Group entries are unweighted means across runs. No new failure threshold is used.\n\n'
    text+=table(acts)+'\n'+table(load_csv(META/'diversity_group_comparison.csv'))+'\n![Activations](../data/stage5b/figures/activation_variance.png)\n\n![Histories](../data/stage5b/figures/learning_histories.png)\n\nHistory means at late epochs include only runs still training, and are descriptive rather than matched epoch effects.\n\n'
    text+='## 7. Primary test performance\n\nAll ten repeats per strategy; no repeated observations pooled into a headline metric. Threshold remains 0.5. AP and trapezoidal PR-AUC are distinct. Confusion counts and prevalence use the original assigned test cohort.\n\n'
    text+=table(summary[summary.track.eq('primary')].drop(columns='track'))+'\nFull per-repeat metrics, counts and prevalence: [test_metrics.csv](../data/stage5b/test_metrics.csv).\n\n![ROC](../data/stage5b/figures/test_roc.png)\n\nMean interpolated ROC curves are visual summaries only; headline AUC is the mean of ten run-specific AUCs.\n\n'
    for section,family,label,artifact in [(8,'dual_view_cnn','frozen Stage 5','stage5_comparisons'),(9,'random_forest','Random Forest (principal baseline)','baseline_comparisons'),(10,'logistic_regression','Logistic Regression (secondary baseline)','baseline_comparisons')]:
        text+=f'## {section}. Stage 5B versus {label}\n\nDifferences are Stage 5B minus reference. Positive ROC-AUC/AP/F1/accuracy is better; **negative Brier is better**. Higher/lower/tied are literal numeric comparisons, not universal win/loss counts. All 20 matched comparisons are retained in [{artifact}.csv](../data/stage5b/{artifact}.csv).\n\n'
        text+=table(comps[(comps.reference==family)&comps.metric.isin(['roc_auc','average_precision','brier','f1','accuracy'])].drop(columns=['reference','partition','track']))+'\n'
    text+='![Stage 5 paired](../data/stage5b/figures/paired_stage5_roc_auc.png)\n\n![Random Forest paired](../data/stage5b/figures/paired_random_forest_roc_auc.png)\n\n'
    text+='## 11. Object versus host-group split\n\nObject minus grouped, paired by repeat ID. The ten seed-matched differences are descriptive; overlapping repeated test sets are not independent replications. A small difference is not equivalence, and changing the split changes training/test composition as well as host overlap, so this is not a pure causal host-leakage estimate.\n\n'
    selected=splits[splits.metric.isin(['roc_auc','average_precision','brier','f1','accuracy'])]
    text+=table(selected.pivot(index='repeat_id',columns='metric',values='object_minus_grouped').reset_index())+'\n'
    text+=table(split_summary[split_summary.metric.isin(['roc_auc','average_precision','brier','f1','accuracy'])].drop(columns=['partition','track']))+'\n![Split comparison](../data/stage5b/figures/object_vs_grouped.png)\n\n'
    text+='## 12. QC sensitivity\n\nSame saved predictions, evaluation-only QC subset; no QC retraining. The full cohort has 6,637 KOIs, 6,354 QC-pass and 283 excluded. Per-repeat original denominator, excluded count, class counts and prevalence are retained in [qc_sensitivity.csv](../data/stage5b/qc_sensitivity.csv) and [test_metrics.csv](../data/stage5b/test_metrics.csv).\n\n'
    text+=table(summary[summary.metric.isin(['roc_auc','brier','sample_count','assigned_count','qc_excluded_count','positive_count','negative_count','positive_prevalence'])])+'\n'
    text+='## 13. Calibration\n\nTen fixed equal-width probability bins, Brier on each repeat and track, no post-hoc calibration. Empty bins are undefined. Reliability figure averages each nonempty bin equally across repeats; faint curves retain run variability. Repeated KOIs are not treated as independent evidence. Lower Brier is better; improvement in ranking does not establish adequate calibration.\n\n'
    text+=table(summary[summary.metric.eq('brier')])+'\n![Calibration](../data/stage5b/figures/calibration.png)\n\n'
    text+='## 14. Interpretation\n\nA: collapse fixed and substantial predictive improvement; B: collapse fixed but prediction remains substantially below Random Forest; C: collapse reduced but persists; D: failure not materially changed. These are descriptive, not predeclared statistical decision thresholds. A and B can overlap: improved prediction need not approach Random Forest.\n\n'
    text+=f"**Best-supported finding: {finding}.** Stabilization is distinct from predictive improvement. The fixed intervention removed the observed endpoint zero-fusion failure, supporting a role for the original fusion activation under this training setup. Paired test differences quantify predictive change without dropping poor runs; the Random Forest deficit remains explicit. Negative-slope activation permits gradients/values on the negative side, but saved endpoint probes cannot prove the trajectory, exact onset, or a unique cause of the original failure. The ablation does not establish that other bottlenecks are absent.\n\n"
    text+='## 15. Limitations and replay\n\nOne cohort, one fixed architecture and ten paired seeds; no independent external confirmation. Test sets overlap across repeats, so descriptive repeat variability is not an independent-sample confidence interval. Training probe covers 128 objects per run. Diagnostics do not reconstruct training trajectories. Stage 4A canonical/cache certification was reused with pinned hashes instead of downloading or re-auditing raw archives.\n\n'
    text+=f"One fresh-process saved-model replay verified {replayed['models']} models and {replayed['rows']:,} probabilities; maximum absolute difference {replayed['max_probability_difference']:.9g}, existing tolerance atol=1e-7, rtol=0. KOI ordering, labels, folds, QC, predictions, recomputed metrics/calibration and unchanged weights passed. No training was executed.\n\n"
    text+='## 16. Next scientific question\n\nWith collapse stabilized, why does the fixed CNN still underperform the principal baseline? A separately authorized, predeclared training/validation-only investigation could test whether the near-flat branch representations and normalization preserve useful transit morphology. Retain Random Forest as the principal benchmark; reserve a new untouched evaluation plan before further model selection. No additional ablation or Stage 6 has been started.\n\n'
    text+='## 17. Provenance, certification and artifact hashes\n\n'
    text+=f"Parent Stage 5 commit: `{integrity['parent_commit']}`. Stage 1–5A scientific anchors verified unchanged; {integrity['parent_scientific_files_verified']} parent tracked scientific files plus {integrity['frozen_files_verified']} frozen entries checked. Runtime model/prediction/history hashes: [runtime_checksums.sha256](../data/stage5b/runtime_checksums.sha256). Full compact output hashes: [checksums.sha256](../data/stage5b/checksums.sha256). Environment, seeds, deterministic settings, code and output hashes are recorded in provenance/certification.\n\n"
    text+=f"{len(integrity['historical_git_checkout_line_ending_only_differences'])} historical text files have Git LF blobs and pre-existing CRLF checkout bytes; normalized content is identical and these files were not edited. All explicit scientific SHA256 anchors were checked byte-for-byte without normalization. Local protected-input byte hashes were captured before analysis and checked again at certification.\n\n"
    if certified:
        cert=read(META/'certification.json'); text+=f"Offline suite: {cert['tests']['passed']} passed, {cert['tests']['skipped']} skipped; network tests excluded. Focused Stage 5B tests passed. `git diff --check` passed. One Stage 5B commit is authorized only after these checks.\n\n"
    paths=[CONFIG,FREEZE,Path('docs/kepler_stage5a_failure_analysis.md'),META/'run_integrity.json',META/'prediction_verification.json',META/'validation_stability_summary.json',META/'metrics.csv',META/'runtime_checksums.sha256']
    if certified:paths += [META/'certification.json',META/'provenance.json']
    text+=table(pd.DataFrame([dict(artifact=p.as_posix(),sha256=sha256(p)) for p in paths]))
    REPORT.write_text(text,encoding='utf-8')
    save(META/'report_integrity.json',dict(path=REPORT.as_posix(),sha256=sha256(REPORT)))


def compact_files():
    return [p for p in sorted(META.rglob('*')) if p.is_file() and p.name not in {'checksums.sha256','runner.lock','status.json'} and p.suffix not in {'.log','.tmp'}]


def checksum_outputs():
    (META/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(META).as_posix()}\n' for p in compact_files()),encoding='utf-8')
    verify_hashes(manifest_entries(META/'checksums.sha256',META))


def certify():
    check_policy()
    verify_hashes(read(META/'protected_input_hashes.json'))
    verify_hashes(manifest_entries(META/'checksums.sha256',META))
    verify_hashes(manifest_entries(META/'runtime_checksums.sha256',ROOT))
    verify_hashes(read(META/'replay_complete.json')['sha256'])
    verify_hashes({str(REPORT):read(META/'report_integrity.json')['sha256']})
    suites=[]
    for filename in ['stage5b-focused.xml','stage5b-offline.xml']:
        suite=ET.parse(Path('data/tmp')/filename).getroot()
        nodes=list(suite.iter('testsuite'))
        require(nodes and sum(int(n.attrib.get('failures',0))+int(n.attrib.get('errors',0)) for n in nodes)==0,'Required tests failed')
        suites.append(dict(file=filename,sha256=sha256(Path('data/tmp')/filename),tests=sum(int(n.attrib['tests']) for n in nodes),
            skipped=sum(int(n.attrib.get('skipped',0)) for n in nodes)))
    subprocess.run(['git','diff','--check'],check=True)
    evidence=dict(status='passed',passed=suites[-1]['tests']-suites[-1]['skipped'],skipped=suites[-1]['skipped'],suites=suites,network_tests='excluded with -m not network')
    save(META/'test_evidence.json',evidence)
    cert=dict(status='certified',models=20,scientific_fits_executed=0,run_integrity_sha256=sha256(META/'run_integrity.json'),
        replay_sha256=sha256(META/'prediction_verification.json'),tests=evidence,git_diff_check='passed',frozen_inputs='unchanged',
        finalizer_sha256=sha256(__file__),runtime_checksums_sha256=sha256(META/'runtime_checksums.sha256'),
        result_hashes={p.relative_to(META).as_posix():sha256(p) for p in compact_files() if p.suffix in {'.csv','.png'}},
        external_confirmation=False)
    save(META/'certification.json',cert)
    provenance=read(META/'provenance.json')
    provenance.update(certification_sha256=sha256(META/'certification.json'),parent_commit=read(META/'run_integrity.json')['parent_commit'],
        environment=read(META/'prediction_verification.json')['environment'],python_executable=read(META/'prediction_verification.json')['python'],
        seeds=read(CONFIG)['seeds'],deterministic_environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','TF_DETERMINISTIC_OPS','TF_ENABLE_ONEDNN_OPTS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']},
        finalization_code_sha256={str(p):sha256(p) for p in [Path(__file__),Path('scripts/stage5b_resume_existing.py'),Path('tests/test_stage5b_finalization.py')]},
        timing=read(META/'run_integrity.json'),certification_status='certified')
    save(META/'provenance.json',provenance)
    report(certified=True); checksum_outputs()
    subprocess.run(['git','diff','--check'],check=True)
    print(json.dumps(cert,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze',action='store_true'); parser.add_argument('--certify',action='store_true')
    args=parser.parse_args()
    require(args.analyze != args.certify,'Choose exactly one finalization phase')
    if args.analyze:
        config,records,objects,assignments=audit()
        print('Integrity and isolated scientific change verified',flush=True)
        replay(config,records,objects,assignments)
        analyze_tables(); figures(); report(); checksum_outputs()
        print('Analysis complete; offline test certification pending',flush=True)
    else:certify()


if __name__=='__main__':main()
