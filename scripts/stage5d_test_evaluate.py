"""One frozen Stage 5D held-out evaluation, one independent replay, no fitting.

--evaluate seals test predictions once and builds the analysis.
--finish runs checks, certifies, writes the short checkpoint, commits and safely
pushes. A partial evaluation or replay is never silently repeated.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

# Import before TensorFlow to retain the pinned deterministic CPU environment.
from scripts import stage5d_run_fusion_width as r
import numpy as np
import pandas as pd
from kepler_baselines.metrics import metrics, calibration_bins
from kepler_baselines.inputs import read_input_tables
from kepler_cnn.artifacts import align, views

OUT = Path('data/stage5d_test')
RUNTIME = Path('data/stage5d_test_runtime')
REPORT = Path('docs/kepler_stage5d_test_evaluation.md')
CHECKPOINT = Path('docs/post_stage5d_checkpoint.md')
PAIRED = ['roc_auc','average_precision','brier','f1','accuracy']


def save(path, value):
    r.atomic(path, value)


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def write(name, frame):
    r.write_csv(OUT/(name+'.csv'),frame)


def status(phase, **extra):
    save(RUNTIME/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**extra))


def verify_inventory(path, root):
    entries = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ',1)
        r.require(name not in entries,'Duplicate checksum entry')
        entries[name] = digest
    r.verify_hashes({str(root/n):h for n,h in entries.items()})
    return entries


def protected_snapshot():
    names = subprocess.check_output(['git','ls-files','--','configs','data','docs','kepler_catalog',
        'kepler_splits','kepler_scale','kepler_lightcurves','kepler_baselines','kepler_cnn',
        'kepler_cnn_variant','kepler_cnn_width128','scripts','tests'],text=True).splitlines()
    return {n:r.sha256(n) for n in names if not n.startswith(('data/stage5d_test/',
        'scripts/stage5d_test_evaluate.py','tests/test_stage5d_test_evaluation.py'))
        and n not in {str(REPORT).replace('\\','/'),str(CHECKPOINT).replace('\\','/')}}


def validate_plan(plan, config, runs):
    expected={(s,i,seed) for s in config['strategies'] for i,seed in enumerate(config['seeds'])}
    r.require(len(plan)==len(runs)==20,'Expected exactly 20 models')
    for collection in [plan,runs]:
        r.require({(x['strategy'],x['repeat_id'],x['seed']) for x in collection}==expected,'Model matrix/seed mismatch')
    by_key={r.key_for(x):x for x in runs}
    for item in plan:
        directory=r.ROOT/r.key_for(item)/'cnn'
        r.require(Path(item['path'])==directory/'model.keras','Unexpected model path')
        r.require(r.sha256(item['path'])==item['sha256'],'Frozen model checksum mismatch')
        who=r.identity(item['strategy'],item['repeat_id'],item['seed'])
        record=by_key[r.key_for(item)]
        receipt=r.receipt(directory,r.sha256(r.FREEZE),who,dict(scope_sha256=record['scope_sha256']),r.CNN_FILES)
        r.require(receipt==record,'Model completion identity differs')
        r.selected_checkpoint(csv(directory/'history.csv'),record)
    return by_key


def gate():
    r.require(not (OUT/'pre_test_integrity.json').exists(),'Pre-test gate already exists: use completed receipts; never repeat evaluation')
    r.require(not RUNTIME.exists() or not any((RUNTIME/'runs').glob('*/*/started.json')),'Partial evaluation exists; stop for review')
    config=r.read(r.CONFIG); parent=r.read(config['parent_config'])
    r.verify_hashes(r.read(r.FREEZE)['sha256'])
    for folder in ['stage4b','stage5b','stage5c','stage5d']:
        verify_inventory(Path('data')/folder/'checksums.sha256',Path('data')/folder)
    verify_inventory(Path('data/stage5d/runtime_checksums.sha256'),r.ROOT)
    preparation=r.read('data/stage5d/implementation_verification.json')
    r.verify_hashes(preparation['source_sha256']);r.verify_hashes(preparation['protected_sha256'])
    decision=r.read('data/stage5d/decision_summary.json')
    r.require(decision['recommendation']=='GO' and not decision['held_out_evaluated'],'Validation decision is not a sealed GO')
    r.require(r.read('data/stage5d/certification.json')['status']=='passed','Validation certification failed')
    plan=decision['proposed_test_protocol']['models'];runs=r.read('data/stage5d/run_manifest.json')['runs']
    validate_plan(plan,config,runs)
    objects,_,assignments=read_input_tables(parent)
    r.require(objects.equals(csv('data/stage4b/objects.csv')),'Feature row identity differs')
    cache=Path(parent['baseline_runtime'])/'features.npy'
    r.require(r.sha256(cache)==config['feature_cache_sha256'],'Frozen feature cache differs')
    from kepler_cnn.runner import context
    r.require(context()==r.read('data/stage5/run_manifest.json')['runs'][0]['context'],'Pinned environment differs')
    protected=protected_snapshot()
    receipt=dict(status='passed',gate_utc=datetime.now(timezone.utc).isoformat(),models=plan,
        protected_sha256=protected,freeze_sha256=r.sha256(r.FREEZE),cache=str(cache),cache_sha256=r.sha256(cache),
        checkpoint='earliest minimum validation BCE; saved weights only',test_inference_started=False,
        source_sha256=r.sha256(__file__),threshold=.5,strategies=config['strategies'],seeds=config['seeds'])
    save(OUT/'pre_test_integrity.json',receipt)  # Before test feature selection or inference.
    return receipt,objects,assignments


def bce(y,p):
    # Same frozen Keras BCE used for CNN validation; clips at Keras epsilon.
    import tensorflow as tf
    return float(tf.keras.losses.BinaryCrossentropy()(np.asarray(y)[:,None],np.asarray(p)[:,None]).numpy())


def measure(prediction):
    rows=[];bins=[]
    for track in ['primary','qc_pass']:
        p=prediction if track=='primary' else prediction.loc[prediction.qc_pass]
        first=prediction.iloc[0]
        identity=dict(strategy=str(first['strategy']),repeat_id=int(first['repeat_id']),seed=int(first['seed']))
        identity.update(partition='test',track=track)
        rows.append(dict(**identity,assigned_count=len(prediction),qc_excluded_count=len(prediction)-len(p),
            bce=bce(p.label,p.probability),**metrics(p.label,p.probability)))
        bins.extend(dict(**identity,**x) for x in calibration_bins(p.label,p.probability,bins=10))
    return rows,bins


def test_frame(objects,assignments,item):
    assigned=align(objects,assignments,item['strategy'],item['repeat_id'])
    mask=assigned.partition.eq('test').to_numpy()
    frame=objects.loc[mask,r.IDENTITY+['qc_pass']].reset_index(drop=True)
    frame['partition']='test';frame['fold_id']=assigned.loc[mask,'fold_id'].to_numpy()
    for k in ['strategy','repeat_id','seed']:frame[k]=item[k]
    return mask,frame


def verify_prediction(expected,actual):
    from kepler_cnn.model import verify_probability
    columns=r.IDENTITY+['qc_pass','partition','fold_id','strategy','repeat_id','seed']
    r.require(expected[columns].equals(actual[columns]),'Replay KOI order/labels/scope differ')
    verify_probability(expected.probability.to_numpy(),actual.probability.to_numpy())
    r.require(np.array_equal(expected.prediction,actual.prediction),'Replay classification differs')
    return float(np.max(np.abs(expected.probability-actual.probability)))


def inference(replay=False):
    if replay:
        receipt=r.read(OUT/'pre_test_integrity.json')
        r.verify_hashes(receipt['protected_sha256'])
        r.require(not (RUNTIME/'replay_started.json').exists(),'Independent replay already started; never repeat')
        save(RUNTIME/'replay_started.json',dict(started=True))
        parent=r.read(r.read(r.CONFIG)['parent_config'])
        objects,_,assignments=read_input_tables(parent)
        r.require(r.sha256(receipt['cache'])==receipt['cache_sha256'],'Replay cache differs')
    else:
        receipt,objects,assignments=gate()
    import tensorflow as tf
    from kepler_cnn.model import configure,predict,weight_hash
    configure()
    matrix=np.load(receipt['cache'],mmap_mode='r',allow_pickle=False)
    all_metrics=[];all_bins=[];verified=[]
    for item in receipt['models']:
        directory=RUNTIME/'runs'/r.key_for(item)
        r.require(r.sha256(item['path'])==item['sha256'],'Model changed after gate')
        if not replay:
            r.require(not directory.exists(),'Duplicate/partial test run; stop for review')
            save(directory/'started.json',dict(identity=item,gate_sha256=r.sha256(OUT/'pre_test_integrity.json')))
        else:
            complete=r.read(directory/'complete.json')
            r.verify_hashes({str(directory/n):h for n,h in complete['sha256'].items()})
        mask,frame=test_frame(objects,assignments,item)
        # Materialize only the original test rows after the complete gate passed.
        inputs=views(matrix[mask])
        model=tf.keras.models.load_model(item['path'],compile=False)
        before=weight_hash(model)
        record=r.read(r.ROOT/r.key_for(item)/'cnn'/'record.json')
        r.require(before==record['final_weight_sha256'],'Checkpoint weight identity differs')
        probability=predict(model,inputs)
        r.require(weight_hash(model)==before,'Inference changed weights')
        frame['probability']=probability.astype(float);frame['prediction']=(probability>=.5).astype(int)
        measured,bins=measure(frame)
        if replay:
            expected=csv(directory/'test_predictions.csv')
            error=verify_prediction(expected,frame)
            old=r.read(directory/'complete.json')['metrics']
            for a,b in zip(old,measured):
                for k in ['sample_count','positive_count','negative_count','tn','fp','fn','tp']+PAIRED+['bce','pr_auc_trapezoidal']:
                    r.require(abs(a[k]-b[k])<=1e-12,'Independent replay metric differs: '+k)
            verified.append(dict(strategy=item['strategy'],repeat_id=item['repeat_id'],seed=item['seed'],
                rows=len(frame),maximum_absolute_probability_error=error,model_sha256=item['sha256']))
        else:
            r.write_csv(directory/'test_predictions.csv',frame)
            save(directory/'complete.json',dict(identity=item,gate_sha256=r.sha256(OUT/'pre_test_integrity.json'),
                metrics=measured,sha256={'test_predictions.csv':r.sha256(directory/'test_predictions.csv')}))
            all_metrics.extend(measured);all_bins.extend(bins)
        del model,inputs;tf.keras.backend.clear_session()
    if replay:
        save(OUT/'prediction_verification.json',dict(status='passed',fresh_process=True,models=20,
            probability_tolerance_atol=1e-7,rtol=0,metric_tolerance_atol=1e-12,runs=verified))
    else:
        write('test_metrics',pd.DataFrame(all_metrics));write('calibration_bins',pd.DataFrame(all_bins))
        save(RUNTIME/'evaluation_complete.json',dict(models=20,scientific_fits=0,
            test_metrics_sha256=r.sha256(OUT/'test_metrics.csv')))


def summary(values,lower=False):
    a=pd.Series(values,dtype=float)
    direction=-a if lower else a
    return dict(count=len(a),mean=a.mean(),median=a.median(),sd=a.std(ddof=1),min=a.min(),max=a.max(),
        improved=int((direction>0).sum()),worsened=int((direction<0).sum()),tied=int((direction==0).sum()))


def paired(cnn,baseline):
    keys=['strategy','repeat_id','partition','track']
    joined=cnn.merge(baseline,on=keys,suffixes=('_5d','_baseline'),validate='one_to_one')
    r.require(len(joined)==40,'Incomplete baseline pairing')
    for name in ['sample_count','positive_count','negative_count','assigned_count','qc_excluded_count']:
        r.require(np.array_equal(joined[name+'_5d'],joined[name+'_baseline']),'Baseline denominator mismatch: '+name)
    records=[]
    for x in joined.to_dict('records'):
        for metric in PAIRED:
            records.append({**{k:x[k] for k in keys},'metric':metric,'stage5d':x[metric+'_5d'],
                'baseline':x[metric+'_baseline'],'delta':x[metric+'_5d']-x[metric+'_baseline']})
    return pd.DataFrame(records)


def markdown(frame):
    def cell(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def analyze():
    data=csv(OUT/'test_metrics.csv');primary=data[data.track.eq('primary')]
    baseline5b=csv('data/stage5b/metrics.csv');baseline4b=csv('data/stage4b/metrics.csv')
    tables={};summaries=[]
    for name,baseline in [('stage5b',baseline5b),('random_forest',baseline4b[baseline4b.model.eq('random_forest')]),
                          ('logistic',baseline4b[baseline4b.model.eq('logistic_regression')])]:
        table=paired(data,baseline[baseline.partition.eq('test')])
        write('paired_'+name+'_stage5d',table);tables[name]=table
        for (strategy,track,metric),g in table.groupby(['strategy','track','metric']):
            summaries.append(dict(baseline=name,strategy=strategy,track=track,metric=metric,**summary(g.delta,metric=='brier')))
        for (track,metric),g in table.groupby(['track','metric']):
            summaries.append(dict(baseline=name,strategy='overall',track=track,metric=metric,**summary(g.delta,metric=='brier')))
    comparisons=pd.DataFrame(summaries);write('paired_summary',comparisons)
    means=[]
    for (strategy,track),g in data.groupby(['strategy','track']):
        for metric in ['roc_auc','bce','average_precision','pr_auc_trapezoidal','brier','precision','recall','f1','accuracy']:
            means.append(dict(strategy=strategy,track=track,metric=metric,**summary(g[metric],metric in ['bce','brier'])))
    means=pd.DataFrame(means);write('test_summary',means)
    differences=[]
    for metric in PAIRED:
        p=primary.pivot(index='repeat_id',columns='strategy',values=metric)
        for repeat,row in p.iterrows():
            differences.append(dict(repeat_id=repeat,metric=metric,object=row.object_stratified,host_group=row.host_group_stratified,
                difference=row.object_stratified-row.host_group_stratified))
    differences=pd.DataFrame(differences);write('object_group_test_differences',differences)
    split_summary=pd.DataFrame([dict(metric=m,**summary(g.difference,m=='brier')) for m,g in differences.groupby('metric')])
    write('object_group_summary',split_summary)
    write('qc_sensitivity',data[['strategy','repeat_id','seed','track','assigned_count','sample_count','qc_excluded_count','positive_prevalence']+PAIRED])
    val=csv('data/stage5d/run_summary.csv')
    b=comparisons[(comparisons.baseline=='stage5b')&(comparisons.track=='primary')&(comparisons.metric=='roc_auc')]
    generalization=val[['strategy','mean']].rename(columns={'mean':'validation_delta'}).merge(b[['strategy','mean']],on='strategy').rename(columns={'mean':'test_delta'})
    generalization['direction_reproduced']=(generalization.validation_delta*generalization.test_delta)>0
    generalization['effect']=np.where(~generalization.direction_reproduced,'reversed',np.where(generalization.test_delta>=generalization.validation_delta,'preserved/increased','reduced'))
    write('validation_to_test',generalization)
    object_delta=float(b.loc[b.strategy.eq('object_stratified'),'mean'].iloc[0]);group_delta=float(b.loc[b.strategy.eq('host_group_stratified'),'mean'].iloc[0])
    overall=float(b.loc[b.strategy.eq('overall'),'mean'].iloc[0])
    if object_delta*group_delta<0:category='E'
    elif overall<=0:category='D' if overall<0 else 'C'
    else:category='B'
    result=dict(interpretation_category=category,stage5d_test_auc=primary.groupby('strategy').roc_auc.mean().to_dict(),
        baseline_comparisons=comparisons[(comparisons.track=='primary')&(comparisons.metric=='roc_auc')].to_dict('records'),
        validation_to_test=generalization.to_dict('records'),object_group=split_summary.to_dict('records'),
        brier=data.groupby(['strategy','track']).brier.mean().reset_index().to_dict('records'),
        qc=data.groupby(['strategy','track']).agg(mean_denominator=('sample_count','mean'),mean_excluded=('qc_excluded_count','mean'),
             mean_prevalence=('positive_prevalence','mean'),roc_auc=('roc_auc','mean'),average_precision=('average_precision','mean'),brier=('brier','mean'),f1=('f1','mean'),accuracy=('accuracy','mean')).reset_index().to_dict('records'),
        representation=r.read('data/stage5d/decision_summary.json')['gap_reduction'],
        recommendation='Publication/manuscript consolidation of the frozen Kepler benchmark and representation diagnosis; no further architecture search now. Observational transfer such as Kepler-to-TESS requires a separately designed future study.')
    save(OUT/'result_summary.json',result)
    figures(primary,comparisons,csv(OUT/'calibration_bins.csv'))
    report='# Stage 5D — one-time frozen held-out evaluation\n\n'
    report+='## Frozen protocol and integrity\n\nTwenty predeclared models, two strategies × ten repeats; saved earliest-minimum-validation-BCE checkpoints, threshold 0.5. No fitting, tuning, recalibration, split change or test-based selection. pre_test_integrity.json pins every model, the Stage 3 splits, certified cache and prior tracked scientific artifacts before test feature selection. All subsequent inference uses only the matching frozen test rows. Primary track retains the complete assigned test population; QC-pass is a predefined sensitivity subset.\n\n'
    report+='## Test performance\n\n'+markdown(means[means.metric.isin(['roc_auc','bce','average_precision','brier'])].drop(columns=['improved','worsened','tied']))+'\n\n'
    report+='## Frozen matched comparisons\n\nDeltas are Stage 5D minus baseline; lower Brier is better. Sample SD is across repeats; counts reverse the direction for Brier. All twenty individual pairs for five metrics and both tracks are preserved in the paired CSVs. Denominators and class counts are checked before joining. Random Forest is the principal benchmark; Logistic Regression supplies secondary context. Certified baseline metrics are reused without refitting or recomputation.\n\n'+markdown(comparisons[comparisons.track.eq('primary')])+'\n\n'
    report+='## Validation-to-test generalization\n\n'+markdown(generalization)+'\n\n'
    report+='## Object versus host-group\n\n'+markdown(split_summary)+'\n\nThe difference is descriptive: neither a small difference nor a reversed sign proves equivalence or identifies a pure causal host-leakage effect. Repeated partitions overlap; these SDs are descriptive rather than independent-sample uncertainty estimates.\n\n'
    report+='## QC and calibration\n\n'+markdown(pd.DataFrame(result['qc']))+'\n\nPer-repeat denominators, exclusions and prevalence are in qc_sensitivity.csv. Calibration uses ten fixed equal-width bins, with empty bins retained. The reliability figure pools bin counts across repeats per strategy (objects can recur); it is descriptive, not an independent-cohort calibration estimate. No calibration model is fitted.\n\n'
    report+='## Representation diagnosis and scientific interpretation\n\nStage 5C correctly localized a major loss of fixed linear-probe discrimination at fusion. The hidden-probe AUC rose from 0.722274 to 0.826806, near the width128 concatenated AUC 0.829044. The signed concat-minus-hidden gap fell from 0.105972 to 0.002238, improving in 19/20 pairs (97.9% mean reduction). These frozen validation probes were not rerun and are a separability proxy, not information-theoretic information loss. Width128 fixed much of that probe-level loss; the following held-out comparison establishes how much translated into classification.\n\n'
    report+=f'Interpretation category **{category}**. Object mean CNN AUC change: {object_delta:+.6f}; host-group: {group_delta:+.6f}; overall: {overall:+.6f}. '
    rf=comparisons[(comparisons.baseline=='random_forest')&(comparisons.strategy=='overall')&(comparisons.track=='primary')&(comparisons.metric=='roc_auc')]['mean'].iloc[0]
    report+=f'Random Forest comparison: Stage 5D minus RF {rf:+.6f}. The fusion bottleneck was not established as the sole cause of CNN underperformance. Branches trained end-to-end, probe preservation does not guarantee a well-trained CNN output head, and repeated validation and baseline test reuse constrain independent confirmation. No new scientific adjustment follows from these test outcomes.\n\n'
    report+='## Replay, certification and artifacts\n\nprediction_verification.json records the single fresh-process replay of all twenty models: KOI/label/split order, saved probabilities, threshold labels, and metrics. Predictions, sample IDs, labels, QC flags and probabilities are preserved under the excluded data/stage5d_test_runtime/runs/ tree and bound by compact per-run completion receipts. certification.json and checksums.sha256 bind the compact results and checks. The evaluation cannot silently repeat: any started partial phase stops for review.\n\n'
    for name in ['paired_test_auc','test_differences','reliability']:
        report+=f'![{name}](../data/stage5d_test/figures/{name}.png)\n\n'
    REPORT.write_text(report,encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def figures(data,comparisons,bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    def savefig(name):plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=150);plt.close()
    fig,ax=plt.subplots(figsize=(6,4))
    for strategy,g in data.groupby('strategy'):ax.plot(g.repeat_id,g.roc_auc,'o-',label=strategy)
    ax.set(xlabel='Repeat',ylabel='Stage 5D test ROC-AUC');ax.legend(fontsize=8);savefig('paired_test_auc')
    fig,ax=plt.subplots(figsize=(6,4))
    p=comparisons[(comparisons.track=='primary')&(comparisons.metric=='roc_auc')&(comparisons.strategy!='overall')]
    for baseline,g in p.groupby('baseline'):ax.plot(g.strategy,g['mean'],'o-',label=baseline)
    ax.axhline(0,color='gray');ax.set(ylabel='Mean Stage 5D minus baseline test ROC-AUC');ax.legend(fontsize=8);savefig('test_differences')
    fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],'--',color='gray')
    for strategy,g in bins[bins.track.eq('primary')].groupby('strategy'):
        rows=[]
        for _,h in g.groupby('bin_id'):
            n=h.sample_count.sum()
            if n:rows.append((np.nansum(h.mean_probability*h.sample_count)/n,h.positive_count.sum()/n))
        a=np.array(rows);ax.plot(a[:,0],a[:,1],'o-',label=strategy)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8);savefig('reliability')


def certify():
    receipt=r.read(OUT/'pre_test_integrity.json');r.verify_hashes(receipt['protected_sha256'])
    r.require(r.read(OUT/'prediction_verification.json')['status']=='passed','Replay did not pass')
    tests={}
    for name in ['focused-tests.xml','offline-tests.xml']:
        p=OUT/name;suite=ET.parse(p).getroot()[0]
        r.require(int(suite.attrib['failures'])==int(suite.attrib['errors'])==0,'Tests failed')
        tests[name]=dict(suite.attrib,sha256=r.sha256(p),exit_code=0)
    runtime={p.relative_to(RUNTIME).as_posix():r.sha256(p) for p in sorted((RUNTIME/'runs').rglob('*')) if p.is_file()}
    r.require(len(list((RUNTIME/'runs').glob('*/*/complete.json')))==20,'Incomplete test runs')
    (OUT/'runtime_checksums.sha256').write_text(''.join(f'{h}  {n}\n' for n,h in runtime.items()),encoding='utf-8')
    result=r.read(OUT/'result_summary.json')
    save(OUT/'certification.json',dict(status='passed',models=20,scientific_fits=0,threshold=.5,
        evaluation='first and only frozen Stage 5D test evaluation plus one required independent replay',
        tests=tests,protected_files_unchanged=len(receipt['protected_sha256']),replay='passed',
        gate_sha256=r.sha256(OUT/'pre_test_integrity.json'),source_sha256=r.sha256(__file__),
        interpretation=result['interpretation_category'],report_sha256=r.sha256(REPORT)))
    # The compact checkpoint is created only after the scientific certification.
    auc=result['stage5d_test_auc'];comp=result['baseline_comparisons']
    rf=next(x['mean'] for x in comp if x['baseline']=='random_forest' and x['strategy']=='overall')
    delta=next(x['mean'] for x in comp if x['baseline']=='stage5b' and x['strategy']=='overall')
    checkpoint='# Post-Stage-5D project checkpoint\n\n'
    checkpoint+='The three strongest findings are:\n\n'
    checkpoint+='1. The certified Stage 4B flux-only Random Forest remains the principal benchmark; Stage 5D minus RF mean test ROC-AUC is '+f'{rf:+.6f}.\n'
    checkpoint+='2. Stage 5C located a strong linear-probe loss at fusion; width128 reduced the concat-to-hidden AUC gap by 97.9%, improving 19/20 matched pairs without observed collapse.\n'
    checkpoint+=f'3. Frozen Stage 5D test ROC-AUC is {auc["object_stratified"]:.6f} (object) and {auc["host_group_stratified"]:.6f} (host-group), with overall paired change {delta:+.6f} versus Stage 5B.\n\n'
    checkpoint+='The strongest negative result is that preserving linearly accessible information does not by itself make this CNN competitive with the tree baseline. The fusion bottleneck explains an important representation loss, but is not established as the sole limitation of end-to-end optimization or classification.\n\n'
    split=next(x for x in result['object_group'] if x['metric']=='roc_auc')
    checkpoint+=f'Object-minus-host-group test ROC-AUC averages {split["mean"]:+.6f} (sample SD {split["sd"]:.6f}). These split comparisons are descriptive, not evidence of equivalence or a pure causal leakage estimate; strategy-specific validation gains and reused test partitions require caution.\n\n'
    checkpoint+='Recommended next step: **publication/manuscript consolidation**, focusing on the fixed benchmark, failure diagnosis, and the distinction between probe preservation and held-out prediction. Do not start another architecture experiment now. Kepler-to-TESS observational transfer is a possible separately designed later direction, with domain and label-shift controls; it is not executed here.\n'
    CHECKPOINT.write_text(checkpoint,encoding='utf-8')
    cert=r.read(OUT/'certification.json');cert['checkpoint_sha256']=r.sha256(CHECKPOINT);save(OUT/'certification.json',cert)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='checksums.sha256']
    (OUT/'checksums.sha256').write_text(''.join(f'{r.sha256(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')
    verify_inventory(OUT/'checksums.sha256',OUT)


def finish():
    # All logs/temp files live in an ignored tree; no subprocess polling needed.
    status('focused_tests')
    commands=[('focused-tests.xml',['tests/test_stage5d_test_evaluation.py']),('offline-tests.xml',[])]
    for name,selection in commands:
        status(name)
        subprocess.run([sys.executable,'-m','pytest',*selection,'-m','not network','-q',
            '--basetemp='+str(RUNTIME/('tmp-'+name)), '--junitxml='+str(OUT/name)],check=True)
    subprocess.run(['git','diff','--check'],check=True)
    status('certifying');certify()
    publish()


def publish():
    """Publish only already-certified outputs; never rerun evaluation or tests."""
    r.require(r.read(OUT/'certification.json')['status']=='passed','No completed certification')
    r.verify_hashes(r.read(OUT/'pre_test_integrity.json')['protected_sha256'])
    verify_inventory(OUT/'checksums.sha256',OUT)
    paths=['.gitignore','.gitattributes','scripts/stage5d_test_evaluate.py','tests/test_stage5d_test_evaluation.py',OUT.as_posix(),REPORT.as_posix(),CHECKPOINT.as_posix()]
    subprocess.run(['git','add','--',*paths],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
    allowed=lambda n:n in paths or n.startswith('data/stage5d_test/')
    r.require(all(allowed(n) for n in staged),'Unexpected staged paths; refusing commit')
    r.require(all(Path(n).stat().st_size<5_000_000 for n in staged),'Unexpected large staged file; refusing commit')
    subprocess.run(['git','commit','-m','Stage 5D: certify frozen held-out width128 evaluation'],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    clean=not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    r.require(clean and branch=='main','Commit made; tree not clean/main, so no push')
    status('pushing',commit=commit)
    pushed=subprocess.run(['git','push','origin','main'],check=False)
    status('complete',certification='passed',commit=commit,push='passed' if pushed.returncode==0 else 'failed',
        git_status=subprocess.check_output(['git','status','--short'],text=True),results=str(OUT/'result_summary.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--evaluate',action='store_true');modes.add_argument('--replay',action='store_true');modes.add_argument('--finish',action='store_true')
    args=parser.parse_args()
    try:
        if args.replay:inference(True)
        elif args.finish:finish()
        else:
            status('pre_test_integrity');inference()
            status('independent_replay')
            subprocess.run([sys.executable,'-m','scripts.stage5d_test_evaluate','--replay'],check=True)
            status('reporting');analyze();status('ready_for_tests')
    except Exception as exc:
        status('stopped_for_review',error=str(exc));raise


if __name__=='__main__':main()
