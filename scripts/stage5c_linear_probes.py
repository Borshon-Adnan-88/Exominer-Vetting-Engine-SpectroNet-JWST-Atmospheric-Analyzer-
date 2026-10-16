"""Six fixed training-only linear probes; validation-only loss localization."""
from scripts import stage5b_run_activation_variant as stage5b
import json
import os
from pathlib import Path
import time
import warnings
import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from kepler_baselines.inputs import require, sha256, read_input_tables
from kepler_baselines.metrics import metrics
from kepler_baselines.runner import write_csv
from kepler_cnn.artifacts import views, align
from scripts.stage5a_run_diagnostics import verify_hashes
from scripts.stage5b_resume_existing import resilient_atomic as atomic

OUT = Path('data/stage5c')
PROBES = ['raw_global', 'raw_local', 'global_branch', 'local_branch', 'concatenated_branch', 'fusion_hidden']


def read(path):
    return json.loads(Path(path).read_text())


def fit_probe(features, labels, training, parameters):
    features = np.asarray(features, dtype=np.float64)
    require(features.ndim == 2 and np.isfinite(features).all(), 'Invalid probe representation')
    training = np.asarray(training, dtype=bool)
    require(training.any() and (~training).any(), 'Probe needs disjoint training and validation rows')
    scaler = StandardScaler().fit(features[training])
    require(scaler.n_samples_seen_ == training.sum(), 'Scaler used non-training rows')
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        classifier = LogisticRegression(**parameters)
        classifier.fit(scaler.transform(features[training]), labels[training])
        require(classifier.n_iter_.max() < parameters['max_iter'], 'Probe reached iteration cap; stop without tuning')
        probability = classifier.predict_proba(scaler.transform(features[~training]))[:, 1]
    require(np.isfinite(probability).all(), 'Nonfinite probe predictions')
    return probability, int(classifier.n_iter_.max()), scaler


def summarize(rows):
    return rows.groupby(['strategy', 'probe']).roc_auc.agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()


def paired_losses(frame, baseline):
    wide = frame.pivot(index=['strategy', 'repeat_id'], columns='probe', values='roc_auc').reset_index()
    reference = baseline[(baseline.model == 'logistic_regression') & (baseline.partition == 'validation') & (baseline.track == 'primary')]
    wide = wide.merge(reference[['strategy', 'repeat_id', 'roc_auc']], on=['strategy', 'repeat_id'], validate='one_to_one')
    require(len(wide) == 20, 'Incomplete paired probe matrix')
    wide = wide.rename(columns={'roc_auc': 'raw_concatenated_existing'})
    wide['global_branch_loss'] = wide.raw_global - wide.global_branch
    wide['local_branch_loss'] = wide.raw_local - wide.local_branch
    wide['joint_branch_loss'] = wide.raw_concatenated_existing - wide.concatenated_branch
    wide['fusion_loss'] = wide.concatenated_branch - wide.fusion_hidden
    wide['branch_minus_fusion_loss'] = wide.joint_branch_loss - wide.fusion_loss
    return wide


def table(frame):
    def cell(v):
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in frame.itertuples(index=False, name=None))


def finish(records, protocol):
    frame = pd.DataFrame([v for r in records for v in r['metrics']])
    require(len(frame) == 120 and not frame.duplicated(['strategy','repeat_id','probe']).any(), 'Probe matrix mismatch')
    summary = summarize(frame)
    paired = paired_losses(frame, pd.read_csv('data/stage4b/metrics.csv', float_precision='round_trip'))
    loss_columns = ['global_branch_loss','local_branch_loss','joint_branch_loss','fusion_loss','branch_minus_fusion_loss']
    long = paired.melt(id_vars=['strategy','repeat_id'], value_vars=loss_columns, var_name='comparison', value_name='difference')
    loss_summary = long.groupby(['strategy','comparison']).difference.agg(['count','mean','median','std','min','max']).reset_index()
    counts = long.assign(positive=long.difference>0, negative=long.difference<0, tied=long.difference==0).groupby(['strategy','comparison'])[['positive','negative','tied']].sum().reset_index()
    loss_summary = loss_summary.merge(counts,on=['strategy','comparison'],validate='one_to_one')
    groups = paired.groupby('strategy')
    # Descriptive directional agreement, not a significance test or information theorem.
    a = all(g.joint_branch_loss.mean()>0 and g.branch_minus_fusion_loss.mean()>0 and (g.branch_minus_fusion_loss>0).sum()>5 for _,g in groups)
    b = all(g.fusion_loss.mean()>0 and g.branch_minus_fusion_loss.mean()<0 and (g.branch_minus_fusion_loss<0).sum()>5 for _,g in groups)
    location = 'A: convolutional branches' if a else 'B: fusion/compression' if b else 'inconclusive or mixed'
    recommendation = ('Retain the stabilized LeakyReLU head and test one raw-flux skip connection into fusion, bypassing the convolutional branches while leaving the rest of the protocol frozen.' if a else
        'Retain the branches and test one wider fusion hidden layer, leaving the rest of the protocol frozen.' if b else
        'Do not choose an architecture change from these probes alone; the two locations are not cleanly distinguished.')
    for name,value in [('metrics',frame),('summary',summary),('paired_comparisons',paired),('loss_summary',loss_summary)]:
        write_csv(OUT/(name+'.csv'),value)
    verdict = dict(location=location,recommendation=recommendation,runs=20,linear_probe_fits=120,cnn_fits=0,
        evaluation='validation only; test rows excluded',
        branch_dominant_pairs=int((paired.branch_minus_fusion_loss>0).sum()),
        fusion_dominant_pairs=int((paired.branch_minus_fusion_loss<0).sum()),
        runtime_seconds=sum(r['runtime_seconds'] for r in records),protocol_sha256=sha256(OUT/'protocol.json'))
    atomic(OUT/'conclusion.json',verdict)
    text='# Stage 5C: short linear-probe localization\n\n'
    text+=f'**Conclusion: {location}.** This concerns validation-set linear accessibility under one fixed probe protocol, not loss of all possible information.\n\n'
    text+='Six probes only, across the same 2 strategies x 10 repeats: raw global (2,000), raw local (200), global Flatten (2,560), local Flatten (1,600), concatenated branches (4,160), and post-LeakyReLU fusion hidden (32). CNNs are frozen; inference disables dropout. Every scaler and logistic regression is fitted on its run\'s training rows only. No test rows participate.\n\n'
    text+='StandardScaler plus the existing Stage 4B L2 logistic-regression settings (C=1, lbfgs, tolerance 1e-4, max_iter=2,000, no class weighting); no tuning or extra probes. The already completed raw-concatenated Stage 4B logistic-regression validation score is reused as a like-for-like reference with identical seeds, training rows, scaling and regularization settings.\n\n'
    text+='## Six-probe validation ROC-AUC\n\n'+table(summary)+'\n'
    text+='## Paired loss localization\n\nPositive loss means lower ROC-AUC downstream. Branch loss is existing raw-concatenated LR minus concatenated-branch LR. Fusion loss is concatenated-branch LR minus hidden LR. Their difference compares the two losses without selecting the better raw branch after seeing results. Global/local losses are supporting checks. All 20 comparisons are retained.\n\n'
    text+=table(paired[['strategy','repeat_id']+loss_columns])+'\n'+table(loss_summary)+'\n'
    text+=f"Branch loss exceeds fusion loss in {verdict['branch_dominant_pairs']}/20 pairs; the reverse holds in {verdict['fusion_dominant_pairs']}/20. The predeclared descriptive rule requires the same mean direction and a majority of pairs within each strategy; it is not a statistical significance claim. Repeated splits overlap, so no independent-observation confidence interval is claimed.\n\n"
    text+='## One next experiment\n\n'+recommendation+' This is a recommendation only; no architecture experiment was started.\n\n'
    text+='## Limits and stop\n\nFrozen representations, different feature dimensions and fixed regularization can change linear accessibility without proving irreversible information destruction. Standardization removes marginal scale differences but does not make the hypothesis classes identical. Probes diagnose the learned endpoints and do not establish when a loss arose. Only validation results were used to support the next hypothesis; a future experiment needs a predeclared evaluation plan.\n\nNo Random Forest probes, PCA, correlation matrices, morphology analysis or secondary study. Stage 5C stops here.\n\n'
    text+='Evidence: [protocol](../data/stage5c/protocol.json), [all probe metrics](../data/stage5c/metrics.csv), [all paired comparisons](../data/stage5c/paired_comparisons.csv), [conclusion](../data/stage5c/conclusion.json). Per-run receipts and compressed validation predictions permit audit without rerunning fits.\n'
    report=Path('docs/kepler_stage5c_linear_probes.md'); report.write_text(text,encoding='utf-8')
    atomic(OUT/'report_integrity.json',dict(path=str(report),sha256=sha256(report)))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'checksums.sha256','status.json','runner.lock'}]
    (OUT/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')
    atomic(OUT/'status.json',dict(status='complete',completed_runs=20,probe_fits=120,pid=os.getpid()))
    print(json.dumps(verdict),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(OUT/'runner.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    config=read(stage5b.CONFIG); parameters=read('configs/kepler_stage4b_baselines_v1.json')['models']['logistic_regression']
    anchors={str(p):sha256(p) for p in [stage5b.CONFIG,stage5b.FREEZE,Path('data/stage5b/run_manifest.json'),Path('data/stage5b/runtime_checksums.sha256'),Path('data/stage4b/metrics.csv'),Path('configs/kepler_stage4b_baselines_v1.json')]}
    anchors.update(config['frozen_sha256'])
    protocol=dict(probes=PROBES,parameters=parameters,evaluation='validation only',threshold=.5,
        fitting='StandardScaler and logistic regression fitted on training rows only; no tuning',
        decision='Compare mean joint branch loss with mean fusion loss, requiring same direction and majority of pairs in each strategy. Otherwise stop inconclusive.',
        existing_raw_concatenated_reference='Stage 4B logistic_regression validation/primary; no new fit',
        anchors=anchors,source_sha256=sha256(__file__),seeds=config['seeds'])
    if (OUT/'protocol.json').exists(): require(read(OUT/'protocol.json')==protocol,'Protocol changed; stop')
    else: atomic(OUT/'protocol.json',protocol)
    verify_hashes(read(stage5b.FREEZE)['sha256'])
    context=sha256(stage5b.FREEZE)
    records=read(stage5b.META/'run_manifest.json')['runs']
    require(len(records)==20 and len({r['run_path'] for r in records})==20,'Incomplete parent run matrix')
    objects,_,assignments=read_input_tables(config)
    cache=Path(config['baseline_runtime'])/'features.npy'
    require(sha256(cache)==config['feature_cache_sha256'],'Input cache changed')
    require(objects.equals(pd.read_csv('data/stage4b/objects.csv')),'Input cache ordering differs')
    import tensorflow as tf
    from kepler_cnn.model import configure,weight_hash
    from kepler_cnn.runner import context as environment
    require(environment()==read('data/stage5/run_manifest.json')['runs'][0]['context'],'Pinned environment changed')
    configure(); inputs=views(np.load(cache,mmap_mode='r',allow_pickle=False)); all_records=[]
    atomic(OUT/'environment.json',dict(python=os.sys.executable,environment=environment()))
    for r in records:
        key=r['run_path']; directory=OUT/'runs'/key; receipt=directory/'complete.json'
        require(stage5b.verified_receipt(stage5b.ROOT/key,'training',context)==r,'Parent model receipt mismatch')
        if receipt.exists():
            record=read(receipt); require(record['protocol_sha256']==sha256(OUT/'protocol.json'),'Probe receipt context differs')
            verify_hashes(record['outputs']); all_records.append(record); continue
        start=time.perf_counter(); directory.mkdir(parents=True,exist_ok=True)
        assigned=align(objects,assignments,r['strategy'],r['repeat_id']); keep=assigned.partition.ne('test').to_numpy()
        train=assigned.loc[keep,'partition'].eq('train').to_numpy(); labels=objects.loc[keep,'label'].to_numpy()
        subset={k:v[keep] for k,v in inputs.items()}
        model=tf.keras.models.load_model(stage5b.ROOT/key/'model.keras',compile=False); before=weight_hash(model)
        require(before==r['final_weight_sha256'],'Loaded model weights differ')
        probe=tf.keras.Model(model.inputs,[model.get_layer(n).output for n in ['global_flatten','local_flatten','hidden_leaky_relu']])
        batches=[probe({k:v[i:i+64] for k,v in subset.items()},training=False) for i in range(0,len(labels),64)]
        representations=[np.concatenate([b[j].numpy() for b in batches]) for j in range(3)]
        features=[subset['global_flux'][:,:,0],subset['local_flux'][:,:,0],representations[0],representations[1],np.concatenate(representations[:2],axis=1),representations[2]]
        require(weight_hash(model)==before,'Extraction changed weights')
        del model,probe,batches; tf.keras.backend.clear_session()
        rows=[]; saved={}
        for name,x in zip(PROBES,features):
            probability,iterations,scaler=fit_probe(x,labels,train,parameters)
            rows.append(dict(strategy=r['strategy'],repeat_id=r['repeat_id'],seed=r['seed'],probe=name,
                features=x.shape[1],training_rows=int(train.sum()),iterations=iterations,**metrics(labels[~train],probability)))
            saved[name]=probability
        saved['kepoi_name']=objects.loc[keep].loc[~train,'kepoi_name'].to_numpy(dtype=str)
        saved['label']=labels[~train]
        np.savez_compressed(directory/'validation_predictions.npz',**saved)
        record=dict(run_path=key,metrics=rows,runtime_seconds=time.perf_counter()-start,protocol_sha256=sha256(OUT/'protocol.json'),
            parent_weight_sha256=before,outputs={str(directory/'validation_predictions.npz'):sha256(directory/'validation_predictions.npz')})
        atomic(receipt,record); all_records.append(record)
        atomic(OUT/'status.json',dict(status='running',completed_runs=len(all_records),probe_fits=6*len(all_records),pid=os.getpid(),current=key))
        print('Completed six probes: '+key,flush=True)
    verify_hashes(anchors); require(sha256(__file__)==protocol['source_sha256'],'Probe source changed')
    finish(all_records,protocol)
    lock.close()


if __name__=='__main__':main()
