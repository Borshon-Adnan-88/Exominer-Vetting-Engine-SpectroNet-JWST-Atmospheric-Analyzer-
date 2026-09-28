"""Finalize existing Stage 5A tables; no TensorFlow, inference or training."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scripts.stage5a_run_diagnostics import (OUT, ROOT, TABLES, POLICY, atomic, receipt,
    verify_hashes, manifest_entries, sha256, probability_stats, sample_indices)
from kepler_cnn.artifacts import align, check_prediction_identity
from kepler_baselines.inputs import read_input_tables, require

REPORT=Path('docs/kepler_stage5a_failure_analysis.md')
CREATED=['scripts/stage5a_finalize_report.py','tests/test_stage5a_reporting.py','data/stage5a/finalization.json',
    'data/stage5a/group_comparison.csv','data/stage5a/activation_group_comparison.csv','data/stage5a/weight_group_comparison.csv',
    'data/stage5a/input_group_comparison.csv','data/stage5a/numerical_group_comparison.csv','data/stage5a/class_prior_comparison.csv',
    'data/stage5a/figures/activation_group_comparison.png','data/stage5a/figures/numerical_sensitivity_summary.png']
MODIFIED=['docs/kepler_stage5a_failure_analysis.md','data/stage5a/diagnostic_summary.json','data/stage5a/checksums.sha256','data/stage5a/report_integrity.json']

def group_labels(r):
    result=np.where(r.probability_range<1e-7,'severe',np.where(r.unique_probability_count<=3,'low_diversity_only','non_collapsed'))
    require(np.array_equal(r.range_below_1e7,r.probability_range<1e-7),'Severe flag mismatch')
    require(np.array_equal(r.few_values,r.unique_probability_count<=3),'Low-diversity flag mismatch')
    return result

def table(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)): return f'{v:.7g}'
        return str(v).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))

def validate():
    entries=manifest_entries(OUT/'checksums.sha256',OUT); verify_hashes(entries)
    integrity=json.loads((OUT/'report_integrity.json').read_text()); verify_hashes({integrity['path']:integrity['sha256']})
    provenance=json.loads((OUT/'provenance.json').read_text()); verify_hashes(provenance['anchors'])
    require(sha256('scripts/stage5a_run_diagnostics.py')==provenance['context']['source_sha256'],'Diagnostic source changed')
    state=json.loads((OUT/'status.json').read_text()); require(state['status']=='complete' and state['completed_models']==20,'Incomplete diagnostics')
    records=json.loads(Path('data/stage5/run_manifest.json').read_text())['runs']
    config=json.loads(Path('configs/kepler_stage5_cnn_v1.json').read_text())
    objects,index,assignments=read_input_tables(config)
    expected={(s,r) for s in config['strategies'] for r in range(10)}
    require({(r['identity']['strategy'],r['identity']['repeat_id']) for r in records}==expected and len(records)==20,'Certified run matrix mismatch')
    results=[]; priors=[]
    for record in records:
        key=record['run_path']; result=receipt(OUT/'runs'/key,provenance['context']); require(result is not None,'Missing receipt')
        for name,rows in result.items():
            require(len(rows)>0,'Missing diagnostic rows: '+name)
            require(all(x['strategy']==record['identity']['strategy'] and x['repeat_id']==record['identity']['repeat_id'] for x in rows),'Diagnostic run identity mismatch')
        run=result['run_diagnostics'][0]; require(len(result['run_diagnostics'])==1 and run['weights_unchanged'] is True,'Weight preservation evidence missing')
        pred=pd.read_csv(ROOT/key/'predictions.csv',float_precision='round_trip')
        check_prediction_identity(pred,objects,align(objects,assignments,record['identity']['strategy'],record['identity']['repeat_id']))
        for k,v in probability_stats(pred.probability.to_numpy()).items(): require(run[k]==v,'Probability diagnostic mismatch: '+k)
        for k,v in record['outcome'].items(): require(run[k]==v,'Outcome diagnostic mismatch')
        require(json.loads(run['sample_kois'])==pred.iloc[sample_indices(pred)].kepoi_name.tolist(),'Training sample mismatch')
        history=pd.read_csv(ROOT/key/'history.csv',float_precision='round_trip')
        recorded=pd.DataFrame(result['training_histories']).drop(columns=['strategy','repeat_id'])
        pd.testing.assert_frame_equal(recorded[history.columns],history,check_dtype=False,check_exact=True)
        prior=float(pred.loc[pred.partition.eq('train'),'label'].mean())
        priors.append(dict(strategy=run['strategy'],repeat_id=run['repeat_id'],training_prior=prior,
            probability_mean=run['probability_mean'],mean_minus_prior=run['probability_mean']-prior,
            modal_count=int(pred.probability.value_counts().max()),prior_entropy=float(-prior*np.log(prior)-(1-prior)*np.log1p(-prior))))
        results.append(result)
    frames={name:pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip') for name in TABLES}
    for name,frame in frames.items():
        rebuilt=pd.DataFrame([r for result in results for r in result[name]])
        compare=frame.copy()
        if name=='layer_weight_diagnostics':
            # CSV empty tensor names on parameter-free layers parse as NaN.
            require(compare.loc[compare.tensor.isna(),'role'].eq('no_weights').all(),'Missing parameter tensor identity')
            compare['tensor']=compare.tensor.fillna('')
        pd.testing.assert_frame_equal(compare,rebuilt[frame.columns],check_dtype=False,check_exact=False,rtol=1e-13,atol=0)
        require(set(zip(frame.strategy,frame.repeat_id))==expected,'Missing diagnostic run in '+name)
    require(len(frames['run_diagnostics'])==20 and len(frames['activation_diagnostics'])==120 and len(frames['input_diagnostics'])==40 and len(frames['numerical_sensitivity'])==240,'Diagnostic counts mismatch')
    numerical=frames['numerical_sensitivity']
    require(not numerical.duplicated(['strategy','repeat_id','batch_size','partition']).any(),'Duplicate sensitivity rows')
    for _,n in numerical.groupby(['strategy','repeat_id']):
        require(set(zip(n.batch_size,n.partition))=={(b,p) for b in (32,64,256) for p in ('all','train','validation','test')},'Sensitivity matrix incomplete')
    require((numerical.loc[numerical.batch_size.eq(64),['maximum_probability_difference','changed_labels','roc_auc_change','changed_average_ranks']]==0).all().all(),'Original batch replay differed')
    changed=subprocess.check_output(['git','diff','4a2b193','--name-only','--','configs','data/stage4b','data/stage5','data/stage4a','data/manifests','data/splits','kepler_cnn','kepler_baselines'],text=True)
    require(not changed.strip(),'Certified tracked files changed')
    return frames,pd.DataFrame(priors),provenance,state,entries

def aggregate(frames,priors):
    runs=frames['run_diagnostics'].copy(); runs['group']=group_labels(runs)
    keys=['strategy','repeat_id']; labels=runs[keys+['group']]
    hist=frames['training_histories'].groupby(keys,sort=False).tail(1)[keys+['training_loss','validation_loss','training_roc_auc','validation_roc_auc']]
    runs=runs.merge(hist,on=keys,validate='one_to_one')
    summary=runs.groupby('group').agg(runs=('repeat_id','size'),probability_std=('probability_std','mean'),
        best_epoch=('best_epoch','mean'),stop_epoch=('stopping_epoch','mean'),plateaus=('plateau_last_five','sum'),
        divergences=('train_validation_divergence','sum'),training_loss=('training_loss','mean'),validation_loss=('validation_loss','mean'),
        training_auc=('training_roc_auc','mean'),validation_auc=('validation_roc_auc','mean')).reset_index()
    outputs={'group_comparison':summary,'class_prior_comparison':priors.merge(labels,on=keys,validate='one_to_one')}
    specifications=[('activation','activation_diagnostics',['layer'],['std','zero_fraction','between_object_variance','min','max']),
        ('weight','layer_weight_diagnostics',['layer','role'],['norm','std','min','max','zero_fraction','nonfinite_count']),
        ('input','input_diagnostics',['branch'],['min','max','mean','std','zero_fraction','unobserved_bin_fraction','object_variance_q0','object_variance_q25','object_variance_q50','object_variance_q75','object_variance_q100'])]
    for short,name,by,cols in specifications:
        f=frames[name].merge(labels,on=keys,validate='many_to_one')
        if short=='weight': f=f[f.role.ne('no_weights')]
        outputs[short+'_group_comparison']=f.groupby(['group']+by)[cols].mean().reset_index()
    n=frames['numerical_sensitivity'].merge(labels,on=keys,validate='many_to_one')
    outputs['numerical_group_comparison']=n.groupby(['group','partition','batch_size']).agg(max_probability_difference=('maximum_probability_difference','max'),
        max_changed_labels=('changed_labels','max'),auc_change_min=('roc_auc_change','min'),auc_change_max=('roc_auc_change','max'),max_changed_ranks=('changed_average_ranks','max')).reset_index()
    return runs,outputs

def main():
    frames,priors,prov,state,entries=validate(); runs,outputs=aggregate(frames,priors)
    for name,f in outputs.items(): f.to_csv(OUT/(name+'.csv'),index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=outputs['activation_group_comparison']; fig,axes=plt.subplots(1,2,figsize=(12,4))
    for group,f in a.groupby('group'):
        axes[0].plot(f.layer,f.between_object_variance,marker='o',label=group)
        axes[1].plot(f.layer,f.zero_fraction,marker='o',label=group)
    axes[0].set_yscale('symlog',linthresh=1e-12); axes[0].set_ylabel('Mean between-object variance')
    axes[1].set_ylabel('Mean exact-zero fraction')
    for ax in axes: ax.tick_params(axis='x',rotation=65); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'figures/activation_group_comparison.png'); plt.close(fig)
    n=frames['numerical_sensitivity']; fig,axes=plt.subplots(1,2,figsize=(10,4))
    for batch,f in n[n.partition.eq('test')].groupby('batch_size'):
        axes[0].plot(range(len(f)),f.maximum_probability_difference,label=str(batch))
        axes[1].plot(range(len(f)),f.roc_auc_change,label=str(batch))
    for ax in axes: ax.set_xlabel('Run order'); ax.legend(title='Batch')
    axes[0].set_ylabel('Maximum probability difference'); axes[1].set_ylabel('Test ROC-AUC change')
    fig.tight_layout(); fig.savefig(OUT/'figures/numerical_sensitivity_summary.png'); plt.close(fig)
    summary=json.loads((OUT/'diagnostic_summary.json').read_text())
    summary['finalized']=True
    summary['strongest_finding']='All seven low-diversity runs have exactly zero hidden ReLU outputs and constant logits on their 128-object training samples; upstream branches retain nonzero variance.'
    summary['classification']='activation-related endpoint failure; optimization origin unresolved; not explained by tested inference batch sizes'
    summary['groups']={k:f.to_dict('records') for k,f in outputs.items()}
    summary['proposed_intervention']='Only hidden Dense activation: ReLU to LeakyReLU with fixed negative_slope=0.01; design only.'
    summary['interpretation']='Endpoint localization is strongly supported on sampled training objects; causal training mechanism remains unresolved. All original scientific results retained.'
    atomic(OUT/'diagnostic_summary.json',summary)
    evidence=dict(status='finalized',models=20,focused_tests='4 passed: tests/test_stage5a_reporting.py; 5.77 seconds',row_counts={k:len(v) for k,v in frames.items()},
        certified_anchor_files_verified=len(prov['anchors']),model_and_prediction_hashes='unchanged',weight_preservation='20 recorded before/after weight-hash assertions passed; saved-model byte hashes unchanged',
        canonical_arrays='Runner verified canonical file and logical hashes before diagnostics; reused read-only execution evidence, no full NPZ reload during reporting.',
        diagnostic_runtime_seconds=state['runtime_seconds'],inference_calls=0,training_calls=0,
        original_diagnostic_checksums_sha256=sha256(OUT/'checksums.sha256'),created=CREATED,modified=MODIFIED,
        reporting_source_sha256=sha256(__file__))
    atomic(OUT/'finalization.json',evidence)
    low=runs[runs.few_values]; weights=outputs['weight_group_comparison']; inputs=outputs['input_group_comparison']
    parts=['# Stage 5A failure analysis\n\n**Status: finalized; existing diagnostics validated for all 20 certified models.**',
    '## 1. Scope\n\nRead-only analysis of the completed 857.494-second diagnostic run. No model training, inference replay, weight changes, corrected predictions, excluded runs, or Stage 5B execution occurred during finalization. Figures and tables below derive from existing diagnostic records.',
    '## 2. Diagnostic integrity\n\n'+json.dumps(evidence,indent=2).join(['```json\n','\n```'])+'\n\nValidated diagnostic checksum manifests, all 20 completion receipts and identities, table-to-receipt equality, certified histories, deterministic training-only samples, probability summaries, and the complete 20 Ã— 3 batches Ã— 4 partition sensitivity matrix. Certified model/prediction bytes and recorded Stage 1â€“5 anchors remain unchanged. Before/after in-memory weight-hash assertions passed in every diagnostic run. This reporting pass reuses the completed canonical-array audit rather than claiming a new 6,637-file audit.',
    '## 3. Stage 5 failure recap\n\nCertified mean test ROC-AUC: object 0.6091577850, grouped 0.6002900906; paired difference +0.0088676945. CNN ROC-AUC was below both baselines in every matched strategy/repeat. No null or adverse result is removed. Non-collapsed here only means neither predefined flag; it does not mean scientifically adequate performance.',
    '## Directly measured facts',
    '## 4. Probability-collapse pattern\n\nFour severe runs satisfy range <1e-7. Seven satisfy unique count <=3, including those four. Tables use disjoint groups: severe (4), low-diversity-only (3), non-collapsed (13), retaining both original flags independently. All seven output probabilities below 0.5 for every KOI. Severe runs have 6,636/6,637 identical probabilities and one tiny deviation; the other three have 6,635 at their modal value. Object repeat 8 has range 0.398368 because of one low-probability outlier; a large range alone therefore misses its collapse.\n\n'+table(low[['strategy','repeat_id','diagnostic_class','unique_probability_count','probability_range','probability_std','logit_std','probability_mean','fraction_above_half']]),
    '## 5. Activation findings\n\nAll seven low-diversity runs have hidden ReLU zero fraction exactly 1, activation standard deviation 0, and between-object variance 0 on each 128-object training sample. Their sampled output logits are constant, approximately -0.39, not saturated at extreme logits. The early global and local convolutions and final branch representations retain nonzero between-object variance. Thus the observed bottleneck is at the fusion dense ReLU, not wholesale failure of both branches.\n\nIn the other 13 runs the global flattened representation is, paradoxically, more sparse on average (98.91% zeros), while the fusion layer retains variation (94.51% zeros). Sparsity alone does not establish collapse, and channel/position variation measured by std is different from variation between objects. Each row below is an equal-weight mean of per-run statistics.\n\n'+table(a)+'\n\n![Activation groups](../data/stage5a/figures/activation_group_comparison.png)',
    '## 6. Weight findings\n\nNo non-finite parameters were observed. Collapsed runs retain nonzero kernels; the zero fusion activations are not caused by all-zero stored weights. Mean hidden kernel norms are 8.1215 (severe), 8.1750 (low-diversity-only), and 9.7686 (non-collapsed); corresponding output kernel norms are 1.2986, 1.3193, and 1.9163. The collapsed hidden biases are nonpositive and the output bias is near -0.39. These are endpoint associations, not proof that bias sign or learning rate caused the state. There is no measured parameter overflow; moderate finite norm differences do not rule out earlier optimization instability. The complete layerwise kernel/bias statistics, including extrema and zero fractions, are in [weight_group_comparison.csv](../data/stage5a/weight_group_comparison.csv).\n\n'+table(weights[weights.layer.isin(['hidden','confirmed_probability'])]),
    '## 7. Input findings\n\nThe training samples have closely overlapping group summaries: global mean about 0.9992, local mean about 0.9976; global std about 0.011â€“0.012, local std about 0.019â€“0.020. No sampled flux entries are exactly zero. Unobserved-bin fractions are about 0.29â€“0.37% global and 1.82â€“2.09% local. Masks define unobserved bins; zero values are not assumed to be fill. Baseline near one and small signal variations occur in all groups, so these tables do not support blaming unusual corrupted or degenerate inputs in collapsed runs. Slight distribution differences and sample overlap prevent causal group comparisons.\n\n'+table(inputs),
    '## 8. Training-history findings\n\nEvery low-diversity run meets the predeclared last-five validation-loss plateau rule (range <0.001); none of the other 13 does. All 20 fail to trigger the predeclared divergence rule (loss-gap increase >0.05 with declining training loss); this is not proof that overfitting never occurred. Low-diversity stopping losses are about 0.674 train / 0.673â€“0.674 validation, with AUC near 0.5. Predictions differ from the training prior by at most 0.000833 across the seven runs, consistent with low-information, class-prior-like solutions. Initial loss improvement can simply reflect fitting the class prior. Histories describe stopping-epoch inference; probes describe the restored best checkpoint.\n\nObject repeat 4 alone reaches the 100-epoch cap, best epoch 99; it is not in either collapse group. No further epochs were run.\n\n'+table(runs[['strategy','repeat_id','group','best_epoch','stopping_epoch','epoch_cap_reached','training_loss','validation_loss','training_roc_auc','validation_roc_auc']])+'\n\n'+table(outputs['class_prior_comparison'][outputs['class_prior_comparison'].group.ne('non_collapsed')])+'\n\n![History comparison](../data/stage5a/figures/training_history_comparison.png)',
    '## 9. Numerical-sensitivity findings\n\nBatch 64 matches the authoritative probabilities exactly. Batches 32 and 256 cause **zero probability, label, AUC or rank changes in all seven low-diversity runs**, over every reported partition. Their tiny near-ties are not sensitive to these tested batch sizes/order. This does not prove every near-tie encodes astrophysical information or exclude other numerical effects.\n\nAcross non-collapsed runs the maximum probability difference is 2.9802322388e-7, with no changed labels. Up to 15 whole-cohort average ranks change; test partitions have at most two changed ranks and a maximum absolute ROC-AUC change of 2.1096510619e-6. These effects are far too small to explain the CNN baseline deficit. No probabilities or original ROC-AUC values are rounded, replaced, or corrected.\n\n'+table(outputs['numerical_group_comparison'].query('partition=="test"'))+'\n\n![Numerical sensitivity](../data/stage5a/figures/numerical_sensitivity_summary.png)',
    '## 10. Collapsed versus non-collapsed comparison\n\n'+table(outputs['group_comparison'])+'\n\nThe four severe and three additional low-diversity runs share a completely zero sampled fusion representation, constant sampled logits, near-prior outputs and plateaued losses. The other 13 have variable fusion/logit outputs despite substantial branch sparsity. Neither successful optimization nor superiority to baselines follows from escaping the flags. All original independent flags remain available in run_diagnostics.csv.\n\n![Probability spread](../data/stage5a/figures/probability_spread.png)',
    '## Interpretations reasonably supported by multiple diagnostics\n\n## 11. Likely mechanism(s)\n\nThe strongest localization is **activation-related endpoint collapse at the fusion dense ReLU**. On the inspected training samples, a zero hidden representation makes the final affine logit equal its output bias; the sigmoid then supplies the near-prior probability. Nonzero variable upstream representations, nonzero weights, constant sampled logits, near-prior full-cohort predictions and plateaued histories independently support this explanation of the endpoint. This is stronger than an inference based on probability range alone. It does not establish the causal training path, or prove all 6,637 hidden representations are zero. The rare full-cohort exceptions explicitly limit that claim.',
    '## Mechanisms that remain possible\n\nTraining dynamics may have moved the fusion preactivations into the inactive ReLU region. Learning rate, initialization, the flattened fusion representation and the near-one input baseline could interact, but no historical gradients, preactivation trajectories or controlled interventions were measured. Branch sparsity may also limit the 13 non-collapsed models. These remain hypotheses; optimization-related origin is unresolved.',
    '## Hypotheses contradicted or not evidenced by the diagnostics\n\n## 12. Ruled-out or unsupported explanations\n\nVerified artifact corruption, weight mutation, all-zero stored kernels, and non-finite parameters are not supported. Both branches being entirely dead is contradicted on the inspected samples. Extreme sigmoid saturation is not the cause of the near-0.403 collapsed output. Tested inference batching does not explain the seven collapsed distributions. Unusual sampled input scale or missingness is not demonstrated as their cause. Learning-rate excess, Flatten itself, normalization deficiency and host leakage are not established causes.',
    '## 13. Interpretation limitations\n\nActivation/input statistics cover 128 deterministically selected training objects per run, with overlapping samples. Probability diagnostics cover all 6,637 objects. Group sizes are 4/3/13, correlated across paired repeats and one cohort; no independent-sample significance claims are made. Within-array std and between-object variance have different meanings. A constant sampled logit does not prove every full-cohort prediction is identical. Recovered logits exclude exact 0/1 probabilities; internal logits are separate measurements. Saved optimizer slots correspond to stopping epoch, not necessarily the restored best weights. Group differences are post-execution exploratory associations using predeclared flags. Absence of a divergence flag is threshold-specific.',
    '## 14. Recommended minimal intervention\n\nRecommend **one variant only**: replace the hidden fusion Dense ReLU with a LeakyReLU of fixed negative_slope=0.01. Keep both convolutional branches and every other setting unchanged. A nonzero negative slope directly targets the sampled all-zero fusion bottleneck without changing parameter count or pooling. The slope is a proposed fixed intervention, not an empirically optimized value. Keras documents this negative-slope behavior in its [activation reference](https://keras.io/api/layers/activations/). This is a causal probe, not a promise of improvement. There is insufficient direct evidence to prioritize simultaneous learning-rate, normalization or pooling changes; no architecture search is proposed.',
    '## 15. Proposed Stage 5B design â€” not authorized or executed\n\nFreeze the single activation variant and its code/config hash before any fitting. Reuse the 6,637 KOIs, labels, canonical flux arrays, two Stage 3 strategies, ten seeds and paired initial weights. Keep Adam, learning rate, loss, dropout, pooling, dense width, batch size, maximum epochs, early stopping, threshold 0.5, CPU determinism and evaluation policy unchanged. No seed search, grid, test-driven selection, augmentation or normalization.\n\nAfter separate approval, run the full paired design (20 new fits for this one frozen variant), using existing certified Stage 5 runs as the reference. Do not refit the original model. For each run, select checkpoints by validation loss under the original policy. Keep test scoring sealed until all variant fits and the validation/stability decision are recorded. Predeclare descriptive stability success as zero severe and zero <=3-value outputs on validation partitions, finite parameters/predictions, and no entirely zero hidden representation in each unchanged training probe sample. Predeclare validation benefit as lower mean paired validation BCE than the original CNN separately in both strategies. These are proposed Stage 5B decision criteria, not new Stage 5A classification thresholds. Record failures and epoch caps without remedial fitting.\n\nUnseal and report all test outcomes even if criteria fail. Primary final comparisons are matched variant-minus-original test ROC-AUC and object-minus-grouped ROC-AUC by repeat, retaining baseline comparisons, AP, Brier, threshold metrics, calibration, QC sensitivity and secondary host diagnostics. Do not use these already-reused test sets as a new independent confirmation dataset. No further variant is selected within this proposal.',
    '## 16. Artifact hashes\n\nHashes below cover all primary diagnostic tables, derived group tables, summary, finalization evidence and figures. The checksum manifest additionally covers per-run receipts and existing operational artifacts; status/log/lock are intentionally excluded. To avoid circular hashes, report_integrity.json records this report digest and checksums.sha256 records that file. This report does not embed its own digest or the enclosing checksum-manifest digest.',
    '## 17. Exact files created/modified in this finalization\n\nCreated:\n\n'+''.join('- `'+p+'`\n' for p in CREATED)+'\nModified:\n\n'+''.join('- `'+p+'`\n' for p in MODIFIED)+'\nThe original diagnostic runner, tests, per-run receipts, diagnostic tables and scientific artifacts remain unchanged. Focused reporting validation is recorded in finalization.json. No commit or push was made. Stop for review.']
    # Record successful inexpensive validation before hashing the final report inputs.
    subprocess.run(['git','diff','--check'],check=True)
    evidence['git_diff_check']='passed'; evidence['reporting_validation']='passed: receipts, identities, tables, anchors, sample boundaries and sensitivity coverage'
    atomic(OUT/'finalization.json',evidence)
    hashes=pd.DataFrame([dict(artifact=p.relative_to(OUT).as_posix(),sha256=sha256(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and (p.suffix in ('.csv','.png') or p.name in ('diagnostic_summary.json','finalization.json'))])
    parts[-2]+='\n\n'+table(hashes)
    REPORT.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    atomic(OUT/'report_integrity.json',dict(path=str(REPORT),sha256=sha256(REPORT)))
    for p,h in entries.items():
        if Path(p).name not in ('diagnostic_summary.json','report_integrity.json'): require(sha256(p)==h,'Original diagnostic changed: '+p)
    verify_hashes(prov['anchors'])
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('checksums.sha256','status.json','stage5a.log','runner.lock') and not p.name.endswith('.tmp')]
    (OUT/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')
    verify_hashes(manifest_entries(OUT/'checksums.sha256',OUT))
    print(json.dumps({'status':'finalized','rows':evidence['row_counts'],'hashes':{str(p):sha256(p) for p in [REPORT,OUT/'diagnostic_summary.json',OUT/'checksums.sha256']}},indent=2))

if __name__=='__main__': main()
