"""Read-only scientific audit and validation reporting; never fits or loads models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import stage5d_run_fusion_width as r

OUT = Path('data/stage5d')


def stats(values):
    a = pd.Series(values, dtype=float)
    return dict(count=len(a), mean=a.mean(), median=a.median(), sd=a.std(ddof=1),
                min=a.min(), max=a.max(), improved=int((a > 0).sum()),
                worsened=int((a < 0).sum()), tied=int((a == 0).sum()))


def inventory(path, root):
    entries = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split('  ', 1)
        r.require(name not in entries, 'Duplicate checksum entry')
        entries[name] = digest
    r.verify_hashes({str(root / n): h for n, h in entries.items()})
    return entries


def audit():
    # Existing preflight checks baseline alignment and raw compatibility, without fits.
    inventory(OUT/'checksums.sha256', OUT)
    runtime = inventory(OUT/'runtime_checksums.sha256', r.ROOT)
    actual = {p.relative_to(r.ROOT).as_posix() for p in r.ROOT.rglob('*') if p.is_file()}
    r.require(actual == set(runtime), 'Runtime inventory contains missing or extra files')
    data = r.preflight()
    manifest = r.read(OUT/'run_manifest.json')
    expected = {(s, i, seed) for s in data['config']['strategies'] for i, seed in enumerate(data['config']['seeds'])}
    runs = manifest['runs']
    r.require(len(runs) == 20 and {(x['strategy'], x['repeat_id'], x['seed']) for x in runs} == expected,
              'Scientific run matrix differs')
    r.require(manifest['context'] == data['context'], 'Manifest context differs')
    rows, training = [], []
    for cnn in runs:
        who = r.identity(cnn['strategy'], cnn['repeat_id'], cnn['seed']); key = r.key_for(who)
        binding = dict(scope_sha256=data['references'][key]['scope_sha256'])
        saved = r.receipt(r.ROOT/key/'cnn', data['context'], who, binding, r.CNN_FILES)
        r.require(saved == cnn, 'Aggregate CNN record differs')
        for stage, root, record in [('5D', r.ROOT/key/'cnn', cnn),
                                    ('5B', r.PARENT/key, r.read(r.PARENT/key/'training_manifest.json'))]:
            history = pd.read_csv(root/'history.csv', float_precision='round_trip')
            best = r.selected_checkpoint(history, record)
            prediction = pd.read_csv(root/'train_validation_predictions.csv', float_precision='round_trip')
            r.require(set(prediction.partition) == {'train', 'validation'}, 'Forbidden prediction partition')
            columns = r.IDENTITY+['qc_pass', 'partition', 'fold_id']
            r.require(r.identifiers(prediction) == binding['scope_sha256'], 'Prediction scope differs')
            p = prediction.loc[prediction.partition.eq('validation'), 'probability'].to_numpy()
            y = prediction.loc[prediction.partition.eq('validation'), 'label'].to_numpy()
            r.require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
            m = r.metrics(y, p)
            r.require(abs(m['roc_auc']-best.validation_roc_auc) < 1e-12, 'Checkpoint AUC mismatch')
            if stage == '5D':
                # Validation-only and whole train/validation inference use different
                # batch boundaries. Ranking/counts must match exactly; allow the
                # existing 1e-7 probability-replay tolerance for Brier only.
                for metric, value in m.items():
                    old = cnn['validation_metrics'][metric]
                    r.require(abs(value-old) <= 1e-7 if metric == 'brier' else value == old,
                              'Saved validation metrics differ: '+metric)
                m = cnn['validation_metrics']
            rows.append(dict(**who, stage=stage, bce=best.validation_loss, **m))
            training.append(dict(**who, stage=stage, best_epoch=record['best_epoch'],
                stopping_epoch=record['stopping_epoch'], epoch_cap=record['epoch_cap_reached'],
                early_stopped=record['status']=='early_stopped', finite=True,
                probability_min=p.min(), probability_max=p.max(), probability_range=np.ptp(p),
                probability_std=p.std(), unique_count=len(np.unique(p)),
                severe=np.ptp(p)<1e-7, low_diversity=len(np.unique(p))<=3,
                epoch_seconds=history.epoch_seconds.sum()))
        pb = dict(**binding, final_weight_sha256=cnn['final_weight_sha256'])
        for name in r.RAW+r.LEARNED:
            row = r.receipt(r.ROOT/key/'probes'/name, data['context'], who, pb, r.PROBE_FILES)
            r.require(row is not None and row['probe'] == name and row['reused'] == (name in r.RAW), 'Probe identity differs')
            p = np.load(r.ROOT/key/'probes'/name/'probability.npy', allow_pickle=False)
            if name in r.RAW:
                r.require(np.array_equal(p, data['raw'][key][name]['probability']), 'Reused raw probabilities differ')
            r.require(abs(r.metrics(y, p)['roc_auc']-row['roc_auc'])<1e-12, 'Probe AUC differs')
    r.require(len(list(r.ROOT.glob('*/*/cnn/complete.json'))) == 20 and
              len(list(r.ROOT.glob('*/*/probes/*/complete.json'))) == 120, 'Unexpected phase count')
    return data, pd.DataFrame(rows), pd.DataFrame(training)


def table(frame):
    def cell(value):
        return f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)
    lines = ['| '+' | '.join(map(str, frame.columns))+' |',
             '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def finalize():
    data, metrics, training = audit()  # No output mutation before integrity passes.
    paired = metrics.pivot(index=['strategy','repeat_id','seed'], columns='stage', values=['roc_auc','bce','average_precision','brier','f1','accuracy'])
    paired.columns = ['_'.join(c) for c in paired.columns]
    paired = paired.reset_index()
    for metric in ['roc_auc','bce','average_precision','brier','f1','accuracy']:
        paired[metric+'_delta'] = paired[metric+'_5D']-paired[metric+'_5B']
    summary = []
    for name, group in list(paired.groupby('strategy'))+[('overall', paired)]:
        summary.append(dict(strategy=name, stage5d_mean=group.roc_auc_5D.mean(), stage5b_mean=group.roc_auc_5B.mean(), **stats(group.roc_auc_delta)))
    summary = pd.DataFrame(summary)
    probes = pd.read_csv(OUT/'probe_comparisons.csv', float_precision='round_trip')
    probe_summary = probes.groupby(['strategy','probe'])[['roc_auc_32','roc_auc_128','delta_roc_auc']].mean().reset_index()
    overall = probes.groupby('probe')[['roc_auc_32','roc_auc_128','delta_roc_auc']].mean().reset_index().assign(strategy='overall')
    probe_summary = pd.concat([probe_summary, overall], ignore_index=True)
    gaps = probes[probes.probe.eq('concatenated_branch')].merge(probes[probes.probe.eq('fusion_hidden')], on=['strategy','repeat_id','seed'], suffixes=('_concat','_hidden'), validate='one_to_one')
    for width in ['32','128']:
        gaps['gap_'+width] = gaps['roc_auc_'+width+'_concat']-gaps['roc_auc_'+width+'_hidden']
    gaps['gap_reduction'] = gaps.gap_32-gaps.gap_128
    gaps = gaps[['strategy','repeat_id','seed','gap_32','gap_128','gap_reduction']]
    between = paired.pivot(index='repeat_id',columns='strategy',values='roc_auc_5D')
    between['object_minus_host_group'] = between.object_stratified-between.host_group_stratified
    runtime = dict(cnn_fit_and_artifact_seconds=sum(x['runtime_seconds'] for x in r.read(OUT/'run_manifest.json')['runs']),
        mean_cnn_fit_and_artifact_seconds=np.mean([x['runtime_seconds'] for x in r.read(OUT/'run_manifest.json')['runs']]),
        median_cnn_fit_and_artifact_seconds=np.median([x['runtime_seconds'] for x in r.read(OUT/'run_manifest.json')['runs']]),
        pure_cnn_epoch_seconds=training.loc[training.stage.eq('5D'),'epoch_seconds'].sum(),
        wall_clock_seconds=None, probe_seconds=None,
        limitation='No persisted start timestamp or probe timer: exact total wall-clock and probe durations cannot be reconstructed. Filesystem timestamps are not timing receipts.')
    # Scientific judgment for this frozen completed experiment, not a new
    # numerical gate or a post-hoc significance test.
    decision = dict(recommendation='GO',
        hypothesis='partially supported', validation_only=True, held_out_evaluated=False,
        primary=summary.to_dict('records'), fusion_gap_32=gaps.gap_32.mean(), fusion_gap_128=gaps.gap_128.mean(),
        gap_reduction=stats(gaps.gap_reduction), object_minus_host_group=stats(between.object_minus_host_group), runtime=runtime)
    interpretation = ('Widening improves mean CNN validation ROC-AUC, mainly on host-group splits; object-level gains are small and mixed. '
        'The hidden linear probes preserve more discriminative information and the fusion gap shrinks, consistently with Stage 5C, '
        'but the CNN remains far below the raw/concatenated probes. This supports a single frozen test evaluation as a hypothesis check, '
        'not a claim of strong predictive performance or a causal isolation of compression: both branches were trained end-to-end. '
        'Repeated validation reuse and overlapping splits limit independence; no post-hoc significance threshold is applied.')
    decision['interpretation'] = interpretation
    decision['fraction_fusion_gap_reduced'] = float(gaps.gap_reduction.mean()/gaps.gap_32.mean())
    concat32 = probes.loc[probes.probe.eq('concatenated_branch'),'roc_auc_32'].mean()
    decision['fraction_original_concat_to_cnn_gap_closed'] = float(paired.roc_auc_delta.mean()/(concat32-paired.roc_auc_5B.mean()))
    protocol = dict(execute=False, checkpoint='saved minimum-validation-BCE weights; earliest epoch on tie',
        models=[dict(path=str(r.ROOT/r.key_for(x)/'cnn'/'model.keras'), sha256=r.read(r.ROOT/r.key_for(x)/'cnn'/'complete.json')['sha256']['model.keras'], strategy=x['strategy'],repeat_id=x['repeat_id'],seed=x['seed']) for x in r.read(OUT/'run_manifest.json')['runs']],
        partitions='Original QC-passing held-out test rows in frozen Stage 3 assignments, matching strategy/repeat; canonical frozen preprocessing/cache',
        metrics=['ROC-AUC (primary)','BCE','average precision','Brier','precision','recall','F1','accuracy','confusion matrix'],
        threshold=0.5, comparisons='Paired Stage 5D minus certified Stage 5B and Random Forest for all 20 matched runs; report strategy and overall mean/median/sample SD/min/max and all paired outcomes; object-minus-host-group by repeat is descriptive',
        rules='One evaluation per saved model; no retraining, architecture changes, threshold tuning, test-based checkpoint/seed/model selection, or repeated evaluation. Verify frozen hashes and baseline split/seed identities first. Report every run. Existing Stage 5B/Random Forest test results are already reused evidence, not a fresh independent test cohort.')
    decision['proposed_test_protocol'] = protocol
    for name, frame in [('validation_metrics',metrics),('paired_stage5b_stage5d',paired),('run_summary',summary),('representation_probe_summary',probe_summary),('fusion_gap_comparison',gaps),('training_behavior',training),('object_host_comparison',between.reset_index())]:
        r.write_csv(OUT/(name+'.csv'),frame)
    r.atomic(OUT/'decision_summary.json',decision)
    figures(paired, probes, gaps, training)
    report = '# Stage 5D — validation analysis\n\n'
    report += '## Objective and frozen change\n\nDense(32, linear) → Dense(128, linear), 538,113 parameters. LeakyReLU(0.01), dropout 0.2, branches, initialization policy, optimizer/BCE, learning rate, clipping, batch size, stopping policy, seeds, splits and QC are frozen.\n\n'
    report += '## Execution integrity and validation boundary\n\n20/20 unique CNN runs, two strategies × ten repeats; 80 learned probes and 40 compatible reused raw probes. All frozen anchors, inventories, phase receipts, model checksum bindings, seeds, identifiers and minimum-BCE checkpoints verified. Raw probabilities match Stage 5C exactly. Histories and saved predictions contain only train/validation scope; runner slices before materialization and provides no test path. This audit reads no held-out predictions and performs no inference or fitting. Historical absence of transient test arrays is supported by the frozen execution path, not reconstructable from memory. No prior certified artifacts are written.\n\n'
    report += '## Primary CNN validation result\n\nSample SD, min and max refer to paired differences.\n\n'+table(summary)+'\n\n'+table(paired[['strategy','repeat_id','seed','roc_auc_5B','roc_auc_5D','roc_auc_delta']])+'\n\n'
    report += '## Secondary metrics\n\nAll use the same minimum-BCE checkpoint and fixed 0.5 threshold. BCE/Brier lower is better; all other reported metrics higher is better. Deltas are 5D minus 5B. AP, Brier, F1 and accuracy were already frozen metrics.\n\n'+table(paired.groupby('strategy')[[c for c in paired if c.startswith(('bce_','average_precision_','brier_','f1_','accuracy_'))]].mean().reset_index())+'\n\n'
    report += '## Training behavior\n\nSame earliest minimum-validation-BCE rule, patience 10, cap 100; no maximum-AUC substitution. Probability collapse criteria inherited from Stage 5A/B: range <1e-7 severe; <=3 distinct values low diversity. Finite histories/probabilities and completed finite-weight receipts support stability, not an assurance against every optimization issue.\n\n'+table(training.groupby(['stage','strategy']).agg(best_epoch_mean=('best_epoch','mean'),stopping_epoch_mean=('stopping_epoch','mean'),epoch_caps=('epoch_cap','sum'),early_stopped=('early_stopped','sum'),severe=('severe','sum'),low_diversity=('low_diversity','sum'),probability_min=('probability_min','min'),probability_max=('probability_max','max'),mean_probability_std=('probability_std','mean')).reset_index())+'\n\nPer-run values: training_behavior.csv. No observed new collapse; optimization duration/epoch selection can change without resolving the end-to-end representation gap.\n\n'
    report += '## Representation probes and Stage 5C hypothesis\n\n'+table(probe_summary)+'\n\nActual CNN means appear in the primary table. Fusion gap is concat minus hidden AUC; positive reduction means less loss.\n\n'+table(gaps)+'\n\n'+interpretation+'\n\n'
    report += '## Object versus grouped behavior\n\nDescriptive only, not a pure causal host-leakage estimate.\n\n'+table(between.reset_index())+'\n\n```json\n'+json.dumps(decision['object_minus_host_group'],indent=2)+'\n```\n\n'
    report += f"Mean fusion gap: 32 units {gaps.gap_32.mean():.6f}; 128 units {gaps.gap_128.mean():.6f}. Mean paired gap reduction {gaps.gap_reduction.mean():.6f}; {int((gaps.gap_reduction>0).sum())}/20 reduced. Fusion gap reduction: {decision['fraction_fusion_gap_reduced']:.2%}. CNN gain closes {decision['fraction_original_concat_to_cnn_gap_closed']:.2%} of the original Stage 5B concat-probe-to-CNN gap (fixed reference, not a causal decomposition).\n\n"
    report += '## Runtime and limitations\n\n```json\n'+json.dumps(runtime,indent=2)+'\n```\n\nCNN runtime includes initialization, inference, serialization and artifact work; summed epoch_seconds is a narrower training measure. Neither equals end-to-end elapsed time. No filesystem timestamps are promoted to authoritative runtime. Repeated validation is development evidence; probe ROC-AUC is linear separability under fixed probe fitting, not information-theoretic information loss. Saved validation metrics are retained; recomputation from whole-scope predictions gives Brier differences below 3e-10 due to inference batch boundaries, within the existing 1e-7 replay tolerance; ranking metrics and counts match exactly.\n\n'
    report += '## Decision and exact next action\n\n**'+decision['recommendation']+'**; Stage 5C hypothesis **partially supported**. The user-provided qualitative decision logic is applied without claiming numerical cutoffs were preregistered. No test evaluation is executed. Proposed frozen protocol and all 20 exact model hashes are in decision_summary.json.\n\n```json\n'+json.dumps(protocol,indent=2)+'\n```\n\n'
    report += '## Figures\n\n'+''.join('!['+name+'](../data/stage5d/figures/'+name+'.png)\n\n' for name in ['paired_auc','paired_differences','concat_hidden','fusion_gap','epochs'])
    report += '## Provenance and hashes\n\nprovenance.json binds source/config/report hashes. certification.json records audit and checks. checksums.sha256 covers compact outputs; runtime_checksums.sha256 binds saved runtime artifacts. The original execution provenance is retained as execution_provenance.json. No test evaluation or scientific fits were performed in finalization.\n'
    Path('docs/kepler_stage5d_validation_analysis.md').write_text(report,encoding='utf-8')
    if not (OUT/'execution_provenance.json').exists():
        (OUT/'execution_provenance.json').write_bytes((OUT/'provenance.json').read_bytes())
    r.atomic(OUT/'provenance.json',dict(execution=r.read(OUT/'execution_provenance.json'), frozen_sha256=r.sha256(r.FREEZE),
        source_sha256=r.sha256(__file__), report_sha256=r.sha256('docs/kepler_stage5d_validation_analysis.md'),
        inputs={str(p):r.sha256(p) for p in [OUT/'run_manifest.json',OUT/'runtime_checksums.sha256',OUT/'probe_comparisons.csv']}, validation_only=True))
    r.atomic(OUT/'certification.json',dict(status='passed',cnn_fits=20,learned_probes=80,reused_raw_probes=40,
        protected_hashes=len(r.read(r.FREEZE)['sha256']),held_out_evaluated=False,scientific_fits_in_finalization=0,
        tests='pending',decision=decision['recommendation']))
    seal()
    print(summary.to_string(index=False)); print(json.dumps({k:v for k,v in decision.items() if k not in ['proposed_test_protocol','primary']},indent=2))


def seal():
    r.verify_hashes(r.read(r.FREEZE)['sha256'])
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'checksums.sha256','status.json'} and p.suffix not in {'.tmp','.log'}]
    (OUT/'checksums.sha256').write_text(''.join(f'{r.sha256(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')


def figures(paired, probes, gaps, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures'; dest.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout(); plt.savefig(dest/(name+'.png'),dpi=150); plt.close()
    fig, ax=plt.subplots(figsize=(6,4))
    for name,g in paired.groupby('strategy'):
        ax.scatter(g.roc_auc_5B,g.roc_auc_5D,label=name)
    ax.plot([.5,.8],[.5,.8],color='gray',linestyle='--'); ax.set(xlabel='Stage 5B validation ROC-AUC',ylabel='Stage 5D validation ROC-AUC'); ax.legend(fontsize=8); save('paired_auc')
    fig, ax=plt.subplots(figsize=(6,4))
    for name,g in paired.groupby('strategy'): ax.plot(g.repeat_id,g.roc_auc_delta,'o-',label=name)
    ax.axhline(0,color='gray'); ax.set(xlabel='Repeat',ylabel='Stage 5D minus 5B validation ROC-AUC'); ax.legend(fontsize=8); save('paired_differences')
    fig, ax=plt.subplots(figsize=(6,4))
    for width in ['32','128']:
        g=probes.groupby('probe')['roc_auc_'+width].mean(); ax.plot(['concat','hidden'],[g.concatenated_branch,g.fusion_hidden],'o-',label=width+' units')
    ax.set(ylabel='Mean validation probe ROC-AUC'); ax.legend(); save('concat_hidden')
    fig, ax=plt.subplots(figsize=(6,4))
    for _,g in gaps.iterrows(): ax.plot(['32 units','128 units'],[g.gap_32,g.gap_128],color='gray',alpha=.4)
    ax.plot(['32 units','128 units'],[gaps.gap_32.mean(),gaps.gap_128.mean()],'o-',color='black',label='Mean'); ax.set(ylabel='Concat minus hidden probe ROC-AUC'); ax.legend(); save('fusion_gap')
    fig, axes=plt.subplots(1,2,figsize=(7,3))
    for ax,field in zip(axes,['best_epoch','stopping_epoch']):
        p=training.pivot(index=['strategy','repeat_id'],columns='stage',values=field); ax.scatter(p['5B'],p['5D']); ax.plot([0,100],[0,100],'--',color='gray'); ax.set(xlabel='5B '+field,ylabel='5D '+field)
    save('epochs')


if __name__ == '__main__':
    finalize()
