"""Isolated 128-unit experiment. --preflight never fits; --execute is explicit.

Only train/validation arrays cross the feature boundary. No held-out evaluation
function, test-evaluation phase, or test CLI option exists in this runner.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[_key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1', TF_DETERMINISTIC_OPS='1', TF_ENABLE_ONEDNN_OPTS='0', TF_CPP_MIN_LOG_LEVEL='2')
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
from kepler_baselines.inputs import IDENTITY, require, sha256, read_input_tables
from kepler_baselines.metrics import metrics
from kepler_baselines.runner import write_csv
from kepler_cnn.artifacts import align, views

CONFIG = Path('configs/kepler_stage5d_width128_v1.json')
FREEZE = Path('configs/kepler_stage5d_freeze_v1.json')
META = Path('data/stage5d')
ROOT = Path('data/stage5d_runtime/width128_v1')
PARENT = Path('data/stage5b_runtime/activation_v1')
PROBE_ROOT = Path('data/stage5c/runs')
RAW = ('raw_global', 'raw_local')
LEARNED = ('global_branch', 'local_branch', 'concatenated_branch', 'fusion_hidden')
CNN_FILES = {'model.keras', 'history.csv', 'train_validation_predictions.csv', 'record.json'}
PROBE_FILES = {'probability.npy', 'record.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(31):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 30:
                raise
            time.sleep(1)


def verify_hashes(entries):
    for name, digest in entries.items():
        require(Path(name).is_file() and sha256(name) == digest, 'Pinned artifact changed: '+name)


def identity(strategy, repeat, seed):
    return dict(strategy=strategy, repeat_id=int(repeat), seed=int(seed))


def key_for(who):
    return f"{who['strategy']}/repeat-{who['repeat_id']:02d}"


def identifiers(frame):
    columns = IDENTITY + ['qc_pass', 'partition', 'fold_id']
    return hashlib.sha256(frame[columns].to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def selected_checkpoint(history, outcome):
    require(len(history) > 0 and np.isfinite(history.to_numpy()).all(), 'Invalid training history')
    require(history.epoch.tolist() == list(range(1, len(history)+1)), 'Epoch sequence changed')
    best = history.loc[history.validation_loss.idxmin()]
    require(int(best.epoch) == outcome['best_epoch'], 'Checkpoint is not minimum validation BCE')
    require(abs(float(best.validation_loss)-outcome['best_validation_loss']) <= 1e-12, 'Checkpoint BCE differs')
    require(len(history) == outcome['stopping_epoch'] == min(outcome['best_epoch']+10, 100), 'Stopping rule changed')
    require(outcome['epoch_cap_reached'] == (len(history) == 100), 'Epoch-cap flag differs')
    return best


def scope_frame(objects, assigned):
    require(set(assigned.partition) <= {'train', 'validation', 'test'}, 'Unknown partition')
    keep = assigned.partition.isin(['train', 'validation']).to_numpy()
    frame = objects.loc[keep, IDENTITY+['qc_pass']].reset_index(drop=True)
    frame['partition'] = assigned.loc[keep, 'partition'].to_numpy()
    frame['fold_id'] = assigned.loc[keep, 'fold_id'].to_numpy()
    require(set(frame.partition) == {'train', 'validation'}, 'Missing train/validation partition')
    return keep, frame


@dataclass
class Scope:
    frame: pd.DataFrame
    inputs: dict

    def validate(self):
        require(set(self.frame.partition) == {'train', 'validation'}, 'Forbidden evaluation partition')
        require(all(len(v) == len(self.frame) for v in self.inputs.values()), 'Scope array alignment failed')

    def mask(self, partition):
        require(partition in {'train', 'validation'}, 'Forbidden evaluation partition')
        self.validate()
        return self.frame.partition.eq(partition).to_numpy()

    def select(self, partition):
        mask = self.mask(partition)
        return {k: v[mask] for k, v in self.inputs.items()}


def load_scope(matrix, objects, assigned):
    keep, frame = scope_frame(objects, assigned)
    # Select rows BEFORE conversion/materialization or inference. No full-cohort views.
    result = Scope(frame, views(matrix[keep]))
    result.validate()
    return result


def baseline_alignment(who, frame, baseline_record, prediction, history):
    require(all(baseline_record[k] == v for k, v in who.items()), 'Stage 5B strategy/repeat/seed mismatch')
    require(baseline_record['run_path'] == key_for(who), 'Stage 5B run path differs')
    for k, v in who.items():
        require(prediction[k].eq(v).all(), 'Stage 5B prediction run identity differs')
    require(set(prediction.partition) == {'train', 'validation'}, 'Baseline includes forbidden partition')
    columns = IDENTITY+['qc_pass', 'partition', 'fold_id']
    require(prediction[columns].reset_index(drop=True).equals(frame[columns]), 'Stage 5B validation/training identifiers differ')
    best = selected_checkpoint(history, baseline_record)
    return dict(**who, validation_roc_auc=float(best.validation_roc_auc), validation_bce=float(best.validation_loss),
                best_epoch=int(best.epoch), scope_sha256=identifiers(frame))


def raw_compatibility(who, frame, protocol, receipt, saved, config, parent):
    require(receipt['run_path'] == key_for(who), 'Stage 5C run path differs')
    require(receipt['protocol_sha256'] == sha256(config['probe_protocol']), 'Stage 5C receipt context differs')
    require(protocol['seeds'] == config['seeds'], 'Raw-probe seeds differ')
    require(protocol['parameters'] == config['probe_parameters'], 'Raw-probe fitting policy differs')
    require(protocol['evaluation'] == 'validation only' and protocol['threshold'] == .5, 'Probe evaluation policy differs')
    require(protocol['anchors']['configs\\kepler_stage5b_activation_v1.json'] == config['parent_config_sha256'], 'Raw-probe parent/cache configuration differs')
    for name, digest in parent['frozen_sha256'].items():
        require(protocol['anchors'].get(name) == digest, 'Raw-probe split/cohort/preprocessing anchor differs')
    val = frame.partition.eq('validation')
    require(np.array_equal(saved['kepoi_name'], frame.loc[val, 'kepoi_name']), 'Raw-probe validation KOIs differ')
    require(np.array_equal(saved['label'], frame.loc[val, 'label']), 'Raw-probe validation labels differ')
    result = {}
    for name, width in zip(RAW, (2000, 200)):
        rows = [r for r in receipt['metrics'] if r['probe'] == name]
        require(len(rows) == 1, 'Missing/duplicate raw probe')
        row = rows[0]
        require(all(row[k] == v for k, v in who.items()), 'Raw-probe strategy/repeat/seed differs')
        require(row['features'] == width and row['training_rows'] == int((~val).sum()), 'Raw-probe training population differs')
        actual = metrics(saved['label'], saved[name])
        require(row['sample_count'] == int(val.sum()) and abs(row['roc_auc']-actual['roc_auc']) <= 1e-12, 'Raw-probe metric differs')
        result[name] = dict(row=row, probability=np.array(saved[name], copy=True))
    return result


def preflight():
    config = read(CONFIG); frozen = read(FREEZE)
    verify_hashes(frozen['sha256'])
    parent = read(config['parent_config'])
    require(sha256(config['parent_config']) == config['parent_config_sha256'], 'Parent configuration changed')
    for name in ('training', 'seeds', 'strategies', 'feature_cache_sha256'):
        require(config[name] == parent[name], 'Unexpected scientific change: '+name)
    require(config['hidden_units'] == 128 and config['expected_parameters'] == 538113, 'Width policy changed')
    require(config['evaluation_partitions'] == ['train', 'validation'], 'Evaluation scope changed')
    from kepler_cnn.runner import context as environment
    require(environment() == read('data/stage5/run_manifest.json')['runs'][0]['context'], 'Pinned Python/TensorFlow/Keras environment differs')
    objects, _, assignments = read_input_tables(parent)
    require(objects.equals(pd.read_csv('data/stage4b/objects.csv')), 'Feature-cache KOI ordering differs')
    cache = Path(parent['baseline_runtime'])/'features.npy'
    require(sha256(cache) == config['feature_cache_sha256'], 'Certified feature cache changed')
    require(read('data/stage5/input_validation.json')['canonical_equals_existing_cache'] is True, 'Cache not certified canonical-equivalent')
    protocol = read(config['probe_protocol'])
    require(sha256('scripts/stage5c_linear_probes.py') == protocol['source_sha256'], 'Raw-probe source changed')
    require(sha256('data/stage5b/run_manifest.json') == protocol['anchors']['data\\stage5b\\run_manifest.json'], 'Certified Stage 5B manifest changed')
    parent_runs = read('data/stage5b/run_manifest.json')['runs']
    expected = {(s, i) for s in config['strategies'] for i in range(10)}
    require(len(parent_runs) == 20 and {(r['strategy'], r['repeat_id']) for r in parent_runs} == expected, 'Parent run matrix differs')
    references = {}; raw = {}
    for repeat, seed in enumerate(config['seeds']):
        for strategy in config['strategies']:
            who = identity(strategy, repeat, seed); key = key_for(who)
            _, frame = scope_frame(objects, align(objects, assignments, strategy, repeat))
            parent_receipt = read(PARENT/key/'training_complete.json')
            require(parent_receipt['context'] == sha256('configs/kepler_stage5b_freeze_v1.json'), 'Parent training context differs')
            require(set(parent_receipt['sha256']) == {'model.keras','history.csv','train_validation_predictions.csv','training_manifest.json'}, 'Parent training receipt schema differs')
            verify_hashes({str(PARENT/key/n): h for n,h in parent_receipt['sha256'].items() if n != 'model.keras'})
            record = read(PARENT/key/'training_manifest.json')
            require(record == next(r for r in parent_runs if r['run_path'] == key), 'Parent aggregate receipt differs')
            references[key] = baseline_alignment(who, frame, record,
                pd.read_csv(PARENT/key/'train_validation_predictions.csv', float_precision='round_trip'),
                pd.read_csv(PARENT/key/'history.csv', float_precision='round_trip'))
            receipt = read(PROBE_ROOT/key/'complete.json')
            verify_hashes(receipt['outputs'])
            with np.load(PROBE_ROOT/key/'validation_predictions.npz', allow_pickle=False) as saved:
                raw[key] = raw_compatibility(who, frame, protocol, receipt, saved, config, parent)
    return dict(config=config, parent=parent, objects=objects, assignments=assignments, cache=cache,
                references=references, raw=raw, context=sha256(FREEZE), environment=environment())


def receipt(directory, context, who, binding, names):
    path = directory/'complete.json'
    if not path.exists():
        require(not directory.exists() or not any(directory.iterdir()), 'Partial phase requires review; no automatic refit: '+str(directory))
        return None
    saved = read(path)
    require(saved['context'] == context and saved['identity'] == who and saved['binding'] == binding, 'Resume context/identity/binding differs')
    require(set(saved['sha256']) == set(names), 'Completion schema differs')
    verify_hashes({str(directory/n): h for n, h in saved['sha256'].items()})
    record = read(directory/'record.json')
    require(all(record[k] == v for k,v in who.items()), 'Receipt record identity differs')
    return record


def seal(directory, context, who, binding, names):
    atomic(directory/'complete.json', dict(context=context, identity=who, binding=binding,
        sha256={n: sha256(directory/n) for n in sorted(names)}))


def once(directory, context, who, binding, names, producer):
    saved = receipt(directory, context, who, binding, names)
    if saved is not None:
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    atomic(directory/'started.json', dict(context=context, identity=who, binding=binding))
    value = producer(directory)
    atomic(directory/'record.json', value)
    seal(directory, context, who, binding, names)
    return receipt(directory, context, who, binding, names)


def train_cnn(directory, who, scope, reference):
    import tensorflow as tf
    from kepler_cnn.model import fit, predict, weight_hash, verify_probability
    from kepler_cnn_width128.model import checked_initialization
    scope.validate(); started = time.perf_counter()
    model, initialization = checked_initialization(who['seed'])
    train = scope.mask('train'); val = scope.mask('validation')
    history, outcome = fit(model, scope.select('train'), scope.frame.loc[train, 'label'].to_numpy(),
                           scope.select('validation'), scope.frame.loc[val, 'label'].to_numpy(), who['seed'])
    best = selected_checkpoint(pd.DataFrame(history), outcome)
    probability = predict(model, scope.select('validation'))
    validation = metrics(scope.frame.loc[val, 'label'], probability)
    require(abs(validation['roc_auc']-float(best.validation_roc_auc)) <= 1e-12, 'Restored checkpoint AUC differs')
    before = weight_hash(model); require(all(np.isfinite(w).all() for w in model.get_weights()), 'Nonfinite weights')
    all_probability = predict(model, scope.inputs)
    model.save(directory/'model.keras')
    reloaded = tf.keras.models.load_model(directory/'model.keras', compile=False)
    actual = predict(reloaded, scope.inputs); verify_probability(all_probability, actual)
    require(np.array_equal(all_probability, actual), 'Same-context serialization is not bit-identical')
    require(weight_hash(reloaded) == before == weight_hash(model), 'Inference changed weights')
    prediction = scope.frame.assign(probability=all_probability.astype(float), prediction=(all_probability>=.5).astype(int), **who)
    write_csv(directory/'train_validation_predictions.csv', prediction)
    write_csv(directory/'history.csv', pd.DataFrame(history))
    result = dict(**{**who, **initialization, **outcome}, validation_roc_auc=validation['roc_auc'],
        validation_metrics=validation, parent_validation_roc_auc=reference['validation_roc_auc'],
        delta_validation_roc_auc=validation['roc_auc']-reference['validation_roc_auc'],
        scope_sha256=identifiers(scope.frame), final_weight_sha256=before, runtime_seconds=time.perf_counter()-started)
    del model, reloaded; tf.keras.backend.clear_session()
    return result


def probe_run(directory, who, scope, cnn, reference, raw, config, context):
    from scripts.stage5c_linear_probes import fit_probe
    rows = []; representations = None; started = time.perf_counter()
    binding = dict(scope_sha256=identifiers(scope.frame), final_weight_sha256=cnn['final_weight_sha256'])
    for name in RAW+LEARNED:
        target = directory/'probes'/name
        existing = receipt(target, context, who, binding, PROBE_FILES)
        if existing is not None:
            rows.append(existing)
            continue
        if name in RAW:
            probability = raw[name]['probability']; row = {**raw[name]['row'], 'reused': True}
            def producer(dest, p=probability, r=row):
                np.save(dest/'probability.npy', p, allow_pickle=False)
                return r
        else:
            if representations is None:
                import tensorflow as tf
                from kepler_cnn.model import weight_hash
                scope.validate()
                model = tf.keras.models.load_model(directory/'cnn'/'model.keras', compile=False)
                require(weight_hash(model) == cnn['final_weight_sha256'], 'Probe model weights differ')
                probe = tf.keras.Model(model.inputs, [model.get_layer(n).output for n in ['global_flatten', 'local_flatten', 'hidden_leaky_relu']])
                batches = [probe({k: v[i:i+64] for k, v in scope.inputs.items()}, training=False)
                           for i in range(0, len(scope.frame), 64)]
                values = [np.concatenate([b[j].numpy() for b in batches]) for j in range(3)]
                require([v.shape[1] for v in values] == [2560,1600,128], 'Probe representation dimensions differ')
                representations = dict(zip(LEARNED, [values[0],values[1],np.concatenate(values[:2],axis=1),values[2]]))
                require(weight_hash(model) == cnn['final_weight_sha256'], 'Probes changed weights')
                del model, probe, batches; tf.keras.backend.clear_session()
            def producer(dest, name=name):
                train = scope.mask('train'); val = scope.mask('validation')
                probability, iterations, _ = fit_probe(representations[name], scope.frame.label.to_numpy(), train, config['probe_parameters'])
                np.save(dest/'probability.npy', probability, allow_pickle=False)
                return dict(**who, probe=name, features=representations[name].shape[1], training_rows=int(train.sum()),
                            iterations=iterations, reused=False, **metrics(scope.frame.loc[val,'label'],probability))
        rows.append(once(target, context, who, binding, PROBE_FILES, producer))
    return rows


class Progress:
    def __init__(self, context, completed):
        self.value = dict(status='running', context=context, completed_cnn_fits=completed, total_cnn_fits=20, pid=os.getpid())
        self.lock = threading.Lock(); self.done = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.update(phase='preflight_complete'); self.thread.start()

    def update(self, **values):
        with self.lock:
            self.value.update(values); self.value['last_update'] = datetime.now(timezone.utc).isoformat()
            atomic(META/'status.json', self.value)

    def heartbeat(self):
        while not self.done.wait(10):
            self.update()

    def close(self):
        self.done.set(); self.thread.join()


def finish(cnns, probes, data):
    frame = pd.DataFrame(cnns)
    require(len(frame) == 20 and not frame.duplicated(['strategy','repeat_id']).any(), 'CNN result matrix differs')
    require(frame.groupby('repeat_id').initial_weight_sha256.nunique().eq(1).all(), 'Paired 128-unit initializations differ')
    paired = frame[['strategy','repeat_id','seed','best_epoch','validation_roc_auc','parent_validation_roc_auc','delta_validation_roc_auc','scope_sha256']]
    summary = paired.groupby('strategy').delta_validation_roc_auc.agg(['count','mean','median','std','min','max']).reset_index()
    counts = paired.assign(higher=paired.delta_validation_roc_auc>0,lower=paired.delta_validation_roc_auc<0,tied=paired.delta_validation_roc_auc==0).groupby('strategy')[['higher','lower','tied']].sum().reset_index()
    write_csv(META/'paired_validation.csv', paired)
    write_csv(META/'paired_summary.csv', summary.merge(counts,on='strategy',validate='one_to_one'))
    p = pd.DataFrame(probes)
    require(len(p) == 120 and not p.duplicated(['strategy','repeat_id','probe']).any() and int(p.reused.sum()) == 40, 'Probe matrix/reuse differs')
    old = pd.read_csv('data/stage5c/metrics.csv', float_precision='round_trip')
    joined = p.merge(old, on=['strategy','repeat_id','seed','probe'], validate='one_to_one', suffixes=('_128','_32'))
    require(len(joined)==120 and joined.sample_count_128.equals(joined.sample_count_32), 'Probe baseline alignment differs')
    joined['delta_roc_auc'] = joined.roc_auc_128-joined.roc_auc_32
    write_csv(META/'probe_metrics.csv', p)
    write_csv(META/'probe_comparisons.csv', joined[['strategy','repeat_id','seed','probe','roc_auc_128','roc_auc_32','delta_roc_auc','reused']])
    atomic(META/'run_manifest.json',dict(status='complete_validation_only',context=data['context'],runs=cnns,cnn_fits=20,new_probe_fits=80,reused_raw_probes=40))
    runtime_files = [p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix != '.tmp']
    (META/'runtime_checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in runtime_files),encoding='utf-8')
    verify_hashes(read(FREEZE)['sha256'])
    atomic(META/'provenance.json',dict(context=data['context'],environment=data['environment'],config_sha256=sha256(CONFIG),
        evaluation='validation only',checkpoint='minimum validation BCE',held_out_evaluation=False,
        warning='Repeated validation reuse is development evidence, not independent confirmation.'))
    files = [p for p in sorted(META.rglob('*')) if p.is_file() and p.name not in {'checksums.sha256','status.json'} and p.suffix not in {'.tmp','.log'}]
    (META/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(META).as_posix()}\n' for p in files),encoding='utf-8')


def execute(data):
    import msvcrt
    from kepler_cnn.model import configure
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    lock = (ROOT.parent/'runner.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    progress = None
    try:
        configure(); config = data['config']; context = data['context']; planned = []
        # Validate every existing phase before any new fit or status reset.
        for repeat, seed in enumerate(config['seeds']):
            for strategy in config['strategies']:
                who = identity(strategy, repeat, seed); key = key_for(who)
                binding = dict(scope_sha256=data['references'][key]['scope_sha256'])
                saved = receipt(ROOT/key/'cnn',context,who,binding,CNN_FILES)
                if saved is not None:
                    probe_binding = dict(**binding,final_weight_sha256=saved['final_weight_sha256'])
                    for name in RAW+LEARNED:
                        receipt(ROOT/key/'probes'/name,context,who,probe_binding,PROBE_FILES)
                else:
                    require(not (ROOT/key/'probes').exists(), 'Orphan probes without completed CNN')
                planned.append((who,key,binding,saved))
        progress = Progress(context,sum(saved is not None for _,_,_,saved in planned))
        atomic(META/'protocol.json',dict(config=config,context=context,freeze_sha256=sha256(FREEZE)))
        matrix = np.load(data['cache'],mmap_mode='r',allow_pickle=False)
        cnns = []; probes = []; completed = sum(saved is not None for _,_,_,saved in planned)
        for who,key,binding,saved in planned:
            verify_hashes(read(FREEZE)['sha256'])
            assigned = align(data['objects'],data['assignments'],who['strategy'],who['repeat_id'])
            scope = load_scope(matrix,data['objects'],assigned)
            require(identifiers(scope.frame)==binding['scope_sha256'],'Scope changed after preflight')
            progress.update(current_run=key,phase='cnn' if saved is None else 'verified_cnn_reused')
            cnn = once(ROOT/key/'cnn',context,who,binding,CNN_FILES,
                       lambda directory:train_cnn(directory,who,scope,data['references'][key]))
            if saved is None:
                completed += 1
            cnns.append(cnn); progress.update(completed_cnn_fits=completed,phase='linear_probes')
            probes.extend(probe_run(ROOT/key,who,scope,cnn,data['references'][key],data['raw'][key],config,context))
            print('Completed CNN and six validation probes: '+key,flush=True)
        finish(cnns,probes,data)
        progress.update(status='complete',phase='finished_validation_only',current_run=None,new_probe_fits=80,reused_raw_probes=40)
    except Exception as exc:
        if progress is not None:
            progress.update(status='stopped_for_review',error=str(exc))
        raise
    finally:
        if progress is not None:
            progress.close()
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight',action='store_true',help='Audit alignment and initialization; fit nothing')
    action.add_argument('--execute',action='store_true',help='Explicitly launch/resume the 20 CNN fits and 80 learned probes')
    args = parser.parse_args(argv)
    data = preflight()
    if args.preflight:
        from kepler_cnn.model import configure
        from kepler_cnn_width128.model import checked_initialization
        configure(); initializations = []
        for seed in data['config']['seeds']:
            model, evidence = checked_initialization(seed); initializations.append(evidence); del model
        evidence = dict(status='passed',mode='preflight_only',scientific_fits_executed=0,linear_probe_fits_executed=0,
            context=data['context'],baseline_runs=len(data['references']),compatible_raw_probes=40,
            initializations=initializations,environment=data['environment'],references=list(data['references'].values()))
        atomic(META/'preflight.json',evidence)
        print(json.dumps({k:v for k,v in evidence.items() if k not in {'initializations','environment','references'}},indent=2))
    else:
        execute(data)


if __name__ == '__main__':
    main()
