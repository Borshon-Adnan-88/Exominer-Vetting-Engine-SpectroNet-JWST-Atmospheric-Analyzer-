"""Operational recovery: preserve completed fits and reuse certified identical flux cache."""
from scripts import stage5b_run_activation_variant as runner
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd


def resilient_atomic(path, value):
    """Use a fresh sibling file and tolerate transient Windows sharing locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main():
    # Check the dedicated environment before touching progress or artifacts.
    import tensorflow as tf
    runner.atomic = resilient_atomic
    from kepler_cnn.runner import context as parent_context
    from kepler_baselines.inputs import require,sha256
    parent=json.loads(Path('data/stage5/run_manifest.json').read_text())['runs'][0]['context']
    require(parent_context()==parent,'Pinned TensorFlow/Python environment mismatch')
    context=sha256(runner.FREEZE)
    frozen=json.loads(runner.FREEZE.read_text())
    runner.verify_hashes(frozen['sha256'])
    config=json.loads(runner.CONFIG.read_text())
    cache=Path(config['baseline_runtime'])/'features.npy'
    runner.locally_available(cache)
    require(sha256(cache)==config['feature_cache_sha256'],'Certified feature cache mismatch')
    proof=json.loads(Path('data/stage5/input_validation.json').read_text())
    require(proof['canonical_equals_existing_cache'] is True,'Canonical/cache equivalence not certified')
    import msvcrt
    lock=(runner.META/'runner.lock').open('r+b'); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        records=[]; partial=[]
        for repeat,seed in enumerate(config['seeds']):
            for strategy in config['strategies']:
                directory=runner.ROOT/strategy/f'repeat-{repeat:02d}'
                if (directory/'training_complete.json').exists():
                    record=runner.verified_receipt(directory,'training',context)
                    require(record['strategy']==strategy and record['repeat_id']==repeat and record['seed']==seed,'Resume identity mismatch')
                    records.append(record)
                elif directory.exists() and any(directory.iterdir()):
                    require(directory==runner.ROOT/'host_group_stratified'/'repeat-04','Unexpected partial run')
                    require({p.name for p in directory.iterdir()}=={'started.json'},'Unexpected partial checkpoint; stop without discarding')
                    partial.append(directory)
        require(len(records)>=9,'Previously completed runs missing')
        preserved=len(records)
        evidence=runner.META/'recovery'/str(time.time_ns()); evidence.mkdir(parents=True)
        if (runner.META/'status.json').exists():
            (evidence/'previous_status.json').write_bytes((runner.META/'status.json').read_bytes())
        for directory in partial:
            source=(directory/'started.json').resolve(); target=(evidence/'interrupted_repeat04_started.json').resolve()
            require(source.is_relative_to(runner.ROOT.resolve()) and target.is_relative_to(runner.META.resolve()),'Unsafe recovery path')
            source.rename(target)
        runner.atomic(evidence/'recovery.json',dict(preserved_models=preserved,context=context,
            python=sys.executable,tensorflow=tf.__version__,cache_sha256=sha256(cache),
            policy='No complete fit repeated. Only interrupted run without checkpoint restarts from its frozen seed. Canonical-equivalent frozen cache reused.',
            protected={str(runner.ROOT/r['run_path']/'training_complete.json'):sha256(runner.ROOT/r['run_path']/'training_complete.json') for r in records}))
    finally: lock.close()
    original_progress=runner.Progress
    class RecoveredProgress(original_progress):
        def update(self,**values):
            values['completed_models']=max(preserved,values.get('completed_models',self.value.get('completed_models',0)))
            return super().update(**values)
    def certified_cached_features(dataset,objects,index):
        require(sha256(cache)==config['feature_cache_sha256'],'Feature cache changed during recovery')
        expected=pd.read_csv('data/stage4b/objects.csv')
        require(objects.equals(expected),'Cached KOI ordering differs')
        matrix=np.load(cache,mmap_mode='r',allow_pickle=False)
        require(matrix.shape==(len(objects),2200) and np.isfinite(matrix).all(),'Invalid cached canonical features')
        return matrix
    runner.Progress=RecoveredProgress
    runner.load_features=certified_cached_features
    runner.main()

if __name__=='__main__': main()
