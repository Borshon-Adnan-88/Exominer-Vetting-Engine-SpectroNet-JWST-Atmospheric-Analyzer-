"""Hash-verified resumable twenty-fit experiment and independent replay."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from kepler_baselines.inputs import IDENTITY, load_config, load_features, read_input_tables, require, sha256
from kepler_baselines.metrics import evaluate_predictions, paired_and_summary
from kepler_baselines.runner import write_json, write_csv
from kepler_baselines.runtime import MemoryMonitor
from .artifacts import (align, baseline_comparisons, check_prediction_identity, checksums,
                        completed_run, verify_baselines, views)
from .model import build, configure, fit, predict, verify_probability, weight_hash

CONFIG = Path("configs/kepler_stage5_cnn_v1.json")
OUTPUT = Path("data/stage5_runtime/cnn_v1")
META = Path("data/stage5")


def context():
    sources = list(Path("kepler_cnn").glob("*.py")) + list(Path("kepler_baselines").glob("*.py")) + [Path("scripts/stage5_run_cnn.py")]
    return {"config_sha256": sha256(CONFIG), "source_sha256": {p.as_posix(): sha256(p) for p in sorted(sources)},
        "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in
        ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "scipy", "h5py")}}


def run(dataset, *, replay=False):
    started = time.perf_counter(); configure()
    config = load_config(CONFIG); expected_context = context()
    require(expected_context["packages"] == config["environment_versions"], "CNN pinned environment mismatch")
    META.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    monitor = MemoryMonitor(); monitor.start()
    records = []; current = "validation"
    try:
        verify_baselines(config)
        objects, index, assignments = read_input_tables(config)
        cache = Path(config["baseline_runtime"]) / "features.npy"
        require(sha256(cache) == config["feature_cache_sha256"], "Stage 4B flux cache mismatch")
        cached = np.load(cache, mmap_mode="r", allow_pickle=False)
        baseline_objects = pd.read_csv("data/stage4b/objects.csv")
        require(objects.equals(baseline_objects), "cached flux identity order mismatch")
        if not replay:
            canonical = load_features(dataset, objects, index)
            require(np.array_equal(canonical, cached), "canonical/cache value mismatch")
            del canonical
        inputs = views(cached); labels = objects.label.to_numpy()
        del cached
        validation_seconds = time.perf_counter() - started
        if not replay:
            require(Path("data/stage5_infrastructure/passed.json").is_file(), "synthetic infrastructure test has not passed")
            smoke = json.loads(Path("data/stage5_infrastructure/passed.json").read_text())
            require(smoke["model_source_sha256"] == sha256("kepler_cnn/model.py"), "infrastructure proof is stale")
            write_json(META / "input_validation.json", {"status": "passed", "kois": len(objects),
                "hosts": int(objects.kepid.nunique()), "qc_pass": int(objects.qc_pass.sum()), "qc_fail": int((~objects.qc_pass).sum()),
                "canonical_files_verified": len(objects), "logical_arrays_verified": len(objects) * 6,
                "assignment_rows": len(assignments), "feature_keys": list(inputs), "canonical_equals_existing_cache": True,
                "frozen_sha256": config["frozen_sha256"], "baseline_sha256": config["baseline_sha256"],
                "validation_seconds": validation_seconds})
        metric_rows, bin_rows, diagnostic_rows = [], [], []
        initial_hashes = {}; replay_max_error = 0.; new_fits = 0
        for repeat, seed in enumerate(config["seeds"]):
            for strategy in config["strategies"]:
                current = f"{strategy}/repeat-{repeat:02d}"
                identity = {"model_version": config["experiment_id"], "strategy": strategy, "repeat_id": repeat}
                directory = OUTPUT / current
                require(context() == expected_context, "CNN code/config changed during execution")
                # Recheck all metadata anchors before every fit/replay.
                for name, digest in {**config["frozen_sha256"], **config["baseline_sha256"]}.items():
                    require(sha256(name) == digest, f"frozen input changed: {name}")
                aligned = align(objects, assignments, strategy, repeat)
                record = completed_run(directory, identity, expected_context)
                if record is None:
                    require(not replay, f"incomplete scientific run: {current}")
                    if directory.exists():
                        # Preserve partial attempts; never delete or silently overwrite them.
                        require(directory.resolve().is_relative_to(OUTPUT.resolve()), "unsafe partial directory")
                        directory.rename(directory.with_name(directory.name + f"-partial-{time.time_ns()}"))
                    directory.mkdir(parents=True)
                    print(f"Scientific fit {len(records) + 1}/20: {current}", flush=True)
                    begin = time.perf_counter(); monitor.phase = current
                    model = build(seed); initial = weight_hash(model)
                    if repeat in initial_hashes:
                        require(initial == initial_hashes[repeat], "paired initial weights differ")
                    initial_hashes[repeat] = initial
                    train = aligned.partition.eq("train").to_numpy(); val = aligned.partition.eq("validation").to_numpy()
                    require(not (train & val).any(), "training/validation overlap")
                    # Test rows are never passed to the fitting boundary or its callbacks.
                    history, outcome = fit(model, {k: v[train] for k, v in inputs.items()}, labels[train],
                        {k: v[val] for k, v in inputs.items()}, labels[val], seed)
                    fit_seconds = time.perf_counter() - begin
                    probability = predict(model, inputs)
                    verify_probability(probability, predict(model, inputs))
                    model.save(directory / "model.keras")
                    prediction = objects[IDENTITY + ["qc_pass"]].copy()
                    prediction["model"] = "dual_view_cnn"; prediction["strategy"] = strategy
                    prediction["repeat_id"] = repeat; prediction["seed"] = seed
                    prediction["partition"] = aligned.partition; prediction["fold_id"] = aligned.fold_id
                    prediction["probability"] = probability.astype(np.float64)
                    prediction["prediction"] = (probability >= .5).astype(int)
                    prediction["host_seen_in_training"] = objects.kepid.isin(set(objects.loc[train, "kepid"]))
                    write_csv(directory / "predictions.csv", prediction)
                    write_csv(directory / "history.csv", pd.DataFrame(history))
                    reload_model = tf.keras.models.load_model(directory / "model.keras", compile=False)
                    verify_probability(probability, predict(reload_model, inputs))
                    del reload_model, model
                    record = {"identity": identity, "context": expected_context, "seed": seed,
                        "initial_weight_sha256": initial, "parameters": 138561, "fit_seconds": fit_seconds,
                        "fit_and_artifact_seconds": time.perf_counter() - begin, "outcome": outcome,
                        "model_bytes": (directory / "model.keras").stat().st_size,
                        "sha256": {name: sha256(directory / name) for name in ("model.keras", "predictions.csv", "history.csv")}}
                    # Completion is published only after serialized prediction verification succeeds.
                    write_json(directory / "manifest.json", record)
                    new_fits += 1
                else:
                    print(f"Verified completed run {len(records) + 1}/20: {current}", flush=True)
                initial = record["initial_weight_sha256"]
                require(repeat not in initial_hashes or initial_hashes[repeat] == initial, "paired initial hash mismatch")
                initial_hashes[repeat] = initial
                prediction = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
                check_prediction_identity(prediction, objects, aligned)
                # Exact identity joins with both frozen baseline prediction tables.
                for baseline in ("random_forest", "logistic_regression"):
                    b = pd.read_csv(Path(config["baseline_runtime"]) / "predictions" / baseline / (current + ".csv"))
                    require(prediction[IDENTITY + ["partition", "qc_pass"]].equals(b[IDENTITY + ["partition", "qc_pass"]]), "baseline prediction identity mismatch")
                if replay:
                    model = tf.keras.models.load_model(directory / "model.keras", compile=False)
                    actual = predict(model, inputs)
                    verify_probability(prediction.probability.to_numpy(), actual)
                    replay_max_error = max(replay_max_error, float(np.max(np.abs(prediction.probability - actual))))
                    prediction["probability"] = actual.astype(np.float64)
                    del model
                rows, bins, diagnostic = evaluate_predictions(prediction, "dual_view_cnn", strategy, repeat)
                metric_rows.extend(rows); bin_rows.extend(bins); diagnostic_rows.extend(diagnostic)
                records.append({"run_path": current, **record})
                if not replay:
                    write_json(META / "run_manifest.json", {"status": "running", "completed_fits": len(records), "runs": records})
                    print(f"Completed {len(records)}/20: {record['outcome']['status']}, best epoch {record['outcome']['best_epoch']}", flush=True)
        frame = pd.DataFrame(metric_rows)
        require(len(records) == 20 and len(frame) == 120, "scientific run count mismatch")
        verify_baselines(config)
        if replay:
            saved = pd.read_csv(META / "metrics.csv")
            for column in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[column]):
                    require(np.allclose(frame[column], saved[column], rtol=0, atol=1e-7, equal_nan=True), f"replay metrics differ: {column}")
                else:
                    require(frame[column].fillna("").equals(saved[column].fillna("")), f"replay metric identity mismatch: {column}")
            report = {"status": "passed", "models": 20, "predictions": len(objects) * 20, "new_fits": new_fits,
                "absolute_tolerance": 1e-7, "relative_tolerance": 0, "maximum_probability_error": replay_max_error,
                "labels_identical": True, "metrics_recomputed": 120, "runtime_seconds": time.perf_counter() - started}
            write_json(META / "prediction_verification.json", report)
            provenance = json.loads((META / "provenance.json").read_text())
            require(provenance["context"] == expected_context, "replay provenance context mismatch")
            provenance["status"] = "complete_verified"; provenance["prediction_verification_sha256"] = sha256(META / "prediction_verification.json")
            write_json(META / "provenance.json", provenance)
            monitor.stop(); checksums(META, OUTPUT); print(json.dumps(report), flush=True)
            return
        paired, summary, differences = paired_and_summary(frame)
        comparisons, comparison_summary = baseline_comparisons(frame)
        for name, table in (("metrics", frame), ("calibration_bins", pd.DataFrame(bin_rows)),
            ("host_diagnostic", pd.DataFrame(diagnostic_rows)), ("paired_differences", paired),
            ("repeat_summary", summary), ("paired_summary", differences),
            ("baseline_comparisons", comparisons), ("baseline_comparison_summary", comparison_summary)):
            write_csv(META / (name + ".csv"), table)
        write_json(META / "run_manifest.json", {"status": "complete", "completed_fits": 20, "new_fits_this_invocation": new_fits, "runs": records})
        times = [r["fit_seconds"] for r in records]
        write_json(META / "runtime.json", {"invocation_seconds": time.perf_counter() - started,
            "validation_seconds": validation_seconds, "fit_seconds_sum": sum(times),
            "fit_seconds_median": float(np.median(times)), "fit_seconds_min": min(times), "fit_seconds_max": max(times),
            "model_bytes": sum(r["model_bytes"] for r in records), **monitor.stop()})
        write_json(META / "provenance.json", {"status": "complete_pending_fresh_replay", "context": expected_context,
            "experiment_id": config["experiment_id"], "frozen_input_sha256": config["frozen_sha256"],
            "baseline_sha256": config["baseline_sha256"], "platform": platform.platform(),
            "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
            "parameters": 138561, "scientific_fits": 20, "thread_policy": "CPU sequential, intra/inter-op and numerical threads 1; oneDNN disabled",
            "seed_policy": "Stage 3 seed; initializer offsets 0..4 and 6, dropout offset 5; NumPy default_rng training permutations",
            "replay_tolerance": {"atol": 1e-7, "rtol": 0, "labels": "exact"},
            "history_loss_definition": "end-of-epoch inference BCE; dropout disabled; ROC-AUC computed by shared sklearn definition",
            "reproducibility_scope": "Pinned environment; synthetic training determinism and scientific prediction replay; no independent scientific retraining or cross-platform claim",
            "validation_sha256": sha256(META / "input_validation.json"), "run_manifest_sha256": sha256(META / "run_manifest.json")})
        checksums(META, OUTPUT)
    except Exception as exc:
        if not monitor.stopped.is_set():
            monitor.stop()
        write_json(META / "stopped_for_review.json", {"run": current, "reason": str(exc), "type": type(exc).__name__, "completed_runs": len(records)})
        raise
