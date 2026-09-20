"""One terminal invocation for the forty sequential, fixed baseline fits."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from .inputs import (IDENTITY, StopForReview, load_config, load_features,
                     read_input_tables, require, sha256, verify_frozen_hashes)
from .metrics import evaluate_predictions, paired_and_summary
from .models import fit_training_only, probabilities, verify_identical_predictions
from .reporting import make_plots
from .runtime import MemoryMonitor


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode())


def source_hashes():
    paths = sorted(Path("kepler_baselines").glob("*.py")) + [Path("scripts/stage4b_run_baselines.py")]
    return {path.as_posix(): sha256(path) for path in paths}


def environment(config):
    actual = {name: importlib.metadata.version(name) for name in config["environment_versions"]}
    require(actual == config["environment_versions"], "pinned dependency version mismatch")
    with threadpool_limits(limits=1):
        libraries = [{key: item.get(key) for key in ("user_api", "internal_api", "num_threads", "version", "architecture")} for item in threadpool_info()]
        require(all(item["num_threads"] == 1 for item in libraries), "numerical thread limit not enforced")
    return {"python": sys.version.split()[0], "system": platform.system(), "machine": platform.machine(), "packages": actual, "thread_libraries": libraries}


def run(dataset, output, metadata, config_path="configs/kepler_stage4b_baselines_v1.json"):
    started = time.perf_counter()
    config = load_config(config_path)
    output, metadata = Path(output), Path(metadata)
    require(not output.exists() or not any(output.iterdir()), "run output already exists; automatic overwrite/refit prohibited")
    require(not metadata.exists() or not any(metadata.iterdir()), "metadata output already exists; review required")
    output.mkdir(parents=True, exist_ok=True); metadata.mkdir(parents=True, exist_ok=True)
    monitor = MemoryMonitor(); monitor.start()
    records = []
    current_run = "input_validation"
    try:
        env = environment(config)
        code_hashes = source_hashes()
        config_hash = sha256(config_path)
        objects, index, assignments = read_input_tables(config)
        features = load_features(dataset, objects, index)
        labels = objects.label.to_numpy(dtype=np.int64)
        require(features.shape == (6637, 2200), "feature matrix shape mismatch")
        np.save(output / "features.npy", features, allow_pickle=False)
        write_csv(metadata / "objects.csv", objects)
        validation = {"status": "passed", "cohort_size": len(objects), "host_count": int(objects.kepid.nunique()),
                      "feature_count": 2200, "feature_order": ["global_flux[0:2000]", "local_flux[0:200]"],
                      "qc_pass": int(objects.qc_pass.sum()), "qc_fail": int((~objects.qc_pass).sum()),
                      "canonical_npz_verified": len(objects), "logical_arrays_verified": len(objects) * 6,
                      "assignment_rows": len(assignments), "frozen_sha256": config["frozen_sha256"],
                      "features_sha256": sha256(output / "features.npy"), "objects_sha256": sha256(metadata / "objects.csv"),
                      "validation_seconds": time.perf_counter() - started}
        write_json(metadata / "input_validation.json", validation)
        metric_rows, bin_rows, diagnostic_rows, overlap_rows = [], [], [], []
        for family in config["fit_order"]:
            for repeat, seed in enumerate(config["seeds"]):
                for strategy in config["strategies"]:
                    current_run = f"{family}/{strategy}/repeat-{repeat:02d}"
                    print(f"Fit {len(records) + 1}/40: {current_run}", flush=True)
                    verify_frozen_hashes(config)
                    require(sha256(config_path) == config_hash and source_hashes() == code_hashes, "experiment code/configuration changed during execution")
                    selected = assignments.loc[assignments.strategy.eq(strategy) & assignments.repeat_id.eq(repeat)]
                    aligned = objects[IDENTITY].merge(selected[IDENTITY + ["partition", "fold_id"]], on=IDENTITY, how="left", validate="one_to_one", sort=False)
                    require(aligned.partition.notna().all() and aligned.kepoi_name.equals(objects.kepoi_name), "fit identity alignment failed")
                    partitions = aligned.partition.to_numpy()
                    begin = time.perf_counter(); monitor.phase = family
                    model, iterations = fit_training_only(family, features, labels, partitions, config, seed)
                    fit_seconds = time.perf_counter() - begin
                    probability = probabilities(model, features)
                    verify_identical_predictions(probability, probabilities(model, features), "repeated prediction")
                    model_path = output / "models" / (current_run + ".joblib")
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_path, compress=3)
                    reloaded = joblib.load(model_path)
                    verify_identical_predictions(probability, probabilities(reloaded, features), "model reload")
                    training_hosts = set(objects.loc[partitions == "train", "kepid"])
                    prediction = objects[IDENTITY + ["qc_pass"]].copy()
                    prediction["model"] = family; prediction["strategy"] = strategy; prediction["repeat_id"] = repeat
                    prediction["seed"] = seed; prediction["partition"] = partitions; prediction["fold_id"] = aligned.fold_id
                    prediction["probability"] = probability
                    prediction["prediction"] = (probability >= .5).astype(int)
                    prediction["host_seen_in_training"] = prediction.kepid.isin(training_hosts)
                    prediction_path = output / "predictions" / (current_run + ".csv")
                    write_csv(prediction_path, prediction)
                    roundtrip = pd.read_csv(prediction_path, float_precision="round_trip")
                    verify_identical_predictions(probability, roundtrip.probability.to_numpy(), "CSV roundtrip")
                    rows, bins, diagnostic = evaluate_predictions(prediction, family, strategy, repeat)
                    metric_rows.extend(rows); bin_rows.extend(bins); diagnostic_rows.extend(diagnostic)
                    if family == config["fit_order"][0]:
                        host_sets = {part: set(prediction.loc[prediction.partition.eq(part), "kepid"]) for part in ("train", "validation", "test")}
                        overlap_rows.append({"strategy": strategy, "repeat_id": repeat,
                            "train_validation_hosts": len(host_sets["train"] & host_sets["validation"]),
                            "train_test_hosts": len(host_sets["train"] & host_sets["test"]),
                            "validation_test_hosts": len(host_sets["validation"] & host_sets["test"])})
                    record = {"model": family, "strategy": strategy, "repeat_id": repeat, "seed": seed,
                              "training_count": int((partitions == "train").sum()), "fit_seconds": fit_seconds,
                              "fit_predict_serialize_seconds": time.perf_counter() - begin,
                              "convergence": "converged" if iterations is not None else "not_applicable",
                              "iterations": iterations, "deterministic_prediction_verified": True,
                              "model_path": model_path.relative_to(output).as_posix(), "model_bytes": model_path.stat().st_size,
                              "model_sha256": sha256(model_path), "prediction_path": prediction_path.relative_to(output).as_posix(),
                              "prediction_sha256": sha256(prediction_path)}
                    records.append(record)
                    write_json(metadata / "run_manifest.json", {"status": "running", "completed_fits": len(records), "runs": records})
                    print(f"Completed {len(records)}/40; fit={fit_seconds:.2f}s; iterations={iterations}", flush=True)
                    del model, reloaded
        monitor.phase = "reporting"
        metric_frame = pd.DataFrame(metric_rows)
        calibration_frame = pd.DataFrame(bin_rows)
        diagnostic_frame = pd.DataFrame(diagnostic_rows)
        paired, summary, differences = paired_and_summary(metric_frame)
        for name, frame in (("metrics.csv", metric_frame), ("calibration_bins.csv", calibration_frame),
                            ("host_diagnostic.csv", diagnostic_frame), ("host_overlap.csv", pd.DataFrame(overlap_rows)),
                            ("paired_differences.csv", paired), ("repeat_summary.csv", summary), ("paired_summary.csv", differences)):
            write_csv(metadata / name, frame)
        make_plots(metric_frame, calibration_frame, diagnostic_frame, metadata / "figures")
        verify_frozen_hashes(config)
        require(len(records) == config["expected_fits"] == 40, "fit denominator mismatch")
        require(source_hashes() == code_hashes and sha256(config_path) == config_hash, "experiment code/configuration changed")
        memory = monitor.stop()
        runtime = {"total_experiment_seconds": time.perf_counter() - started, "input_validation_seconds": validation["validation_seconds"],
                   **memory, "families": {family: {"fits": sum(r["model"] == family for r in records),
                     "fit_seconds": sum(r["fit_seconds"] for r in records if r["model"] == family),
                     "fit_predict_serialize_seconds": sum(r["fit_predict_serialize_seconds"] for r in records if r["model"] == family),
                     "model_artifact_bytes": sum(r["model_bytes"] for r in records if r["model"] == family)} for family in config["fit_order"]}}
        write_json(metadata / "runtime.json", runtime)
        write_json(metadata / "run_manifest.json", {"status": "complete", "completed_fits": len(records), "runs": records})
        write_json(metadata / "provenance.json", {"experiment_id": config["experiment_id"], "status": "complete_pending_independent_prediction_replay",
            "config_sha256": config_hash, "source_sha256": code_hashes, "frozen_input_sha256": config["frozen_sha256"],
            "environment": env, "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
            "git_worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
            "fit_count": len(records), "feature_cache_sha256": validation["features_sha256"],
            "input_validation_sha256": sha256(metadata / "input_validation.json"),
            "run_manifest_sha256": sha256(metadata / "run_manifest.json"),
            "metric_artifacts_sha256": {p.name: sha256(p) for p in sorted(metadata.glob("*.csv"))},
            "primary_comparison": "test ROC-AUC object-minus-grouped paired by repeat_id",
            "host_diagnostic_role": "secondary_only_no_effect_on_fitting_threshold_or_primary_comparison",
            "qc_sensitivity_role": "evaluation_only_identical_fitted_models",
            "reproducibility_scope": "Pinned environment and single numerical thread; repeated/reloaded predictions bit-identical. No cross-platform binary identity claim."})
        write_checksums(metadata, output, records)
        print(json.dumps(runtime, indent=2), flush=True)
    except Exception as exc:
        memory = monitor.stop() if not monitor.stopped.is_set() else {}
        write_json(metadata / "stopped_for_review.json", {"status": "stopped_for_review", "run": current_run,
                   "reason": str(exc), "exception_type": type(exc).__name__, "completed_fits": len(records),
                   "runtime_seconds": time.perf_counter() - started, **memory})
        raise


def write_checksums(metadata, output, records):
    metadata, output = Path(metadata), Path(output)
    lines = [f"{sha256(path)}  {path.relative_to(metadata).as_posix()}\n" for path in sorted(metadata.rglob("*")) if path.is_file() and path.name != "checksums.sha256"]
    (metadata / "checksums.sha256").write_bytes("".join(lines).encode())
    (metadata / "prediction_checksums.sha256").write_bytes("".join(f"{r['prediction_sha256']}  {r['prediction_path']}\n" for r in records).encode())
    (metadata / "model_checksums.sha256").write_bytes("".join(f"{r['model_sha256']}  {r['model_path']}\n" for r in records).encode())
    # Recompute after writing the two runtime-root manifests.
    lines = [f"{sha256(path)}  {path.relative_to(metadata).as_posix()}\n" for path in sorted(metadata.rglob("*")) if path.is_file() and path.name != "checksums.sha256"]
    (metadata / "checksums.sha256").write_bytes("".join(lines).encode())


def verify_predictions(output, metadata, config_path="configs/kepler_stage4b_baselines_v1.json"):
    """Fresh-process replay: load stored estimators; never fit anything."""
    started = time.perf_counter()
    output, metadata = Path(output), Path(metadata)
    config = load_config(config_path); environment(config)
    objects, _, assignments = read_input_tables(config)
    provenance = json.loads((metadata / "provenance.json").read_text())
    require(provenance["config_sha256"] == sha256(config_path) and provenance["source_sha256"] == source_hashes(), "replay code/config mismatch")
    for line in (metadata / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(sha256(metadata / relative) == expected, "result metadata checksum mismatch")
    manifest = json.loads((metadata / "run_manifest.json").read_text())
    require(manifest["status"] == "complete" and manifest["completed_fits"] == 40, "experiment incomplete")
    require(sha256(output / "features.npy") == provenance["feature_cache_sha256"], "feature-cache hash mismatch")
    features = np.load(output / "features.npy", allow_pickle=False)
    require(features.shape == (len(objects), 2200) and np.isfinite(features).all(), "invalid replay features")
    seen = set()
    for record in manifest["runs"]:
        key = (record["model"], record["strategy"], record["repeat_id"])
        require(key not in seen, "duplicate fit in replay manifest"); seen.add(key)
        model_path, prediction_path = output / record["model_path"], output / record["prediction_path"]
        require(sha256(model_path) == record["model_sha256"] and sha256(prediction_path) == record["prediction_sha256"], "runtime model/prediction hash mismatch")
        prediction = pd.read_csv(prediction_path, float_precision="round_trip")
        require(prediction[IDENTITY].equals(objects[IDENTITY]), "prediction identity join mismatch")
        expected = assignments.loc[assignments.strategy.eq(record["strategy"]) & assignments.repeat_id.eq(record["repeat_id"])]
        joined = prediction.merge(expected[IDENTITY + ["partition"]], on=IDENTITY, validate="one_to_one", suffixes=("", "_frozen"))
        require(joined.partition.eq(joined.partition_frozen).all(), "prediction partition changed")
        probability = probabilities(joblib.load(model_path), features)
        verify_identical_predictions(probability, prediction.probability.to_numpy(), "independent process")
        require(np.array_equal((probability >= .5).astype(int), prediction.prediction.to_numpy()), "prediction threshold changed")
    require(len(seen) == 40, "independent replay count mismatch")
    result = {"status": "passed", "models_replayed": 40, "predictions_verified": 40 * len(objects),
              "new_fits": 0, "bit_identical_probabilities": True, "runtime_seconds": time.perf_counter() - started}
    write_json(metadata / "prediction_verification.json", result)
    provenance["status"] = "complete_verified"
    provenance["prediction_verification_sha256"] = sha256(metadata / "prediction_verification.json")
    write_json(metadata / "provenance.json", provenance)
    write_checksums(metadata, output, manifest["runs"])
    print(json.dumps(result, indent=2), flush=True)
