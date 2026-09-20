"""Atomic run completion, immutable input anchors and replayable artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kepler_baselines.inputs import IDENTITY, require, sha256, verify_frozen_hashes
from kepler_baselines.runner import write_json, write_csv


def verify_baselines(config):
    verify_frozen_hashes(config)
    for filename, digest in config["baseline_sha256"].items():
        require(Path(filename).is_file() and sha256(filename) == digest, f"Stage 4B artifact mismatch: {filename}")
    root = Path(config["baseline_runtime"])
    for line in Path("data/stage4b/prediction_checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        require(sha256(root / filename) == digest, f"baseline prediction mismatch: {filename}")


def views(matrix):
    require(matrix.ndim == 2 and matrix.shape[1] == 2200 and np.isfinite(matrix).all(), "invalid flux matrix")
    # The source values are canonical float32, promoted exactly by Stage 4B.
    result = matrix.astype(np.float32)
    require(np.array_equal(matrix, result.astype(matrix.dtype)), "flux cast changed canonical values")
    return {"global_flux": result[:, :2000, None], "local_flux": result[:, 2000:, None]}


def align(objects, assignments, strategy, repeat):
    selected = assignments.loc[assignments.strategy.eq(strategy) & assignments.repeat_id.eq(repeat)]
    result = objects[IDENTITY].merge(selected[IDENTITY + ["partition", "fold_id"]], on=IDENTITY,
        how="left", validate="one_to_one", sort=False)
    require(result.partition.notna().all() and result.kepoi_name.equals(objects.kepoi_name), "CNN identity alignment failed")
    return result


def completed_run(directory, identity, context):
    """Absence of completion manifest means partial; corrupt completed runs stop."""
    directory = Path(directory)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return None
    record = json.loads(manifest.read_text())
    require(record["identity"] == identity and record["context"] == context, "completed-run context mismatch")
    require(set(record["sha256"]) == {"model.keras", "predictions.csv", "history.csv"}, "incomplete run manifest")
    for name, digest in record["sha256"].items():
        require((directory / name).is_file() and sha256(directory / name) == digest, f"completed-run artifact mismatch: {name}")
    return record


def check_prediction_identity(prediction, objects, aligned):
    require(prediction[IDENTITY].equals(objects[IDENTITY]), "prediction KOI ordering/labels mismatch")
    require(prediction.partition.equals(aligned.partition), "prediction partitions changed")
    require(prediction.qc_pass.equals(objects.qc_pass), "prediction QC changed")
    require(prediction.fold_id.equals(aligned.fold_id), "prediction folds changed")
    require(np.array_equal(prediction.prediction, prediction.probability >= .5), "prediction threshold changed")
    expected_seen = objects.kepid.isin(set(objects.loc[aligned.partition.eq("train"), "kepid"]))
    require(prediction.host_seen_in_training.equals(expected_seen), "host diagnostic identity mismatch")


def baseline_comparisons(metrics):
    baseline = pd.read_csv("data/stage4b/metrics.csv")
    keys = ["strategy", "repeat_id", "partition", "track"]
    from kepler_baselines.metrics import METRICS
    output = []
    for family in ("random_forest", "logistic_regression"):
        joined = metrics.merge(baseline.loc[baseline.model.eq(family)], on=keys,
            suffixes=("_cnn", "_baseline"), validate="one_to_one")
        require(len(joined) == 120, "baseline metric pairing incomplete")
        for count in ("sample_count", "positive_count", "negative_count", "assigned_count", "qc_excluded_count"):
            require(joined[count + "_cnn"].equals(joined[count + "_baseline"]), "baseline denominator mismatch")
        for row in joined.to_dict("records"):
            for metric in METRICS:
                output.append({**{k: row[k] for k in keys}, "baseline": family, "metric": metric,
                    "cnn": row[metric + "_cnn"], "baseline_value": row[metric + "_baseline"],
                    "cnn_minus_baseline": row[metric + "_cnn"] - row[metric + "_baseline"]})
    frame = pd.DataFrame(output)
    summary = frame.groupby(["baseline", "strategy", "partition", "track", "metric"]).cnn_minus_baseline.agg(
        ["count", "mean", "median", "std", "min", "max"]).reset_index()
    summary["range"] = summary["max"] - summary["min"]
    return frame, summary


def checksums(metadata, output):
    for root in (Path(output), Path(metadata)):
        entries = [f"{sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*"))
                   if p.is_file() and p.name != "checksums.sha256" and "partial-" not in p.as_posix()]
        (root / "checksums.sha256").write_bytes("".join(entries).encode())
