"""Fixed-threshold metrics, evaluation-only QC subsets and paired repeats."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc

METRICS = ("roc_auc", "average_precision", "pr_auc_trapezoidal", "precision", "recall", "f1", "accuracy", "brier")


def metrics(labels, probability, threshold=0.5):
    y, p = np.asarray(labels, dtype=int), np.asarray(probability, dtype=float)
    n = len(y)
    positive = int(y.sum())
    prediction = p >= threshold
    tp = int(((y == 1) & prediction).sum()); fp = int(((y == 0) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum()); fn = int(((y == 1) & ~prediction).sum())
    row = {"sample_count": n, "positive_count": positive, "negative_count": n - positive,
           "positive_prevalence": positive / n if n else None, "threshold": threshold,
           "tn": tn, "fp": fp, "fn": fn, "tp": tp,
           "precision": tp / (tp + fp) if tp + fp else 0.0,
           "recall": tp / (tp + fn) if tp + fn else 0.0,
           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
           "precision_undefined": tp + fp == 0, "recall_undefined": tp + fn == 0,
           "f1_undefined": 2 * tp + fp + fn == 0,
           "accuracy": (tp + tn) / n if n else None,
           "brier": float(np.mean((p - y) ** 2)) if n else None,
           "roc_auc": float(roc_auc_score(y, p)) if 0 < positive < n else None,
           "average_precision": float(average_precision_score(y, p)) if positive else None,
           "pr_auc_trapezoidal": None,
           "ranking_undefined_reason": "" if 0 < positive < n else "empty_subset" if not n else "single_class_subset"}
    if positive:
        precision, recall, _ = precision_recall_curve(y, p)
        row["pr_auc_trapezoidal"] = float(auc(recall, precision))
    return row


def calibration_bins(labels, probability, bins=10):
    y, p = np.asarray(labels), np.asarray(probability)
    membership = np.minimum((p * bins).astype(int), bins - 1)
    output = []
    for bin_id in range(bins):
        mask = membership == bin_id
        output.append({"bin_id": bin_id, "lower": bin_id / bins, "upper": (bin_id + 1) / bins,
                       "sample_count": int(mask.sum()), "positive_count": int(y[mask].sum()),
                       "mean_probability": float(p[mask].mean()) if mask.any() else None,
                       "observed_fraction": float(y[mask].mean()) if mask.any() else None})
    return output


def evaluate_predictions(predictions, family, strategy, repeat):
    rows, bins = [], []
    base = {"model": family, "strategy": strategy, "repeat_id": repeat}
    for partition in ("train", "validation", "test"):
        assigned = predictions.loc[predictions.partition.eq(partition)]
        for track in ("primary", "qc_pass"):
            evaluated = assigned if track == "primary" else assigned.loc[assigned.qc_pass]
            keys = {**base, "partition": partition, "track": track}
            rows.append({**keys, "assigned_count": len(assigned), "qc_excluded_count": len(assigned) - len(evaluated),
                         **metrics(evaluated.label, evaluated.probability)})
            bins.extend({**keys, **row} for row in calibration_bins(evaluated.label, evaluated.probability))
    diagnostics = []
    if strategy == "object_stratified":
        test = predictions.loc[predictions.partition.eq("test")]
        for seen in (True, False):
            subset = test.loc[test.host_seen_in_training.eq(seen)]
            diagnostics.append({**base, "partition": "test", "diagnostic_only": True,
                                "host_group": "seen_in_training" if seen else "unseen_in_training",
                                "assigned_test_count": len(test), **metrics(subset.label, subset.probability)})
    return rows, bins, diagnostics


def paired_and_summary(frame):
    keys = ["model", "repeat_id", "partition", "track"]
    objects = frame.loc[frame.strategy.eq("object_stratified"), keys + list(METRICS)]
    grouped = frame.loc[frame.strategy.eq("host_group_stratified"), keys + list(METRICS)]
    merged = objects.merge(grouped, on=keys, validate="one_to_one", suffixes=("_object", "_grouped"))
    paired = []
    for row in merged.to_dict("records"):
        for metric in METRICS:
            left, right = row[metric + "_object"], row[metric + "_grouped"]
            paired.append({**{key: row[key] for key in keys}, "metric": metric,
                           "object": left, "grouped": right, "object_minus_grouped": left - right if pd.notna(left) and pd.notna(right) else np.nan,
                           "lower_is_better": metric == "brier"})
    paired = pd.DataFrame(paired)
    long = frame.melt(id_vars=["model", "strategy", "repeat_id", "partition", "track"], value_vars=list(METRICS), var_name="metric", value_name="value")
    summary = long.groupby(["model", "strategy", "partition", "track", "metric"]).value.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    differences = paired.groupby(["model", "partition", "track", "metric"]).object_minus_grouped.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    summary["range"] = summary["max"] - summary["min"]
    differences["range"] = differences["max"] - differences["min"]
    return paired, summary, differences
