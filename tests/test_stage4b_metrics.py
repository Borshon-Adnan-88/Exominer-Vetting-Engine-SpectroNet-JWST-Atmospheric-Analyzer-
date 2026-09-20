import numpy as np
import pandas as pd

from kepler_baselines.metrics import metrics, calibration_bins, evaluate_predictions, paired_and_summary


def test_fixed_half_threshold_and_confusion_orientation():
    result = metrics([0, 0, 1, 1], [.1, .5, .4, .9])
    assert [result[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["f1"] == .5
    assert np.isclose(result["brier"], (.01 + .25 + .36 + .01) / 4)


def test_empty_and_single_class_metrics_preserve_denominators():
    empty = metrics([], [])
    assert empty["sample_count"] == 0 and empty["roc_auc"] is None
    one = metrics([0, 0], [.1, .2])
    assert one["roc_auc"] is None and one["average_precision"] is None
    assert one["precision"] == 0 and one["precision_undefined"]


def test_calibration_edges_and_empty_bins():
    bins = calibration_bins([0, 1, 0, 1], [0., .1, .9, 1.])
    assert len(bins) == 10 and sum(x["sample_count"] for x in bins) == 4
    assert bins[0]["sample_count"] == bins[1]["sample_count"] == 1
    assert bins[9]["sample_count"] == 2
    assert bins[2]["observed_fraction"] is None


def test_qc_sensitivity_and_secondary_diagnostic_do_not_change_primary():
    prediction = pd.DataFrame({"partition": ["train"] * 2 + ["validation"] * 2 + ["test"] * 2,
                               "label": [0, 1] * 3, "probability": [.1, .9] * 3,
                               "qc_pass": [True, False] * 3, "host_seen_in_training": [True, True, False, False, True, False]})
    before = prediction.copy(deep=True)
    rows, bins, diagnostic = evaluate_predictions(prediction, "model", "object_stratified", 0)
    pd.testing.assert_frame_equal(prediction, before)
    primary = [r for r in rows if r["track"] == "primary"]
    sensitivity = [r for r in rows if r["track"] == "qc_pass"]
    assert all(r["sample_count"] == 2 and r["qc_excluded_count"] == 0 for r in primary)
    assert all(r["sample_count"] == 1 and r["assigned_count"] == 2 and r["qc_excluded_count"] == 1 for r in sensitivity)
    assert len(diagnostic) == 2 and all(r["diagnostic_only"] for r in diagnostic)
    assert len(bins) == 60
    prediction["host_seen_in_training"] = ~prediction.host_seen_in_training
    alternate, _, _ = evaluate_predictions(prediction, "model", "object_stratified", 0)
    assert rows == alternate  # Secondary grouping cannot change either reporting track.


def test_pairing_uses_repeat_id_and_preserves_brier_direction():
    rows = []
    for repeat in range(10):
        for strategy, probabilities in (("object_stratified", [.1, .9]), ("host_group_stratified", [.3, .7])):
            rows.append({"model": "model", "strategy": strategy, "repeat_id": repeat,
                         "partition": "test", "track": "primary", **metrics([0, 1], probabilities)})
    paired, summary, differences = paired_and_summary(pd.DataFrame(rows).sample(frac=1, random_state=5))
    brier = paired.loc[paired.metric.eq("brier")]
    assert len(brier) == 10 and np.allclose(brier.object_minus_grouped, -.08)
    assert brier.lower_is_better.all() and summary["count"].eq(10).all()
    assert differences["count"].eq(10).all()
