"""Build aggregate figures and report from frozen predictions; never fit models."""
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from kepler_baselines.inputs import require, sha256
from kepler_baselines.runner import write_csv, write_json
from kepler_cnn.artifacts import checksums, verify_baselines

META = Path("data/stage5")
ROOT = Path("data/stage5_runtime/cnn_v1")
STRATEGIES = ["object_stratified", "host_group_stratified"]


def table(frame):
    def cell(x):
        return f"{x:.8g}" if isinstance(x, float) else str(x)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)) + "\n"


def main():
    started = time.perf_counter()
    config = json.loads(Path("configs/kepler_stage5_cnn_v1.json").read_text())
    verify_baselines(config)
    provenance = json.loads((META / "provenance.json").read_text())
    require(provenance["status"] == "complete_verified", "fresh replay must pass before final report")
    records = json.loads((META / "run_manifest.json").read_text())["runs"]
    m = pd.read_csv(META / "metrics.csv"); b = pd.read_csv("data/stage4b/metrics.csv")
    summary = pd.read_csv(META / "repeat_summary.csv")
    paired = pd.read_csv(META / "paired_summary.csv")
    compare = pd.read_csv(META / "baseline_comparison_summary.csv")
    bins = pd.read_csv(META / "calibration_bins.csv")
    figroot = META / "figures"; figroot.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(figroot / name, dpi=140); plt.close(fig)
    # One curve per repeat, same colour per model. No pooled observations or CIs.
    for kind in ("roc", "precision_recall"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, strategy in zip(axes, STRATEGIES):
            for family, color in (("dual_view_cnn", "C0"), ("random_forest", "C1"), ("logistic_regression", "C2")):
                for repeat in range(10):
                    path = ROOT / strategy / f"repeat-{repeat:02d}" / "predictions.csv" if family == "dual_view_cnn" else Path(config["baseline_runtime"]) / "predictions" / family / strategy / f"repeat-{repeat:02d}.csv"
                    p = pd.read_csv(path, float_precision="round_trip"); p = p[p.partition.eq("test")]
                    if kind == "roc":
                        x, y, _ = roc_curve(p.label, p.probability)
                    else:
                        y, x, _ = precision_recall_curve(p.label, p.probability)
                    ax.plot(x, y, color=color, alpha=.35, linewidth=.8, label=family if repeat == 0 else None)
            ax.set(title=strategy, xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate" if kind == "roc" else "Recall", ylabel="True positive rate" if kind == "roc" else "Precision")
            ax.legend(fontsize=7)
        fig.suptitle("Test curves: every repeat, no pooling")
        save(fig, kind + "_comparison.png")
    test = m[m.partition.eq("test") & m.track.eq("primary")]
    anomaly_rows = []
    for record in records:
        prediction = pd.read_csv(ROOT / record["run_path"] / "predictions.csv", float_precision="round_trip")
        anomaly_rows.append({"strategy": record["identity"]["strategy"], "repeat_id": record["identity"]["repeat_id"],
            "unique_probabilities": int(prediction.probability.nunique()),
            "minimum_probability": float(prediction.probability.min()), "maximum_probability": float(prediction.probability.max()),
            "constant_predictions": prediction.probability.nunique() == 1,
            "probability_range": float(prediction.probability.max() - prediction.probability.min()),
            "at_most_three_probability_values": prediction.probability.nunique() <= 3,
            "range_at_most_1e_minus_6": float(prediction.probability.max() - prediction.probability.min()) <= 1e-6,
            "epoch_cap_reached": record["outcome"]["epoch_cap_reached"]})
    anomalies = pd.DataFrame(anomaly_rows)
    write_csv(META / "prediction_diagnostics.csv", anomalies)
    primary_means = pd.concat([test, b[b.partition.eq("test") & b.track.eq("primary")]]).groupby(
        ["strategy", "model"])[["roc_auc", "average_precision", "brier"]].mean().reset_index()
    all_comparisons = pd.read_csv(META / "baseline_comparisons.csv")
    auc_comparisons = all_comparisons[all_comparisons.partition.eq("test") & all_comparisons.track.eq("primary") & all_comparisons.metric.eq("roc_auc")]
    comparison_outcomes = []
    for family, rows in auc_comparisons.groupby("baseline"):
        comparison_outcomes.append(f"Against {family}, CNN test ROC-AUC is higher in {int((rows.cnn_minus_baseline > 0).sum())}/20 matched runs, lower in {int((rows.cnn_minus_baseline < 0).sum())}/20, and tied in {int((rows.cnn_minus_baseline == 0).sum())}/20.")
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for _, rows in test.groupby("repeat_id"):
        ax.plot([0, 1], rows.set_index("strategy").loc[STRATEGIES, "roc_auc"], "o-", alpha=.65)
    ax.set_xticks([0, 1], ["Object", "Host grouped"]); ax.set_ylabel("Test ROC-AUC")
    ax.set_title("CNN: ten paired repeats"); save(fig, "cnn_split_pairs.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, strategy in zip(axes, STRATEGIES):
        rf = b[b.model.eq("random_forest") & b.strategy.eq(strategy) & b.partition.eq("test") & b.track.eq("primary")].set_index("repeat_id")
        cnn = test[test.strategy.eq(strategy)].set_index("repeat_id")
        for repeat in range(10):
            ax.plot([0, 1], [rf.loc[repeat, "roc_auc"], cnn.loc[repeat, "roc_auc"]], "o-", alpha=.65)
        ax.set_xticks([0, 1], ["Random Forest", "CNN"]); ax.set(title=strategy, ylabel="Test ROC-AUC")
    save(fig, "cnn_rf_pairs.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, strategy in zip(axes, STRATEGIES):
        for _, rows in bins[bins.strategy.eq(strategy) & bins.partition.eq("test") & bins.track.eq("primary")].groupby("repeat_id"):
            ax.plot(rows.mean_probability, rows.observed_fraction, "o-", alpha=.4, markersize=3)
        ax.plot([0, 1], [0, 1], "k--"); ax.set(title=strategy, xlabel="Mean probability", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
    save(fig, "calibration.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for record in records:
        history = pd.read_csv(ROOT / record["run_path"] / "history.csv")
        col = STRATEGIES.index(record["identity"]["strategy"])
        for row, key in enumerate(("loss", "roc_auc")):
            axes[row, col].plot(history.epoch, history["training_" + key], color="C0", alpha=.3)
            axes[row, col].plot(history.epoch, history["validation_" + key], color="C1", alpha=.3)
            axes[row, col].set(title=STRATEGIES[col], xlabel="Epoch", ylabel=key)
    fig.suptitle("Training blue / validation orange; all repeats; no test histories")
    save(fig, "learning_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, strategy in zip(axes, STRATEGIES):
        means = test[test.strategy.eq(strategy)][["tn", "fp", "fn", "tp"]].mean().to_numpy().reshape(2, 2)
        ax.imshow(means, cmap="Blues")
        for (i, j), value in np.ndenumerate(means):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title=strategy)
    fig.suptitle("Mean test confusion counts per repeat; not pooled unique KOIs")
    save(fig, "confusion_summary.png")
    # Direct paired split-sensitivity contrast with each baseline, preserving signs.
    differences = pd.read_csv(META / "paired_differences.csv")
    bd = pd.read_csv("data/stage4b/paired_differences.csv")
    keys = ["repeat_id", "partition", "track", "metric"]
    joined = differences.merge(bd, on=keys, suffixes=("_cnn", "_baseline"), validate="one_to_many")
    joined["cnn_minus_baseline_split_difference"] = joined.object_minus_grouped_cnn - joined.object_minus_grouped_baseline
    joined["cnn_minus_baseline_absolute_split_difference"] = joined.object_minus_grouped_cnn.abs() - joined.object_minus_grouped_baseline.abs()
    write_csv(META / "split_sensitivity_comparison.csv", joined)
    runtime = json.loads((META / "runtime.json").read_text())
    replay = json.loads((META / "prediction_verification.json").read_text())
    audit_path = Path("data/stage5_infrastructure/post_training_input_audit.json")
    post_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if post_audit is not None:
        require(post_audit["status"] == "passed" and post_audit["canonical_npz_sha256_reverified"] == 6637,
                "post-training canonical audit did not pass")
        provenance["post_training_input_audit_sha256"] = sha256(audit_path)
        runtime["post_training_input_audit_seconds"] = post_audit["runtime_seconds"]
    outcomes = pd.DataFrame([{**r["identity"], "seed":r["seed"], "fit_seconds":r["fit_seconds"], "model_bytes":r["model_bytes"], **r["outcome"]} for r in records])
    write_csv(META / "run_outcomes.csv", outcomes)
    previous_reporting = runtime.get("reporting_seconds_total", runtime.get("reporting_seconds", 0))
    runtime["reporting_seconds"] = time.perf_counter() - started
    runtime["reporting_seconds_total"] = previous_reporting + runtime["reporting_seconds"]
    runtime["total_scientific_execution_seconds"] = runtime["invocation_seconds"] + replay["runtime_seconds"] + runtime["reporting_seconds_total"]
    if post_audit is not None:
        runtime["total_timed_stages_including_post_audit_seconds"] = runtime["total_scientific_execution_seconds"] + post_audit["runtime_seconds"]
    runtime["total_runtime_output_bytes"] = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    runtime["size_scope"] = "Runtime output directory including checksums; compact metadata size reported separately in report; environment excluded."
    write_json(META / "runtime.json", runtime)
    junit = ET.parse("data/tmp/stage5-full.xml").getroot()
    suites = list(junit.iter("testsuite"))
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    require(failures == 0, "offline test failures remain")
    subprocess.run(["git", "diff", "--check"], check=True)
    tests = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    provenance["report_generator_sha256"] = sha256(__file__)
    provenance["environment_lock_sha256"] = sha256("configs/kepler_stage5_environment_lock.txt")
    provenance["infrastructure_proof_sha256"] = sha256("data/stage5_infrastructure/passed.json")
    provenance["offline_tests"] = {"passed": tests - skipped, "skipped": skipped, "failures_or_errors": failures,
        "network_tests_deselected": 3, "junit_sha256": sha256("data/tmp/stage5-full.xml"),
        "cnn_test_source_sha256": sha256("tests/test_stage5_cnn.py"),
        "suite_seconds": sum(float(s.get("time", 0)) for s in suites)}
    write_json(META / "provenance.json", provenance)
    checksums(META, ROOT)
    sections = ["# Stage 5 dual-view CNN report\n",
        "Status: **complete; 20 scientific fits and fresh-process replay verified**. No attribution, tuning, TESS or app integration.\n",
        "Frozen validation passed: **6,637 KOIs / 5,804 hosts; 6,354 QC passes and 283 QC failures**. All canonical NPZ and 39,822 logical hashes passed. Stage 1 labels, Stage 3 assignments, Stage 4A arrays and Stage 4B artifacts remain unchanged.\n",
        ("A second read-only audit after training reverified all **6,637 canonical NPZ byte hashes**, checked missing/orphan membership, and rechecked frozen metadata and Stage 4B artifacts. [Post-training audit](../data/stage5_infrastructure/post_training_input_audit.json).\n" if post_audit is not None else ""),
        "## Architecture and environment\n",
        "Separate global (2000,1) and local (200,1) flux-only branches. Each: Conv16 k5 → pool → Conv32 k5 → pool → flatten. Global pools5 produce2560; local pools2 produce1600. Concatenate4160 → Dense32 ReLU → Dropout0.2 → sigmoid. **138,561 parameters verified programmatically**. No new normalization, augmentation or imputation.\n",
        "```json\n" + json.dumps(provenance["context"], indent=2) + "\n```\n",
        "Training uses the approved fixed Adam/BCE configuration in [config](../configs/kepler_stage5_cnn_v1.json). Only training rows receive gradients. Validation loss selects the earliest best epoch; test rows are absent from the fitting API. End-of-epoch histories use inference mode with dropout disabled. Optimizer slots reflect the stopping epoch; saved weights reflect the best epoch, and within-fit resume is not supported.\n",
        "## Determinism\n",
        "Paired strategies have identical initial-weight hashes. Synthetic training was reproduced exactly; all scientific saved models were reloaded and replayed in a fresh process. Probability tolerance is absolute1e-7, relative0; labels and KOI ordering must match exactly. No independent scientific retraining or cross-platform identity claim. CPU sequential; one numerical/intra/inter-op thread; deterministic operations; oneDNN disabled.\n",
        "```json\n" + json.dumps(replay, indent=2) + "\n```\n",
        "## All run outcomes\n", table(outcomes.drop(columns=["optimizer_state_note"])),
        "Epoch summaries:\n", table(outcomes[["best_epoch", "stopping_epoch"]].describe().reset_index()),
        "## Measured runtime and memory\n", "```json\n" + json.dumps(runtime, indent=2) + "\n```\n",
        "Runtime excludes dependency installation, library import before the runner timer, and tests. Total timed stages sums the measured training command, fresh replay, final report generation and (where present) the separate post-training input audit; it is not an end-to-end wall-clock measurement including gaps and tests. Peak RAM is training-process RSS, including TensorFlow and input arrays. Epoch cap is a recorded outcome, not convergence.\n",
        "## Primary comparison: mean test performance across all ten repeats\n", table(primary_means),
        "These means include every scheduled run, including runs with little probability variation. Random Forest is the principal comparator. Model selection and epoch selection did not use these test results.\n",
        "\n\n".join(comparison_outcomes) + "\n",
        "## Primary test metrics: every repeat\n", table(test),
        "## Run anomalies\n", table(anomalies),
        f"Exactly constant probabilities occur in {int(anomalies.constant_predictions.sum())}/20 runs; {int(anomalies.at_most_three_probability_values.sum())}/20 have only two or three distinct values, and {int(anomalies.range_at_most_1e_minus_6.sum())}/20 have a full-cohort probability range no greater than 1e-6. These descriptive flags were added for anomaly reporting after fitting; they do not exclude runs or change predictions, metrics or thresholds. No corrective refits or seed selection were performed. ROC-AUC in nearly tied outputs can be sensitive to minute numerical differences; the saved fixed-batch results are preserved exactly rather than rounded or repaired. The mechanism of limited discrimination has not been established. Differences between split strategies can reflect optimization outcomes as well as cohort composition and must not automatically be attributed to host leakage.\n",
        "## Primary test repeat summaries\n", table(summary[summary.partition.eq("test") & summary.track.eq("primary")]),
        "## CNN object minus grouped: paired results\n", table(paired[paired.partition.eq("test") & paired.track.eq("primary")]),
        "## CNN minus each baseline\n", table(compare[compare.partition.eq("test") & compare.track.eq("primary")]),
        "Per-repeat differences are in [baseline_comparisons.csv](../data/stage5/baseline_comparisons.csv). Random Forest is the principal comparator. Negative Brier differences favour the CNN. Signed and absolute split-sensitivity contrasts are in [split_sensitivity_comparison.csv](../data/stage5/split_sensitivity_comparison.csv).\n",
        "## QC-pass sensitivity\n", table(summary[summary.partition.eq("test") & summary.track.eq("qc_pass")]),
        "Identical fitted models and probabilities; no QC-only refitting. Denominators/exclusions remain in metrics.csv.\n",
        "## Calibration and secondary diagnostics\n",
        "Brier scores above and [ten-bin calibration table](../data/stage5/calibration_bins.csv) use unchanged Stage 4B definitions. No calibrator or threshold optimization. Host seen/unseen results remain [secondary only](../data/stage5/host_diagnostic.csv).\n",
        "## Figures\n"]
    sections += [f"![{p.stem}](../data/stage5/figures/{p.name})\n" for p in sorted(figroot.glob("*.png"))]
    sections += ["## Interpretation limits\n",
        "All ten repeats are retained. Repeats overlap; summaries are descriptive and no independence-based significance claims are made. Differences compare split protocols, not a causal leakage effect. Better prediction alone cannot prove morphology learning; attribution and controlled morphology analysis remain deferred to separately reviewed Stage 5A/5.1.\n",
        "## Verification, warnings and hashes\n",
        f"Offline suite: **{tests - skipped} passed, {skipped} skipped**, no errors/failures; network tests excluded. `git diff --check` passed. Epoch caps: {int(outcomes.epoch_cap_reached.sum())}/20. No non-finite or integrity/replay failures occurred in scientific fitting.\n",
        table(pd.DataFrame([{"artifact":p.relative_to(META).as_posix(), "sha256":sha256(p)} for p in sorted(META.rglob("*")) if p.is_file()])),
        "Every model/prediction/history/manifest hash is in the runtime [checksum manifest](../data/stage5_runtime/cnn_v1/checksums.sha256); per-run manifests also record initial-weight hashes. Infrastructure emitted dependency DeprecationWarnings from gast AST annotation handling and Keras NumPy conversion; these did not fail the tests. TensorFlow also emitted a reset_default_graph deprecation notice and a native-Windows GPU-unavailable notice; this experiment intentionally uses CPU.\n",
        f"Total Stage 5 artifact bytes (runtime + compact metadata, excluding this report and environment): {sum(p.stat().st_size for root in (ROOT,META) for p in root.rglob('*') if p.is_file())}.\n",
        "## Files created/modified\n",
        "Created: configs/kepler_stage5_cnn_v1.json, configs/kepler_stage5_environment_lock.txt, kepler_cnn/__init__.py, kepler_cnn/model.py, kepler_cnn/artifacts.py, kepler_cnn/runner.py, scripts/stage5_run_cnn.py, scripts/stage5_report.py, scripts/stage5_finish.ps1, tests/test_stage5_cnn.py, docs/kepler_stage5_cnn.md, docs/kepler_stage5_report.md, data/stage5_infrastructure/passed.json, data/stage5_infrastructure/post_training_input_audit.json, and Stage 5 artifact files listed individually in the two checksum manifests. Modified: .gitignore, pyproject.toml, README.md. Stage 4B changes already present before this task were retained. Isolated environment/bootstrap/test scratch are ignored operational files.\n",
        "No automatic commit. Stop for review.\n"]
    Path("docs/kepler_stage5_report.md").write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
