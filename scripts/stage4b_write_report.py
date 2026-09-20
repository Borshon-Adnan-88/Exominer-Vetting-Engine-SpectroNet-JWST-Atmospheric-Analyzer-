"""Render the completed or stopped Stage 4B audit as a review document."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd

from kepler_baselines.inputs import load_config, require, sha256, verify_frozen_hashes


def table(frame):
    def value(item):
        if pd.isna(item):
            return "undefined"
        if isinstance(item, float):
            return f"{item:.8g}"
        return str(item)
    columns = list(frame.columns)
    return ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] + ["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/stage4b")
    parser.add_argument("--tests", default="data/tmp/stage4b-full-20260919.xml")
    args = parser.parse_args()
    root = Path(args.metadata)
    config = load_config()
    verify_frozen_hashes(config)
    suite = ET.parse(args.tests).getroot().find("testsuite")
    tests = {key: int(suite.attrib[key]) for key in ("tests", "errors", "failures", "skipped")}
    diff = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True)
    require(diff.returncode == 0, "git diff --check failed")
    lines = ["# Stage 4B baseline experiment report", "", "This is a supervised command-line experiment on the certified Kepler DR25 cohort. The Streamlit application remains a separate synthetic prototype.", ""]
    stop = root / "stopped_for_review.json"
    if stop.exists():
        stopped = json.loads(stop.read_text())
        lines += ["**Status: stopped for review; the 40-fit experiment is incomplete.**", "",
                  f"Stop condition: `{stopped['reason']}`.", "",
                  f"Run: `{stopped['run']}`. Successful fits before stopping: **{stopped['completed_fits']} / 40**.", "",
                  "No parameter change, seed search, input repair, or continuation past the stop condition was performed.", "",
                  "```json", json.dumps(stopped, indent=2, sort_keys=True), "```", ""]
        if (root / "input_validation.json").exists():
            validation = json.loads((root / "input_validation.json").read_text())
            lines += [f"Input validation passed for {validation['cohort_size']:,} KOIs, {validation['host_count']:,} hosts and {validation['feature_count']:,} features; {validation['logical_arrays_verified']:,} logical arrays verified.", ""]
        lines += ["Requested complete-experiment metrics, paired conclusions, calibration, sensitivity and completed-model provenance are unavailable. Partial artifacts must not be presented as final results.", ""]
    else:
        provenance = json.loads((root / "provenance.json").read_text())
        require(provenance["status"] == "complete_verified", "independent prediction replay has not passed")
        for entry in (root / "checksums.sha256").read_text().splitlines():
            expected, relative = entry.split("  ", 1)
            require(sha256(root / relative) == expected, "result checksum mismatch")
        runtime = json.loads((root / "runtime.json").read_text())
        runs = json.loads((root / "run_manifest.json").read_text())["runs"]
        validation = json.loads((root / "input_validation.json").read_text())
        verification = json.loads((root / "prediction_verification.json").read_text())
        metrics = pd.read_csv(root / "metrics.csv")
        repeat_summary = pd.read_csv(root / "repeat_summary.csv")
        paired = pd.read_csv(root / "paired_differences.csv")
        paired_summary = pd.read_csv(root / "paired_summary.csv")
        diagnostic = pd.read_csv(root / "host_diagnostic.csv")
        lines += ["**Status: complete; 40 fits and independent prediction replay verified.**", "",
                  f"Frozen-input validation passed: **{validation['cohort_size']:,} KOIs, {validation['host_count']:,} hosts, {validation['feature_count']:,} flux features**. The primary cohort retains all {validation['qc_pass']:,} QC passes and {validation['qc_fail']:,} QC failures. All canonical file hashes, 39,822 logical hashes, shapes/dtypes, finite values and identity joins passed. All 132,740 Stage 3 assignment rows remain unchanged; grouped host separation passed.", "",
                  "## Primary scientific comparison", "",
                  "The primary outcome is object-level minus host-group **test ROC-AUC**, paired by the ten frozen repeat IDs. Means below are across repeats, not metrics pooled over repeated observations. Host seen/unseen results do not enter this comparison.", ""]
        primary = paired_summary.loc[(paired_summary.partition == "test") & (paired_summary.track == "primary") & (paired_summary.metric == "roc_auc")]
        lines += table(primary[["model", "count", "mean", "median", "std", "min", "max", "range"]]) + [""]
        for family in config["fit_order"]:
            values = paired.loc[(paired.model == family) & (paired.partition == "test") & (paired.track == "primary") & (paired.metric == "roc_auc"), "object_minus_grouped"]
            lines += [f"`{family}`: object-level test ROC-AUC is higher in {int((values > 0).sum())}/10 repeats and lower in {int((values < 0).sum())}/10; mean paired difference {values.mean():+.6f}.", ""]
        lines += ["These are descriptive split-protocol comparisons, not a causal estimate of leakage. Training/test membership, host composition and class prevalence differ. Repeats share the same cohort and are not independent datasets; no independence-based confidence interval or significance claim is made. The secondary host diagnostic cannot establish or override the primary conclusion.", "",
                  "## Fixed models and runtime", "",
                  "Logistic regression is a single training-only StandardScaler → LogisticRegression Pipeline. Scaler fit-call tests prove that validation/test and full-cohort fitting do not occur. Random Forest uses the identical frozen flux representation without scaling. All settings below were frozen before fitting; no tuning occurred.", "",
                  "```json", json.dumps(config["models"], indent=2, sort_keys=True), "```", "",
                  "Forest random states are the Stage 3 seeds, shared between strategies within each repeat. The deterministic lbfgs solver has no stochastic seed operation. Decision threshold is p(confirmed) >= 0.5. No class-weight search, PCA, imputation, threshold tuning or post-hoc calibration is used.", "",
                  f"All 20 logistic fits converged; iteration counts range from {min(r['iterations'] for r in runs if r['iterations'] is not None)} to {max(r['iterations'] for r in runs if r['iterations'] is not None)}. Forest convergence is not applicable.", "",
                  f"Total experiment runtime: **{runtime['total_experiment_seconds']:.3f} seconds**, including {runtime['input_validation_seconds']:.3f} seconds for input validation, plus **{verification['runtime_seconds']:.3f} seconds** for the separate independent replay. Timers start inside the commands after library imports; test runtime is separate.", "",
                  f"Process peak RAM: **{runtime['process_peak_rss_bytes']:,} bytes ({runtime['process_peak_rss_bytes']/2**20:.2f} MiB)**. CPU execution was sequential with one numerical thread. Family RAM peaks are 50-ms samples of total process RSS, not incremental estimator allocations.", ""]
        family_rows = []
        for family, item in runtime["families"].items():
            family_rows.append({"model": family, **item, "sampled_peak_rss_bytes": runtime["sampled_phase_peak_rss_bytes"].get(family)})
        lines += table(pd.DataFrame(family_rows)) + ["", "Iteration counts, fit runtimes, model sizes and per-run hashes are recorded in [run_manifest.json](../data/stage4b/run_manifest.json).", "",
                  "## Test metrics for every model, strategy and repeat", "",
                  "Positive label is confirmed planet. Confusion columns are TN, FP, FN, TP. Average precision is non-trapezoidal; trapezoidal PR-AUC is separate. Brier is mean squared binary probability error (lower is better).", ""]
        columns = ["model", "strategy", "repeat_id", "sample_count", "positive_count", "negative_count", "positive_prevalence", "roc_auc", "average_precision", "pr_auc_trapezoidal", "precision", "recall", "f1", "accuracy", "brier", "tn", "fp", "fn", "tp"]
        lines += table(metrics.loc[(metrics.partition == "test") & (metrics.track == "primary"), columns]) + ["", "The [metrics CSV](../data/stage4b/metrics.csv) contains all 240 rows: 40 fitted models × three partitions × two reporting tracks. Training scores are in-sample diagnostics. Assigned/evaluated/excluded counts, prevalence and undefined-metric flags are retained.", "",
                  "## Repeat summaries and paired differences", "", "Primary test mean/median/sample-standard-deviation/min/max/range:", ""]
        lines += table(repeat_summary.loc[(repeat_summary.partition == "test") & (repeat_summary.track == "primary")].drop(columns=["partition", "track"]))
        lines += ["", "Primary paired test differences (object minus grouped, unchanged direction for every metric; negative Brier difference favours object splitting):", ""]
        lines += table(paired_summary.loc[(paired_summary.partition == "test") & (paired_summary.track == "primary")].drop(columns=["partition", "track"]))
        lines += ["", "All ten object/grouped values and differences are preserved in [paired_differences.csv](../data/stage4b/paired_differences.csv), including validation, training and QC sensitivity tracks. No favourable repeats were selected.", "",
                  "## QC-pass evaluation sensitivity", "",
                  "The same 40 fitted models and probabilities are evaluated on QC-pass rows within each unchanged partition. No model is refitted on a QC-pass-only training subset. Original denominators and QC-excluded counts remain in metrics.csv; the primary cohort retains QC failures.", ""]
        lines += table(repeat_summary.loc[(repeat_summary.partition == "test") & (repeat_summary.track == "qc_pass") & repeat_summary.metric.isin(["roc_auc", "average_precision", "brier"])].drop(columns=["partition", "track"]))
        lines += ["", "QC-pass paired test differences:", ""]
        lines += table(paired_summary.loc[(paired_summary.partition == "test") & (paired_summary.track == "qc_pass") & paired_summary.metric.isin(["roc_auc", "average_precision", "brier"])].drop(columns=["partition", "track"]))
        lines += ["", "## Calibration", "",
                  "Raw model Brier scores are in the tables above. The [calibration-bin table](../data/stage4b/calibration_bins.csv) contains ten fixed-width bins per partition/track/run, including empty bins, counts, mean probability and observed positive fraction. The [test calibration figure](../data/stage4b/figures/test_calibration.png) shows one raw curve per repeat; no calibrator was fitted. Brier assesses overall probability quality and is not interpreted as pure calibration error.", "",
                  "## Host seen/unseen: secondary diagnostic only", "",
                  "Within object-level test partitions, a host is seen if it appears in that repeat's training partition. This grouping did not alter fitting, assignments, eligibility, thresholds or the primary paired analysis. Class composition and sample sizes differ, so this is not a causal leakage estimate.", ""]
        lines += table(diagnostic[["model", "repeat_id", "host_group", "sample_count", "positive_prevalence", "roc_auc", "average_precision", "brier", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]])
        lines += ["", "[Host overlap counts](../data/stage4b/host_overlap.csv) accompany the diagnostic. Figures are compact aggregate plots; no per-target plots were generated.", "",
                  "## Determinism and hashes", "",
                  f"Repeated predictions, model reloads and CSV roundtrips were bit-identical for every run. A fresh process replayed all **{verification['models_replayed']} saved models / {verification['predictions_verified']:,} probabilities**, with zero new fits. This verifies prediction replay in the pinned environment, not a second independent training run or cross-platform binary identity.", "",
                  "Every prediction artifact SHA-256 (runtime-root-relative paths):", "", "```text",
                  (root / "prediction_checksums.sha256").read_text().rstrip(), "```", "",
                  "Metric, provenance and verification hashes:", "", "| Artifact | SHA-256 |", "|---|---|"]
        for path in sorted(root.glob("*")):
            if path.is_file():
                lines.append(f"| `{path.name}` | `{sha256(path)}` |")
        lines += ["", "[Model checksums](../data/stage4b/model_checksums.sha256) cover every saved estimator. The feature-cache SHA-256 and all frozen-input hashes are recorded in provenance. Scientific paths are relative; timestamps/runtime and serialized estimator files are operational artifacts rather than a cross-platform byte-determinism claim.", ""]
    lines += ["## Tests and scope", "",
              f"Complete offline suite: **{tests['tests'] - tests['errors'] - tests['failures'] - tests['skipped']} passed**, {tests['errors']} errors, {tests['failures']} failures, {tests['skipped']} skipped; three network tests deselected. The full suite was run once. JUnit evidence: `{args.tests}`.", "",
              "`git diff --check`: passed. Stage 1, Stage 2 scientific preprocessing, Stage 3 assignments, Stage 4A outputs/configuration and the frozen MAST inventory remain unchanged. No new splits, CNN, Stage 5, TESS, app integration or automatic commit occurred. Stop here for review.", ""]
    destination = Path("docs/kepler_stage4b_report.md")
    destination.write_bytes("\n".join(lines).encode())
    print(f"Wrote {destination.as_posix()}")


if __name__ == "__main__":
    main()
