"""Certify existing Stage 5 artifacts without TensorFlow, training or array reloads.

Run once from the repository root: python -m scripts.stage5_finalize
The existing model replay, offline suite and canonical NPZ audit are reused.
Only compact certification metadata and the report are written.
"""
import json
import re
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from kepler_baselines.inputs import load_config, read_input_tables, require, sha256
from kepler_baselines.metrics import evaluate_predictions, paired_and_summary
from kepler_baselines.runner import write_json
from kepler_cnn.artifacts import align, baseline_comparisons, check_prediction_identity, verify_baselines

META = Path("data/stage5")
ROOT = Path("data/stage5_runtime/cnn_v1")
REPORT = Path("docs/kepler_stage5_report.md")


def verify_manifest(path, root):
    entries = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        target = (root / name).resolve()
        require(target.is_relative_to(root.resolve()), "unsafe artifact path")
        require(name not in entries and target.is_file() and sha256(target) == digest,
                f"artifact hash mismatch: {target}")
        entries[name] = digest
    return entries


def equivalent(frame, name):
    saved = pd.read_csv(META / name, float_precision="round_trip")
    require(set(frame.columns) == set(saved.columns) and len(frame) == len(saved), f"table schema/count mismatch: {name}")
    for key in frame.columns:
        left, right = frame[key], saved[key]
        if pd.api.types.is_numeric_dtype(left):
            require(np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True), f"numeric mismatch: {name}:{key}")
        else:
            require(left.fillna("").equals(right.fillna("")), f"identity mismatch: {name}:{key}")


def refresh_metadata_checksums():
    files = [p for p in sorted(META.rglob("*")) if p.is_file() and p.name != "checksums.sha256"]
    (META / "checksums.sha256").write_bytes("".join(f"{sha256(p)}  {p.relative_to(META).as_posix()}\n" for p in files).encode())


def main():
    metadata_entries = verify_manifest(META / "checksums.sha256", META)
    runtime_entries = verify_manifest(ROOT / "checksums.sha256", ROOT)
    provenance = json.loads((META / "provenance.json").read_text())
    config = load_config("configs/kepler_stage5_cnn_v1.json")
    require(provenance["status"] == "complete_verified", "scientific execution is not verified")
    require(sha256("configs/kepler_stage5_cnn_v1.json") == provenance["context"]["config_sha256"], "configuration changed")
    for path, digest in provenance["context"]["source_sha256"].items():
        require(sha256(path) == digest, f"scientific source changed: {path}")
    verify_baselines(config)
    verify_manifest(Path("data/stage4b/model_checksums.sha256"), Path(config["baseline_runtime"]))
    objects, _, assignments = read_input_tables(config)
    records = json.loads((META / "run_manifest.json").read_text())["runs"]
    expected = {(strategy, repeat) for strategy in config["strategies"] for repeat in range(10)}
    require(len(records) == 20 and {(r["identity"]["strategy"], r["identity"]["repeat_id"]) for r in records} == expected,
            "scientific run matrix mismatch")
    require(len(runtime_entries) == 80 and len(list(ROOT.rglob("model.keras"))) == 20, "runtime artifact count mismatch")
    rows, bins, diagnostics, outcomes, anomalies = [], [], [], [], []
    initial = {}; expected_runtime = set()
    for record in records:
        directory = ROOT / record["run_path"]
        per_run = json.loads((directory / "manifest.json").read_text())
        require(per_run == {k: v for k, v in record.items() if k != "run_path"}, "aggregate/per-run manifest disagreement")
        require(record["context"] == provenance["context"] and record["parameters"] == 138561, "run context/architecture mismatch")
        strategy, repeat = record["identity"]["strategy"], record["identity"]["repeat_id"]
        require(record["identity"]["model_version"] == config["experiment_id"] and record["seed"] == config["seeds"][repeat], "version/seed changed")
        require(repeat not in initial or initial[repeat] == record["initial_weight_sha256"], "paired initial weights mismatch")
        initial[repeat] = record["initial_weight_sha256"]
        for name in ("manifest.json", "model.keras", "predictions.csv", "history.csv"):
            expected_runtime.add((directory / name).relative_to(ROOT).as_posix())
        for name, digest in record["sha256"].items():
            require(sha256(directory / name) == digest, "run artifact mismatch")
        require((directory / "model.keras").stat().st_size == record["model_bytes"], "model size mismatch")
        prediction = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
        check_prediction_identity(prediction, objects, align(objects, assignments, strategy, repeat))
        for field, value in (("model", "dual_view_cnn"), ("strategy", strategy), ("repeat_id", repeat), ("seed", record["seed"])):
            require(prediction[field].eq(value).all(), "prediction run identity mismatch")
        require(np.isfinite(prediction.probability).all() and prediction.probability.between(0, 1).all(), "invalid saved probabilities")
        a, b, c = evaluate_predictions(prediction, "dual_view_cnn", strategy, repeat)
        rows.extend(a); bins.extend(b); diagnostics.extend(c)
        history = pd.read_csv(directory / "history.csv", float_precision="round_trip")
        outcome = record["outcome"]
        require(np.isfinite(history.to_numpy()).all(), "non-finite history")
        require(history.epoch.tolist() == list(range(1, len(history) + 1)), "history epoch sequence mismatch")
        best = history.loc[history.validation_loss.idxmin()]
        require(int(best.epoch) == outcome["best_epoch"] and abs(best.validation_loss - outcome["best_validation_loss"]) <= 1e-12,
                "best validation checkpoint/history mismatch")
        require(len(history) == outcome["stopping_epoch"] == min(outcome["best_epoch"] + 10, 100), "early stopping/history mismatch")
        require(outcome["epoch_cap_reached"] == (len(history) == 100), "epoch cap flag mismatch")
        require(outcome["status"] == ("epoch_cap_reached" if len(history) == 100 else "early_stopped"), "stopping status mismatch")
        outcomes.append({**record["identity"], "seed": record["seed"], "fit_seconds": record["fit_seconds"], "model_bytes": record["model_bytes"], **outcome})
        anomalies.append({"strategy": strategy, "repeat_id": repeat, "distinct_probabilities": int(prediction.probability.nunique()),
                          "range": float(prediction.probability.max() - prediction.probability.min())})
    require(set(runtime_entries) == expected_runtime, "missing/orphan runtime artifacts")
    frame = pd.DataFrame(rows)
    paired, summary, differences = paired_and_summary(frame)
    comparisons, comparison_summary = baseline_comparisons(frame)
    for name, table in (("metrics.csv", frame), ("calibration_bins.csv", pd.DataFrame(bins)),
                       ("host_diagnostic.csv", pd.DataFrame(diagnostics)), ("run_outcomes.csv", pd.DataFrame(outcomes)),
                       ("paired_differences.csv", paired), ("repeat_summary.csv", summary), ("paired_summary.csv", differences),
                       ("baseline_comparisons.csv", comparisons), ("baseline_comparison_summary.csv", comparison_summary)):
        equivalent(table, name)
    replay = json.loads((META / "prediction_verification.json").read_text())
    require(sha256(META / "prediction_verification.json") == provenance["prediction_verification_sha256"], "replay evidence changed")
    require(replay["status"] == "passed" and replay["models"] == 20 and replay["predictions"] == 132740 and replay["maximum_probability_error"] == 0 and replay["new_fits"] == 0, "replay evidence mismatch")
    audit_path = Path("data/stage5_infrastructure/post_training_input_audit.json")
    audit = json.loads(audit_path.read_text())
    require(sha256(audit_path) == provenance["post_training_input_audit_sha256"] and audit["status"] == "passed" and audit["canonical_npz_sha256_reverified"] == 6637, "canonical audit evidence mismatch")
    require(sha256("data/tmp/stage5-full.xml") == provenance["offline_tests"]["junit_sha256"], "test evidence changed")
    suites = list(ET.parse("data/tmp/stage5-full.xml").getroot().iter("testsuite"))
    require(sum(int(s.get("tests", 0)) for s in suites) == 140 and all(int(s.get("failures", 0)) + int(s.get("errors", 0)) + int(s.get("skipped", 0)) == 0 for s in suites), "offline test evidence mismatch")
    subprocess.run(["git", "diff", "--check"], check=True)
    test = frame[frame.partition.eq("test") & frame.track.eq("primary")]
    split = differences[differences.partition.eq("test") & differences.track.eq("primary") & differences.metric.eq("roc_auc")].iloc[0]
    auc = comparisons[comparisons.partition.eq("test") & comparisons.track.eq("primary") & comparisons.metric.eq("roc_auc")]
    certification = {"status": "certified", "scope": "Existing-artifact consistency and bookkeeping only; no TensorFlow, training, model replay, tests or NPZ re-audit executed.",
        "scientific_fits": 20, "prediction_rows_verified": 132740, "metric_rows_recomputed": 120, "calibration_rows_recomputed": 1200,
        "histories_verified": 20, "runtime_artifacts_verified": 80, "comparison_rows_verified": len(comparisons),
        "frozen_metadata_and_stage4b_outputs": "hashes unchanged", "canonical_arrays": "prior passed post-training 6637-file hash audit reused",
        "test_evidence": "140 passed; 3 network tests deselected; suite not rerun", "replay_evidence": "20 models; 132740 probabilities; zero difference; replay not rerun",
        "primary_test_roc_auc_means": test.groupby("strategy").roc_auc.mean().to_dict(),
        "object_minus_grouped_test_roc_auc": float(split["mean"]),
        "lower_auc_than_baseline_counts": auc.groupby("baseline").cnn_minus_baseline.apply(lambda x: int((x < 0).sum())).to_dict(),
        "two_or_three_probability_values": sum(2 <= r["distinct_probabilities"] <= 3 for r in anomalies),
        "probability_range_below_1e_minus_7": sum(r["range"] < 1e-7 for r in anomalies),
        "epoch_caps": sum(r["outcome"]["epoch_cap_reached"] for r in records),
        "runs_excluded_or_corrected": 0, "corrective_reruns": 0, "git_diff_check": "passed",
        "finalizer_sha256": sha256(__file__), "prior_provenance_sha256": sha256(META / "provenance.json"),
        "existing_runtime_manifest_sha256": sha256(ROOT / "checksums.sha256")}
    require(certification["two_or_three_probability_values"] == 7 and certification["probability_range_below_1e_minus_7"] == 4 and certification["epoch_caps"] == 1, "known anomaly summary mismatch")
    write_json(META / "certification.json", certification)
    (META / "runtime_checksums.sha256").write_bytes((ROOT / "checksums.sha256").read_bytes())
    provenance["certification_status"] = "certified"
    provenance["certification_sha256"] = sha256(META / "certification.json")
    provenance["runtime_checksums_sha256"] = sha256(META / "runtime_checksums.sha256")
    provenance["finalizer_sha256"] = sha256(__file__)
    write_json(META / "provenance.json", provenance)
    refresh_metadata_checksums()
    content = REPORT.read_text(encoding="utf-8")
    content = content.replace("Status: **complete; 20 scientific fits and fresh-process replay verified**.", "Status: **certified; 20 scientific fits, replay, offline tests and final artifact consistency verified**.")
    # Existing scientific tables and figures remain unchanged; refresh the derived digest table.
    start = content.index("| artifact | sha256 |")
    end = content.index("Every model/prediction/history/manifest hash", start)
    hash_table = "| artifact | sha256 |\n| --- | --- |\n" + "".join(f"| {p.relative_to(META).as_posix()} | `{sha256(p)}` |\n" for p in sorted(META.rglob("*")) if p.is_file())
    content = content[:start] + hash_table + "\n" + content[end:]
    content = content.replace("4/20 have a full-cohort probability range no greater than 1e-6", "4/20 have a full-cohort probability range strictly below 1e-7 (also satisfying the existing 1e-6 diagnostic flag)")
    output_bytes = sum(p.stat().st_size for root in (META, ROOT) for p in root.rglob("*") if p.is_file())
    content = re.sub(r"(Total Stage 5 artifact bytes \(runtime \+ compact metadata, excluding this report and environment\): )\d+",
                     lambda match: match.group(1) + str(output_bytes), content)
    marker = "\n## Final certification and commit scope\n"
    content = content.split(marker)[0]
    content += marker + "\nThe finalizer checked all 20 histories and per-run manifests, all 132,740 saved prediction rows, 120 metric rows, 1,200 calibration rows and 1,920 baseline-comparison rows. Frozen metadata, baseline artifacts and recorded source hashes match. Existing passed model replay, offline tests and canonical-array audit were reused; no expensive work or scientific fit was repeated. See [certification.json](../data/stage5/certification.json).\n\n"
    content += "Files created in this finalization: `scripts/stage5_finalize.py`, `data/stage5/certification.json`, `data/stage5/runtime_checksums.sha256`. Files modified: `data/stage5/provenance.json`, `data/stage5/checksums.sha256`, `docs/kepler_stage5_report.md`. Scientific results, histories, models, predictions, figures, runtime measurements and scientific configuration were unchanged.\n\n"
    content += "Recommended commits: (1) certified Stage 4B implementation/results already present in the worktree, if not committed; (2) Stage 5 implementation/configuration/environment lock/tests and README/package/ignore integration; (3) Stage 5 compact results, infrastructure evidence, report and final certification. Split shared README/pyproject/.gitignore hunks by stage. Keep `.venv/`, bootstrap/test scratch and `data/stage5_runtime/` ignored. Commit the compact runtime checksum manifest; retain the actual model/prediction files separately at their current location. No commit was made. Stop for review.\n"
    REPORT.write_text(content, encoding="utf-8")
    # All previously existing result bytes except the explicitly refreshed provenance must be intact.
    for name, digest in metadata_entries.items():
        if name not in {"provenance.json", "certification.json", "runtime_checksums.sha256"}:
            require(sha256(META / name) == digest, f"scientific result changed during finalization: {name}")
    verify_manifest(META / "checksums.sha256", META)
    require(sha256(ROOT / "checksums.sha256") == certification["existing_runtime_manifest_sha256"], "runtime manifest changed")
    subprocess.run(["git", "diff", "--check"], check=True)
    print(json.dumps({"status": "certified", "scientific_fits_executed": 0,
        "hashes": {name: sha256(META / name) for name in ("metrics.csv", "provenance.json", "certification.json", "checksums.sha256", "runtime_checksums.sha256")},
        "report_sha256": sha256(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
