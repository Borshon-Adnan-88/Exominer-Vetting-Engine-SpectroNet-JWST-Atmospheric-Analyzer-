"""Offline certification of existing Stage 4A outputs; never downloads/processes."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import pandas as pd

from .arrays import KEYS
from .attrition import TECHNICAL_STATUSES, validate_outcomes
from .contracts import load_config, load_cohort, validate_inputs
from .index import canonical_relative_path, validate_unique_index

META = Path("data/stage4a")
PREFIX = "kepler_stage4a_"
FORBIDDEN = re.compile(r"[A-Za-z]:[\\/]|/content/drive|/scratch/|Users[/\\]|My Drive|OneDrive")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    return digest(Path(path).read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode()


def require_relative(name):
    require(isinstance(name, str) and not FORBIDDEN.search(name), "machine-specific path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name, "nonportable path")


def verify_npz(root, outcome, config):
    """Recompute file/logical hashes and deterministic ZIP bytes independently."""
    name = outcome["relative_path"]
    require_relative(name)
    require(name == canonical_relative_path(outcome["kepid"], outcome["kepoi_name"]), "noncanonical identity/path")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "canonical path escapes dataset")
    payload = path.read_bytes()
    require(digest(payload) == outcome["npz_sha256"], f"NPZ checksum mismatch: {name}")
    expected_logical = outcome["logical_array_hashes"]
    require(set(expected_logical) == set(KEYS), f"logical hash keys: {name}")
    rebuilt = io.BytesIO()
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays, zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_STORED) as archive:
        require(len(arrays.files) == len(KEYS) and set(arrays.files) == set(KEYS), f"array keys: {name}")
        for key in KEYS:
            value = arrays[key]
            view = "global_view" if key.startswith("global_") else "local_view"
            kind = "mask_dtype" if key.endswith("mask") else "count_dtype" if key.endswith("count") else "dtype"
            expected_dtype = np.dtype(config["output"][kind]).newbyteorder("<")
            require(value.shape == (config[view]["bins"],) and value.dtype.str == expected_dtype.str, f"shape/dtype: {name}:{key}")
            require(np.isfinite(value).all(), f"nonfinite array: {name}:{key}")
            header = f"{value.dtype.str}|{','.join(map(str, value.shape))}|".encode()
            require(digest(header + value.tobytes(order="C")) == expected_logical[key], f"logical hash: {name}:{key}")
            if key.endswith("count"):
                require((value >= 0).all(), f"negative counts: {name}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
        for view in ("global", "local"):
            require(np.array_equal(arrays[f"{view}_observed_mask"], arrays[f"{view}_count"] > 0), f"mask/count inconsistency: {name}")
            observed = float(arrays[f"{view}_observed_mask"].mean())
            require(abs(observed - outcome["processing_metadata"][f"{view}_observed_fraction"]) < 1e-12, f"coverage metadata mismatch: {name}")
    require(rebuilt.getvalue() == payload, f"noncanonical deterministic ZIP bytes: {name}")
    return len(payload)


def read_json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_listing(root, namespace):
    records = []
    require(root.exists(), f"audit root unavailable: {namespace}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.name.lower().endswith((".fits", ".fits.gz", ".part"))):
            relative = path.relative_to(root).as_posix()
            require_relative(relative)
            records.append({"root": namespace, "relative_path": relative, "bytes": path.stat().st_size,
                            "sha256": file_digest(path)})
    return records


def verify_existing(dataset: Path, scratch: Path):
    config = load_config()
    input_hashes = validate_inputs(config)
    cohort = load_cohort(config)
    scientific = read_json_file(Path(config["expected_inputs"]["stage2_config"]))
    pre = read_json_file(META / f"{PREFIX}provenance_v1.json")
    require(file_digest("configs/kepler_stage4a_scale_v1.json") == pre["stage4a_config_sha256"], "Stage 4A config changed")
    inventory_path = META / f"{PREFIX}mast_inventory_v1.csv"
    require(file_digest(inventory_path) == pre["mast_inventory_sha256"], "frozen inventory changed")
    for line in (META / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(file_digest(name) == expected, f"frozen metadata changed: {name}")
    evidence = read_json_file(META / f"{PREFIX}integrity_audit_v1.json")
    require(evidence["fresh_count"] == 8 and evidence["mast_inventory_sha256"] == pre["mast_inventory_sha256"], "missing bounded integrity evidence")
    for sample in evidence["fresh_samples"] + evidence["retained_sources"]:
        require(sample["sha256"] == sample["retrieval_record_sha256"] and sample["bytes"] == sample["advertised_bytes"], "sample byte mismatch")
        require(sample["fits_readable"] and all(h["raw_checksum_valid"] is True for h in sample["hdus"]), "sample checksum/structure invalid")
    roster = pd.read_csv(META / f"{PREFIX}host_roster_v1.csv")
    expected_hosts = set(int(x) for x in cohort.kepid)
    require(not roster.kepid.duplicated().any() and set(roster.kepid) == expected_hosts, "host roster mismatch")
    state_paths = sorted((dataset / "cache/host_state").glob("*.json"))
    expected_names = {f"{host:09d}.json" for host in expected_hosts}
    require({p.name for p in state_paths} == expected_names, "missing or extra host-state files")
    print(f"Reading {len(state_paths)} host states", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(pool.map(read_json_file, state_paths))
    host_ids = [state["kepid"] for state in states]
    require(len(host_ids) == len(set(host_ids)) and set(host_ids) == expected_hosts, "host-state identities")
    for path, state in zip(state_paths, states):
        require(path.name == f"{state['kepid']:09d}.json", "host-state filename mismatch")
        require(all(state.get(key) is True for key in ("inventory_frozen", "sizes_verified", "fits_sha256_recorded", "fits_identity_validated", "all_kois_final", "constructed_arrays_written", "array_checksums_verified", "provenance_written")), f"incomplete host: {state['kepid']}")
        require(all(x["kepid"] == state["kepid"] for x in state["koi_outcomes"]), "KOI stored under wrong host")
    outcomes = sorted([item for state in states for item in state["koi_outcomes"]], key=lambda x: (x["kepid"], x["kepoi_name"]))
    outcome_frame = pd.DataFrame(outcomes)
    validate_outcomes(outcome_frame, cohort)
    identity = ["source_row_index", "kepid", "kepoi_name", "label"]
    joined = cohort[identity].merge(outcome_frame[identity], on=identity, how="outer", indicator=True, validate="one_to_one")
    require(joined._merge.eq("both").all(), "source identity or label changed")
    for outcome in outcomes:
        if outcome["processing_status"] == "constructed":
            require(outcome["technical_status"] == ("constructed_qc_pass" if outcome["qc_pass"] else "constructed_qc_fail"), "QC status inconsistent")
            metadata = outcome["processing_metadata"]
            require(metadata["qc_pass"] == outcome["qc_pass"] and metadata["qc_reasons"] == outcome["qc_reasons"], "QC metadata inconsistent")
    reports_dir = dataset / "cache/benchmarks"
    report_paths = sorted(reports_dir.glob("*.json"))
    expected_reports = {f"{batch}-home_wifi.json" for batch in roster.batch_id.unique()}
    require({p.name for p in report_paths} == expected_reports, "missing or unexpected batch reports")
    with ThreadPoolExecutor(max_workers=8) as pool:
        reports = list(pool.map(read_json_file, report_paths))
    states_by_host = {s["kepid"]: s for s in states}
    for path, report in zip(report_paths, reports):
        require(path.name == f"{report['batch_id']}-home_wifi.json", "batch report identity")
        hosts = roster.loc[roster.batch_id.eq(report["batch_id"])].sort_values("host_order").kepid.tolist()
        require(report["hosts"] == hosts and report["host_count"] == len(hosts), "batch host membership")
        batch_outcomes = [x for host in hosts for x in states_by_host[host]["koi_outcomes"]]
        require(report["koi_count"] == len(batch_outcomes), "batch KOI count")
        for key, count in (("qc_pass", sum(o["technical_status"] == "constructed_qc_pass" for o in batch_outcomes)),
                           ("qc_fail", sum(o["technical_status"] == "constructed_qc_fail" for o in batch_outcomes)),
                           ("hard_failures", sum(o["processing_status"] != "constructed" for o in batch_outcomes))):
            require(report[key] == count, f"batch outcome mismatch: {key}")
    inventory = pd.read_csv(inventory_path)
    source_records = []
    for state in states:
        products = sorted(state["fits_products"], key=lambda x: (x["quarter"], x["product_filename"]))
        payload = "".join(f"{p['sha256']}  {p['product_filename']}\n" for p in products).encode()
        require(digest(payload) == state["fits_set_sha256"], "source hash-set mismatch")
        for product in products:
            require(re.fullmatch(r"[0-9a-f]{64}", product["sha256"]) is not None, "invalid source hash")
            source_records.append({"kepid": state["kepid"], **product})
    sources = pd.DataFrame(source_records)
    keys = ["kepid", "quarter", "product_filename"]
    require(not sources.product_filename.duplicated().any(), "duplicate source product")
    merged = inventory.merge(sources, on=keys, how="outer", indicator=True, validate="one_to_one")
    require(merged._merge.eq("both").all() and merged.product_size.eq(merged.bytes).all(), "source records differ from frozen inventory")
    source_lookup = {x["product_filename"]: x for x in source_records}
    for sample in evidence["fresh_samples"] + evidence["retained_sources"]:
        require(source_lookup[sample["product_filename"]]["sha256"] == sample["sha256"], "source history differs from audit evidence")
    constructed = [o for o in outcomes if o["processing_status"] == "constructed"]
    validate_unique_index(pd.DataFrame(constructed))
    expected_npz = {o["relative_path"] for o in constructed}
    actual_npz = {p.relative_to(dataset).as_posix() for p in dataset.rglob("*.npz") if p.is_file()}
    require(actual_npz == expected_npz, f"NPZ inventory mismatch: missing={sorted(expected_npz-actual_npz)[:10]}, orphan={sorted(actual_npz-expected_npz)[:10]}")
    print(f"Verifying {len(constructed)} NPZ files, logical hashes, shapes, dtypes and deterministic bytes", flush=True)
    sizes = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, size in enumerate(pool.map(lambda item: verify_npz(dataset, item, scientific), constructed), 1):
            sizes.append(size)
            if number % 500 == 0:
                print(f"Verified {number}/{len(constructed)} NPZ files", flush=True)
    for item, size in zip(constructed, sizes):
        item["npz_bytes"] = size
    require(sum(sizes) == sum(r["canonical_npz_bytes"] for r in reports), "batch NPZ bytes mismatch")
    require(len(constructed) == sum(r["canonical_npz_count"] for r in reports), "batch NPZ count mismatch")
    print("Auditing remaining raw and partial files", flush=True)
    residues = raw_listing(dataset, "dataset") + raw_listing(scratch, "scratch")
    expected_retained = {item["relative_path"]: item for item in evidence["retained_sources"]}
    for item in residues:
        if item["root"] == "scratch" and item["relative_path"] in expected_retained:
            original = expected_retained[item["relative_path"]]
            require(item["sha256"] == original["sha256"] and item["bytes"] == original["bytes"], "retained evidence changed")
            item["reason"] = "preserved_integrity_audit_evidence_post_purge_writes"
        else:
            raise ValueError(f"unexpected raw/partial residue: {item['root']}:{item['relative_path']}")
    require(len(residues) == len(expected_retained), "retained evidence missing")
    require(states_by_host[3530668]["raw_purged"] is False, "purge state not reconciled")
    validate_inputs(config)
    return {"config": config, "input_hashes": input_hashes, "cohort": cohort, "states": states,
            "outcomes": outcomes, "constructed": constructed, "reports": reports,
            "sources": merged.drop(columns="_merge"), "evidence": evidence, "residues": residues,
            "pre_provenance": pre,
            "state_checksums": {p.relative_to(dataset).as_posix(): file_digest(p) for p in state_paths},
            "batch_checksums": {p.relative_to(dataset).as_posix(): file_digest(p) for p in report_paths}}


def distribution(values):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all(), "nonfinite diagnostic metadata")
    return dict(zip(("min", "p05", "median", "p95", "max"), map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def build_artifacts(context):
    cohort, states, outcomes = context["cohort"], context["states"], context["outcomes"]
    constructed, sources = context["constructed"], context["sources"]
    artifacts = {}

    def add(name, value):
        artifacts[f"{PREFIX}{name}"] = value

    identity = ["source_row_index", "kepid", "kepoi_name", "label"]
    processing = pd.DataFrame([{**{k: o[k] for k in identity}, "processing_status": o["processing_status"],
        "technical_status": o["technical_status"], "qc_pass": o["qc_pass"],
        "model_data_available": o["processing_status"] == "constructed",
        "qc_reasons": json.dumps(o["qc_reasons"], separators=(",", ":")),
        "relative_path": o.get("relative_path", ""), "npz_sha256": o.get("npz_sha256", ""),
        "hard_failure_reason": o.get("hard_failure_reason", "")} for o in outcomes])
    add("processing_manifest_v1.csv", csv_bytes(processing))
    index = pd.DataFrame([{**{k: o[k] for k in identity}, "relative_path": o["relative_path"],
        "npz_bytes": o["npz_bytes"], "npz_sha256": o["npz_sha256"], "qc_pass": o["qc_pass"],
        "model_data_available": True, **{f"{k}_sha256": o["logical_array_hashes"][k] for k in KEYS}} for o in constructed])
    add("array_index_v1.csv", csv_bytes(index))
    add("array_checksums_v1.sha256", "".join(f"{o['npz_sha256']}  {o['relative_path']}\n" for o in constructed).encode())
    source_columns = ["kepid", "quarter", "product_filename", "data_uri", "bytes", "sha256", "embedded_checksums_valid"]
    source_manifest = sources[source_columns].sort_values(["kepid", "quarter", "product_filename"]).copy()
    source_manifest.rename(columns={"embedded_checksums_valid": "legacy_astropy_embedded_checksums_valid"}, inplace=True)
    sampled = {x["product_filename"] for x in context["evidence"]["fresh_samples"] + context["evidence"]["retained_sources"]}
    fresh_names = {x["product_filename"] for x in context["evidence"]["fresh_samples"]}
    source_manifest["independent_raw_hdu_checksum_status"] = source_manifest.product_filename.map(lambda x: "verified_pass" if x in sampled else "not_rechecked_source_purged")
    source_manifest["fresh_mast_sha256_verified"] = source_manifest.product_filename.isin(fresh_names)
    add("fits_manifest_v1.csv", csv_bytes(source_manifest))
    add("fits_checksums_v1.sha256", "".join(f"{r.sha256}  raw_fits/{r.kepid}/{r.product_filename}\n" for r in source_manifest.itertuples()).encode())
    add("runtime_metadata_checksums_v1.sha256", "".join(f"{sha}  {name}\n" for name, sha in sorted({**context["state_checksums"], **context["batch_checksums"]}.items())).encode())
    host_rows = []
    for state in sorted(states, key=lambda x: x["kepid"]):
        counts = Counter(o["technical_status"] for o in state["koi_outcomes"])
        host_rows.append({"kepid": state["kepid"], "koi_count": len(state["koi_outcomes"]),
            "fits_product_count": len(state["fits_products"]), "fits_bytes": sum(p["bytes"] for p in state["fits_products"]),
            "fits_set_sha256": state["fits_set_sha256"], "raw_purged": state["raw_purged"],
            "raw_retention_reason": state.get("raw_retention_reason", ""),
            "processing_rebuild_verified_at_run": state["deterministic_rebuild_verified"],
            **{key: counts[key] for key in sorted(TECHNICAL_STATUSES)}})
    add("host_outcomes_v1.csv", csv_bytes(pd.DataFrame(host_rows)))
    assignments = pd.read_csv(context["config"]["expected_inputs"]["stage3_assignments"])
    joined = assignments.merge(processing[identity + ["technical_status", "model_data_available"]], on=identity, how="left", validate="many_to_one")
    require(len(joined) == len(assignments) and joined.technical_status.notna().all(), "attrition denominators changed")
    multiplicity = cohort.groupby("kepid").size()
    periods = cohort.set_index("kepoi_name").koi_period
    joined["host_multiplicity"] = joined.kepid.map(multiplicity).map(lambda n: "single" if n == 1 else "multiple")
    period = joined.kepoi_name.map(periods)
    joined["period_stratum"] = np.select([period.le(0) | period.isna(), period.lt(10), period.lt(100)], ["invalid_or_missing", "0_to_lt10_days", "10_to_lt100_days"], default="100_days_or_more")
    dimensions = ["strategy", "repeat_id", "partition", "label", "host_multiplicity", "period_stratum"]
    denominator = joined.groupby(dimensions, dropna=False).size().rename("assigned_denominator").reset_index()
    attrition = joined.groupby(dimensions + ["technical_status", "model_data_available"], dropna=False).size().rename("rows").reset_index()
    attrition = attrition.merge(denominator, on=dimensions, validate="many_to_one").sort_values(dimensions + ["technical_status"])
    require(int(attrition.rows.sum()) == len(assignments), "attrition count mismatch")
    add("attrition_report_v1.csv", csv_bytes(attrition))
    marginal = {}
    for dimension in dimensions:
        table = joined.groupby([dimension, "technical_status", "model_data_available"], dropna=False).size().rename("rows").reset_index()
        marginal[dimension] = json.loads(table.to_json(orient="records"))
    add("attrition_summary_v1.json", json_bytes({"unique_koi_denominator": len(cohort), "assignment_row_denominator": len(assignments),
        "counting_note": "Assignment marginals repeat KOIs across 2 strategies and 10 repeats; each strategy/repeat retains 6637 assignments.",
        "host_multiplicity_definition": "Number of Stage 1 binary-cohort KOIs per kepid: single=1, multiple>=2",
        "period_stratum_definition": "Descriptive audit bins only: (0,10), [10,100), [100,infinity) days, invalid/missing separate; no selection or split changes.",
        "by_dimension": marginal}))
    metadata = [o["processing_metadata"] for o in constructed]
    flags = ["visibly_miscentered_flag", "phase_inversion_flag", "sap_pdcsap_material_difference_flag", "strong_quarter_discontinuity_flag", "unusually_high_interpolation_flag"]
    numbers = ["global_observed_fraction", "local_observed_fraction", "sap_pdcsap_morphology_difference_fraction", "maximum_quarter_boundary_jump", "temporal_baseline_fraction", "observed_transit_epochs", "valid_cadences"]
    diagnostics = {"constructed_denominator": len(constructed), "qc_reason_frequency": dict(sorted(Counter(reason for o in constructed for reason in o["qc_reasons"]).items())),
        "flags": {flag: sum(bool(m[flag]) for m in metadata) for flag in flags},
        "distributions": {key: distribution([m[key] for m in metadata]) for key in numbers},
        "valid_quarters_per_koi": dict(sorted(Counter(len(m["valid_quarters"]) for m in metadata).items())),
        "quarter_valid_koi_counts": dict(sorted(Counter(q for m in metadata for q in m["valid_quarters"]).items())),
        "quarter_product_host_counts": {int(k): int(v) for k, v in sources.groupby("quarter").size().items()},
        "failure_status_counts": dict(sorted(Counter(o["technical_status"] for o in outcomes).items()))}
    for view in ("global", "local"):
        diagnostics["distributions"][f"{view}_interpolation_fraction"] = distribution([1 - m[f"{view}_observed_fraction"] for m in metadata])
    add("aggregate_diagnostics_v1.json", json_bytes(diagnostics))
    add("raw_cleanup_audit_v1.json", json_bytes({"dataset_raw_or_partial_files": 0,
        "scratch_fits_files": len(context["residues"]), "scratch_fits_bytes": sum(x["bytes"] for x in context["residues"]),
        "partial_files": 0, "remaining_files": context["residues"],
        "policy": "Retain the 11 original FITS as integrity evidence; no deletion. Temporary fresh sample is separately retained as audit evidence outside dataset and scratch."}))
    counts = Counter(o["technical_status"] for o in outcomes)
    summary = {"certification_status": "certified_with_documented_warnings", "pipeline_id": context["config"]["pipeline_id"],
        "batches": len(context["reports"]), "hosts": len(states), "kois_assigned": len(outcomes),
        "technical_outcomes": {key: counts[key] for key in sorted(TECHNICAL_STATUSES)},
        "model_data_available_true": len(constructed), "model_data_available_false": len(outcomes) - len(constructed),
        "model_data_definition": "A verified constructible array exists; QC-fail arrays remain available under frozen Stage 2 policy. This is not a model-selection decision.",
        "no_product_hosts": sum(not s["fits_products"] for s in states),
        "canonical_npz_count": len(constructed), "canonical_npz_bytes": sum(o["npz_bytes"] for o in constructed),
        "fits_products_processed": len(sources), "unique_fits_bytes_processed": int(sources.bytes.sum()),
        "downloaded_bytes_in_surviving_batch_reports": sum(r["downloaded_bytes"] for r in context["reports"]),
        "download_bytes_caveat": "Surviving reports can overwrite earlier attempts on resume; cumulative network transfer is not recoverable from these reports.",
        "fresh_audit_download_bytes": context["evidence"]["fresh_download_bytes"],
        "legacy_embedded_checksum_false_products": int((sources.embedded_checksums_valid == False).sum()),
        "source_integrity_resolution": "Astropy 7.2.0 assumes zero data checksum when DATASUM is absent; independent raw HDU sums pass on all audited sources.",
        "fresh_source_comparisons": context["evidence"]["fresh_count"],
        "independent_raw_checksum_unique_products": len(sampled),
        "verification": {"npz_sha256": len(constructed), "logical_array_hashes": len(constructed) * len(KEYS),
            "shape_dtype_contract": "passed", "deterministic_npz_reserialization": len(constructed),
            "missing_duplicate_or_orphan_npz": 0, "stage3_assignments_unchanged": True,
            "failed_kois_replaced": 0, "kois_moved_partition": 0,
            "metadata_rebuild": "byte_identical", "absolute_path_audit": "passed"},
        "diagnostic_flags": diagnostics["flags"], "warnings": [
            "Historical embedded_checksums_valid=false records are preserved, not rewritten as individually reverified FITS.",
            "Raw-source certification combines the frozen inventory, retrieval-time SHA-256/size/identity records, and bounded fresh-byte evidence; purged FITS were not all re-downloaded.",
            "283 constructed QC failures remain available; frozen pilot flags are not exclusions.",
            "Phase centering, SAP/PDCSAP differences and quarter discontinuities are review flags, not silently corrected or excluded.",
            "Eleven original scratch FITS are intentionally retained; host 3530668 purge state is reconciled."]}
    add("summary_v1.json", json_bytes(summary))
    # Include evidence, policy and audit implementation in the final hash chain.
    dependencies = [META / f"{PREFIX}integrity_audit_v1.json", Path("docs/kepler_stage4a_validation_policy.md"),
                    Path("kepler_scale/integrity.py"), Path("kepler_scale/finalize.py"),
                    Path("scripts/stage4a_audit_integrity.py"), Path("scripts/stage4a_certify.py"),
                    Path("kepler_scale/runner.py")]
    provenance = {"pipeline_id": context["config"]["pipeline_id"], "status": summary["certification_status"],
        "frozen_inputs": {**context["input_hashes"], "stage4a_config_sha256": context["pre_provenance"]["stage4a_config_sha256"],
                          "mast_inventory_sha256": context["pre_provenance"]["mast_inventory_sha256"]},
        "artifacts_sha256": {name: digest(payload) for name, payload in sorted(artifacts.items())},
        "audit_dependencies_sha256": {path.as_posix(): file_digest(path) for path in dependencies},
        "state_and_batch_checksums": f"{PREFIX}runtime_metadata_checksums_v1.sha256",
        "limits": context["evidence"]["limits"], "stage4b_started": False, "model_code_introduced": False}
    add("final_provenance_v1.json", json_bytes(provenance))
    checks = {name: digest(payload) for name, payload in artifacts.items()}
    checks[f"{PREFIX}integrity_audit_v1.json"] = file_digest(META / f"{PREFIX}integrity_audit_v1.json")
    add("final_metadata_v1.sha256", "".join(f"{sha}  {name}\n" for name, sha in sorted(checks.items())).encode())
    for name, payload in artifacts.items():
        require(not FORBIDDEN.search(payload.decode()), f"absolute path in deterministic metadata: {name}")
    return artifacts


def certify(dataset, scratch):
    context = verify_existing(Path(dataset), Path(scratch))
    first = build_artifacts(context)
    second = build_artifacts(context)
    require(first == second, "deterministic metadata rebuild mismatch")
    for name, payload in sorted(first.items()):
        (META / name).write_bytes(payload)
    for name, payload in first.items():
        require((META / name).read_bytes() == payload, "metadata write verification failed")
    for line in (META / f"{PREFIX}final_metadata_v1.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(file_digest(META / name) == expected, "final metadata checksum failed")
    print(json.dumps(json.loads(first[f"{PREFIX}summary_v1.json"]), indent=2, sort_keys=True), flush=True)
    return first
