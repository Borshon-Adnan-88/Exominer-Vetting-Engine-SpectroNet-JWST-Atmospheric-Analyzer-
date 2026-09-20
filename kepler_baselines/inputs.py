"""Fail-closed identity, frozen-input and canonical-array validation."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from kepler_scale.arrays import KEYS, logical_hash
from kepler_scale.index import canonical_relative_path

IDENTITY = ["source_row_index", "kepid", "kepoi_name", "label"]
FEATURE_KEYS = ("global_flux", "local_flux")
PARTITIONS = ("train", "validation", "test")


class StopForReview(RuntimeError):
    """Scientific stop condition: never repair or silently skip an input/run."""


def require(condition, reason):
    if not condition:
        raise StopForReview(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path="configs/kepler_stage4b_baselines_v1.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_frozen_hashes(config):
    for name, expected in config["frozen_sha256"].items():
        require(sha256(name) == expected, f"frozen input hash mismatch: {name}")
    return dict(config["frozen_sha256"])


def assemble_features(arrays):
    require(set(arrays) == set(FEATURE_KEYS), "forbidden metadata or unexpected feature field")
    for key, size in zip(FEATURE_KEYS, (2000, 200)):
        require(arrays[key].shape == (size,), f"invalid feature shape: {key}")
        require(np.isfinite(arrays[key]).all(), f"non-finite feature input: {key}")
    return np.concatenate([arrays[key] for key in FEATURE_KEYS]).astype(np.float64)


def validate_assignments(assignments, objects, seeds, strategies):
    require(set(assignments.strategy) == set(strategies), "unexpected split strategies")
    require(set(assignments.repeat_id) == set(range(len(seeds))), "unexpected repeat IDs")
    require(len(assignments) == len(objects) * len(seeds) * len(strategies), "assignment denominator mismatch")
    require(not assignments.duplicated(["strategy", "repeat_id", "kepoi_name"]).any(), "duplicate KOI assignment")
    comparison = assignments.merge(objects[IDENTITY], on=IDENTITY, how="outer", indicator=True, validate="many_to_one")
    require(comparison._merge.eq("both").all(), "KOI identity join failed in assignments")
    for (strategy, repeat), rows in assignments.groupby(["strategy", "repeat_id"]):
        require(len(rows) == len(objects) and set(rows.kepoi_name) == set(objects.kepoi_name), "repeat cohort changed")
        require(rows.seed.eq(seeds[int(repeat)]).all(), "Stage 3 seed changed")
        require(set(rows.partition) == set(PARTITIONS), "partition missing or renamed")
        if "fold_id" in rows:
            mapped = rows.fold_id.map(lambda f: "train" if 0 <= f <= 13 else "validation" if 14 <= f <= 16 else "test" if 17 <= f <= 19 else "invalid")
            require(mapped.eq(rows.partition).all(), "Stage 3 partition assignment changed")
        for _, partition in rows.groupby("partition"):
            require(set(partition.label) == {0, 1}, "partition lacks both classes")
        if strategy == "host_group_stratified":
            require(rows.groupby("kepid").partition.nunique().eq(1).all(), "grouped host overlap detected")


def read_input_tables(config):
    verify_frozen_hashes(config)
    cohort = pd.read_csv(config["inputs"]["cohort"])
    cohort = cohort.loc[cohort.inclusion_status.eq("included") & cohort.label_conflict.eq(False)].copy()
    require(len(cohort) == config["expected_kois"] and cohort.kepid.nunique() == config["expected_hosts"], "cohort size mismatch")
    require(not cohort.kepoi_name.duplicated().any() and not cohort.source_row_index.duplicated().any(), "duplicate cohort identity")
    require(set(cohort.label) == {0, 1}, "binary labels changed")
    index = pd.read_csv(config["inputs"]["index"])
    processing = pd.read_csv(config["inputs"]["processing"])
    require(len(index) == len(processing) == len(cohort), "canonical/processing denominator mismatch")
    require(not index.kepoi_name.duplicated().any() and not index.relative_path.duplicated().any(), "duplicate canonical identity/path")
    for frame, name in ((index, "index"), (processing, "processing")):
        require(not frame.kepoi_name.duplicated().any(), f"duplicate KOI in {name}")
        joined = cohort[IDENTITY].merge(frame[IDENTITY], on=IDENTITY, how="outer", indicator=True, validate="one_to_one")
        require(joined._merge.eq("both").all(), f"KOI identity join failed: {name}")
        require(frame.model_data_available.eq(True).all(), "model-available cohort changed")
    require(processing.processing_status.eq("constructed").all(), "unexpected technical failure")
    joined = index.merge(processing[IDENTITY + ["qc_pass", "npz_sha256", "relative_path"]], on=IDENTITY, validate="one_to_one", suffixes=("", "_processing"))
    for key in ("qc_pass", "npz_sha256", "relative_path"):
        require(joined[key].eq(joined[key + "_processing"]).all(), f"processing/index mismatch: {key}")
    require(int(index.qc_pass.sum()) == config["expected_qc_pass"], "QC count changed")
    objects = index[IDENTITY + ["qc_pass", "relative_path", "npz_sha256"]].sort_values(["kepid", "kepoi_name"], kind="stable").reset_index(drop=True)
    assignments = pd.read_csv(config["inputs"]["assignments"])
    validate_assignments(assignments, objects, config["seeds"], config["strategies"])
    return objects, index.set_index("kepoi_name"), assignments


def load_features(dataset, objects, index):
    dataset = Path(dataset).resolve()
    expected_paths = set(objects.relative_path)
    actual = {p.relative_to(dataset).as_posix() for p in (dataset / "processed").rglob("*.npz")}
    require(actual == expected_paths, "missing or orphan canonical NPZ")
    matrix = np.empty((len(objects), 2200), dtype=np.float64)
    for position, obj in enumerate(objects.itertuples(index=False)):
        relative = PurePosixPath(obj.relative_path)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in obj.relative_path, "unsafe canonical path")
        require(obj.relative_path == canonical_relative_path(obj.kepid, obj.kepoi_name), "canonical path/identity mismatch")
        path = dataset / obj.relative_path
        require(path.resolve().is_relative_to(dataset), "canonical file escapes dataset")
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == obj.npz_sha256, f"NPZ hash mismatch: {obj.kepoi_name}")
        record = index.loc[obj.kepoi_name]
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            require(len(source.files) == 6 and set(source.files) == set(KEYS), "unexpected NPZ arrays")
            for key in KEYS:
                value = source[key]
                size = 2000 if key.startswith("global_") else 200
                dtype = "|b1" if key.endswith("mask") else "<i4" if key.endswith("count") else "<f4"
                require(value.shape == (size,) and value.dtype.str == dtype, f"frozen shape/dtype mismatch: {obj.kepoi_name}:{key}")
                require(np.isfinite(value).all(), f"non-finite model input: {obj.kepoi_name}:{key}")
                require(logical_hash(value) == record[key + "_sha256"], f"logical array hash mismatch: {obj.kepoi_name}:{key}")
            for view in ("global", "local"):
                require((source[view + "_count"] >= 0).all(), "negative bin counts")
                require(np.array_equal(source[view + "_observed_mask"], source[view + "_count"] > 0), "mask/count inconsistency")
            # Explicit allowlist: metadata, masks and counts cannot reach models.
            matrix[position] = assemble_features({key: source[key] for key in FEATURE_KEYS})
        if (position + 1) % 500 == 0:
            print(f"Validated canonical arrays: {position + 1}/{len(objects)}", flush=True)
    require(np.isfinite(matrix).all(), "non-finite feature matrix")
    return matrix
