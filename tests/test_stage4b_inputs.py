import hashlib
import numpy as np
import pandas as pd
import pytest

from kepler_baselines.inputs import (StopForReview, assemble_features, verify_frozen_hashes,
                                    validate_assignments, load_features)
from kepler_scale.arrays import write_canonical_npz


def test_flux_order_and_metadata_rejection():
    allowed = {"global_flux": np.full(2000, 2, dtype="float32"), "local_flux": np.full(200, 3, dtype="float32")}
    features = assemble_features(allowed)
    assert features.shape == (2200,)
    assert (features[:2000] == 2).all() and (features[2000:] == 3).all()
    for forbidden in ("kepid", "label", "koi_score", "qc_pass", "global_count", "global_observed_mask"):
        with pytest.raises(StopForReview, match="forbidden metadata"):
            assemble_features({**allowed, forbidden: np.ones(1)})


def test_frozen_hash_failure_is_fatal(tmp_path):
    path = tmp_path / "frozen.csv"; path.write_bytes(b"original")
    config = {"frozen_sha256": {str(path): hashlib.sha256(b"original").hexdigest()}}
    verify_frozen_hashes(config)
    path.write_bytes(b"changed")
    with pytest.raises(StopForReview, match="hash mismatch"):
        verify_frozen_hashes(config)


def assignments():
    objects = pd.DataFrame({"source_row_index": range(6), "kepid": range(10, 16),
                            "kepoi_name": [f"K{i}" for i in range(6)], "label": [0, 1, 0, 1, 0, 1]})
    rows = objects.assign(strategy="host_group_stratified", repeat_id=0, seed=1729,
                          partition=["train", "train", "validation", "validation", "test", "test"])
    return objects, rows


def test_grouped_overlap_and_identity_mismatch_stop():
    objects, rows = assignments()
    validate_assignments(rows, objects, [1729], ["host_group_stratified"])
    objects.loc[4, "kepid"] = objects.loc[0, "kepid"]
    rows.loc[4, "kepid"] = rows.loc[0, "kepid"]
    with pytest.raises(StopForReview, match="grouped host overlap"):
        validate_assignments(rows, objects, [1729], ["host_group_stratified"])
    rows.loc[4, "label"] = 1
    with pytest.raises(StopForReview, match="identity join"):
        validate_assignments(rows, objects, [1729], ["host_group_stratified"])


def test_canonical_loader_keeps_qc_fail_and_rejects_corruption(tmp_path):
    arrays = {}
    for view, size in (("global", 2000), ("local", 200)):
        arrays[view + "_flux"] = np.ones(size, dtype="float32")
        arrays[view + "_count"] = np.ones(size, dtype="int32")
        arrays[view + "_observed_mask"] = np.ones(size, dtype=bool)
    relative = "processed/42/K00042_01.npz"
    hashes = write_canonical_npz(tmp_path / relative, arrays)
    row = {"source_row_index": 0, "kepid": 42, "kepoi_name": "K00042.01", "label": 1,
           "qc_pass": False, "relative_path": relative, "npz_sha256": hashes.pop("npz_sha256")}
    objects = pd.DataFrame([row]); index = pd.DataFrame([{**row, **{k + "_sha256": v for k, v in hashes.items()}}]).set_index("kepoi_name")
    assert load_features(tmp_path, objects, index).shape == (1, 2200)
    path = tmp_path / relative; path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(StopForReview, match="NPZ hash"):
        load_features(tmp_path, objects, index)
