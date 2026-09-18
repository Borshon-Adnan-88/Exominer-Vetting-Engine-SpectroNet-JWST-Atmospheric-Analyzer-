import json

import numpy as np
import pytest

from kepler_scale.arrays import write_canonical_npz
from kepler_scale.finalize import verify_npz


def sample(tmp_path, *, dtype="float32"):
    config = json.load(open("configs/kepler_lightcurve_preprocessing_v1.json"))
    arrays = {}
    for view, size in (("global", 2000), ("local", 200)):
        arrays[f"{view}_flux"] = np.ones(size, dtype=dtype)
        arrays[f"{view}_count"] = np.ones(size, dtype="int32")
        arrays[f"{view}_observed_mask"] = np.ones(size, dtype=bool)
    name = "processed/42/K00042_01.npz"
    hashes = write_canonical_npz(tmp_path / name, arrays)
    outcome = {"kepid": 42, "kepoi_name": "K00042.01", "relative_path": name,
               "npz_sha256": hashes.pop("npz_sha256"), "logical_array_hashes": hashes,
               "processing_metadata": {"global_observed_fraction": 1.0, "local_observed_fraction": 1.0}}
    return config, outcome


def test_certification_rejects_damage_after_processing(tmp_path):
    config, outcome = sample(tmp_path)
    assert verify_npz(tmp_path, outcome, config) > 0
    path = tmp_path / outcome["relative_path"]
    value = bytearray(path.read_bytes()); value[-1] ^= 1; path.write_bytes(value)
    with pytest.raises(ValueError, match="NPZ checksum mismatch"):
        verify_npz(tmp_path, outcome, config)


def test_certification_rejects_wrong_dtype_even_with_matching_recorded_hashes(tmp_path):
    config, outcome = sample(tmp_path, dtype="float64")
    with pytest.raises(ValueError, match="shape/dtype"):
        verify_npz(tmp_path, outcome, config)


def test_certification_rejects_identity_path_mismatch(tmp_path):
    config, outcome = sample(tmp_path)
    outcome["kepid"] = 43
    with pytest.raises(ValueError, match="noncanonical identity/path"):
        verify_npz(tmp_path, outcome, config)
