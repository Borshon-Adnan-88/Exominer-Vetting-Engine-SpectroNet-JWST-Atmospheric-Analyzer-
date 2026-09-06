from pathlib import Path

import numpy as np

from kepler_scale.arrays import write_canonical_npz
from kepler_scale.contracts import load_config
from kepler_scale.paths import resolve_data_paths, resolve_scratch_paths


def arrays():
    return {
        "global_count": np.ones(2000, dtype=np.int32),
        "global_flux": np.ones(2000, dtype=np.float32),
        "global_observed_mask": np.ones(2000, dtype=bool),
        "local_count": np.ones(200, dtype=np.int32),
        "local_flux": np.ones(200, dtype=np.float32),
        "local_observed_mask": np.ones(200, dtype=bool),
    }


def test_scratch_precedence_and_fallback(tmp_path, monkeypatch):
    config = load_config(); config["paths"]["require_external_to_repository"] = False
    data = resolve_data_paths(str(tmp_path / "data"), config, tmp_path, create=True)
    monkeypatch.setenv("EXOMINER_SCRATCH_ROOT", str(tmp_path / "environment"))
    explicit = resolve_scratch_paths(str(tmp_path / "explicit"), data)
    assert explicit.root == (tmp_path / "explicit" / "kepler_dr25").resolve()
    assert resolve_scratch_paths(None, data).root == (tmp_path / "environment" / "kepler_dr25").resolve()
    monkeypatch.delenv("EXOMINER_SCRATCH_ROOT")
    assert resolve_scratch_paths(None, data).root == data.temporary


def test_scratch_choice_cannot_change_canonical_output(tmp_path):
    first = resolve_scratch_paths(str(tmp_path / "scratch-a"), type("D", (), {"temporary": tmp_path / "fallback"})())
    second = resolve_scratch_paths(str(tmp_path / "scratch-b"), type("D", (), {"temporary": tmp_path / "fallback"})())
    assert first.root != second.root
    a = write_canonical_npz(tmp_path / "data" / "a.npz", arrays())
    b = write_canonical_npz(tmp_path / "data" / "b.npz", arrays())
    assert a == b
