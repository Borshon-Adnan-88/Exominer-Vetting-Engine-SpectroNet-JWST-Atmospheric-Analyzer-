import numpy as np

from kepler_lightcurves.preprocess import save_arrays


def test_npz_byte_hash_is_deterministic(tmp_path):
    arrays = {"global_flux": np.arange(10, dtype=np.float32), "local_count": np.arange(3, dtype=np.int32)}
    first = save_arrays(tmp_path / "first.npz", arrays)
    second = save_arrays(tmp_path / "second.npz", dict(reversed(list(arrays.items()))))
    assert first == second
    assert (tmp_path / "first.npz").read_bytes() == (tmp_path / "second.npz").read_bytes()
