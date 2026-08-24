import numpy as np
import pytest

from exominer.preprocessing import generate_folded_views


def test_default_shapes_dtypes_and_finite_values():
    global_view, local_view = generate_folded_views(10.0, 4.0, seed=7)
    assert global_view.shape == (2000,)
    assert local_view.shape == (200,)
    assert global_view.dtype == np.float32
    assert local_view.dtype == np.float32
    assert np.isfinite(global_view).all()
    assert np.isfinite(local_view).all()


def test_seed_reproduces_views_and_different_seed_changes_them():
    first = generate_folded_views(10.0, 4.0, seed=7)
    repeated = generate_folded_views(10.0, 4.0, seed=7)
    different = generate_folded_views(10.0, 4.0, seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(first, repeated))
    assert any(not np.array_equal(a, b) for a, b in zip(first, different))


def test_generator_does_not_change_legacy_global_rng_state():
    np.random.seed(123)
    state_before = np.random.get_state()
    generate_folded_views(10.0, 4.0, seed=7)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_transit_is_a_downward_relative_flux_feature():
    global_view, local_view = generate_folded_views(10.0, 4.0, seed=7)
    assert global_view[len(global_view) // 2] < np.median(global_view[:300])
    assert np.median(local_view[80:120]) < np.median(local_view[:30])


@pytest.mark.parametrize(
    ("args", "kwargs", "exception"),
    [
        ((0.0, 4.0), {}, ValueError),
        ((np.nan, 4.0), {}, ValueError),
        ((10.0, np.inf), {}, ValueError),
        ((1.0, 24.0), {}, ValueError),
        ((10.0, 4.0), {"global_bins": 0}, ValueError),
        ((10.0, 4.0), {"local_bins": 2.5}, TypeError),
        ((10.0, 4.0), {"seed": 1.5}, TypeError),
    ],
)
def test_invalid_transit_inputs_raise(args, kwargs, exception):
    with pytest.raises(exception):
        generate_folded_views(*args, **kwargs)
