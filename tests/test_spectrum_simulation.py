import numpy as np
import pytest

from spectronet.preprocessing import generate_jwst_spectrum


def test_spectrum_shape_grid_and_finite_values():
    wavelengths, flux = generate_jwst_spectrum(0.2, 0.3, 0.4, 0.5, 0.6, seed=9)
    assert wavelengths.shape == flux.shape == (500,)
    assert np.all(np.diff(wavelengths) > 0)
    assert wavelengths[0] == pytest.approx(1.0)
    assert wavelengths[-1] == pytest.approx(5.0)
    assert np.isfinite(flux).all()


def test_seed_is_reproducible_and_does_not_change_global_rng():
    np.random.seed(456)
    state_before = np.random.get_state()
    first = generate_jwst_spectrum(0.2, 0.3, 0.4, 0.5, 0.6, seed=9)
    repeated = generate_jwst_spectrum(0.2, 0.3, 0.4, 0.5, 0.6, seed=9)
    state_after = np.random.get_state()
    assert np.array_equal(first[0], repeated[0])
    assert np.array_equal(first[1], repeated[1])
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_increasing_ch4_deepens_its_toy_feature():
    wavelengths, without_ch4 = generate_jwst_spectrum(0, 0, 0, 0, 0, seed=3)
    _, with_ch4 = generate_jwst_spectrum(1, 0, 0, 0, 0, seed=3)
    feature_index = np.abs(wavelengths - 3.3).argmin()
    assert with_ch4[feature_index] < without_ch4[feature_index] - 0.25


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_invalid_abundances_raise(value):
    with pytest.raises(ValueError):
        generate_jwst_spectrum(value, 0, 0, 0, 0, seed=1)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"wavelength_bins": 1}, ValueError),
        ({"wavelength_bins": 2.5}, TypeError),
        ({"seed": "1"}, TypeError),
    ],
)
def test_invalid_spectrum_options_raise(kwargs, exception):
    with pytest.raises(exception):
        generate_jwst_spectrum(0, 0, 0, 0, 0, **kwargs)
