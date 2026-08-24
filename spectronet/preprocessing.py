"""Validated toy spectrum generation for the synthetic SpectroNet demo."""

from numbers import Integral, Real

import numpy as np


GAS_DEFAULTS = {"CH4": 0.25, "CO2": 0.60, "H2O": 0.40, "O2": 0.20, "O3": 0.10}


def _validate_abundance(value: Real, gas: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{gas} abundance must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{gas} abundance must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{gas} abundance must be between 0 and 1")
    return value


def generate_jwst_spectrum(
    ch4: Real,
    co2: Real,
    h2o: Real,
    o2: Real,
    o3: Real,
    wavelength_bins: Integral = 500,
    seed: Integral | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy spectrum from Gaussian dips plus white noise.

    The output is not a JWST instrument simulation or atmospheric retrieval.
    """
    abundances = {
        gas: _validate_abundance(value, gas)
        for gas, value in {"CH4": ch4, "CO2": co2, "H2O": h2o, "O2": o2, "O3": o3}.items()
    }
    if isinstance(wavelength_bins, bool) or not isinstance(wavelength_bins, Integral):
        raise TypeError("wavelength_bins must be an integer")
    wavelength_bins = int(wavelength_bins)
    if wavelength_bins < 2:
        raise ValueError("wavelength_bins must be at least 2")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")

    rng = np.random.default_rng(None if seed is None else int(seed))
    wavelengths = np.linspace(1.0, 5.0, wavelength_bins)
    flux = np.ones(wavelength_bins)
    features = {
        "O2": (1.27, 0.08, 0.15),
        "H2O": (2.70, 0.25, 0.20),
        "CH4": (3.30, 0.15, 0.30),
        "CO2": (4.30, 0.12, 0.40),
        "O3": (4.70, 0.10, 0.25),
    }
    for gas, (center, width, scale) in features.items():
        flux -= abundances[gas] * scale * np.exp(-((wavelengths - center) / width) ** 2)
    flux += rng.normal(0.0, 0.01, wavelength_bins)
    return wavelengths, flux


def parse_planet_template(uploaded_file_bytes: bytes) -> dict[str, float]:
    """Parse a UTF-8 gas template, raising ``ValueError`` for invalid entries."""
    if not isinstance(uploaded_file_bytes, bytes):
        raise TypeError("uploaded_file_bytes must be bytes")
    try:
        decoded_content = uploaded_file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("template must be valid UTF-8") from exc

    gas_mix = GAS_DEFAULTS.copy()
    for line_number, raw_line in enumerate(decoded_content.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"line {line_number}: expected KEY = VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip().upper()
        if key not in gas_mix:
            raise ValueError(f"line {line_number}: unsupported gas {key!r}")
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {key} must be numeric") from exc
        try:
            gas_mix[key] = _validate_abundance(value, key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
    return gas_mix
