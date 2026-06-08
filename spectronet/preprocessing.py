import numpy as np

def generate_jwst_spectrum(ch4, co2, h2o, o2, o3, wavelength_bins=500):
    """
    Generates a high-fidelity 5-gas infrared transmission spectrum matching JWST metrics.
    Valleys drop downward below 1.0 baseline representing structural starlight absorption.
    """
    wavelengths = np.linspace(1.0, 5.0, wavelength_bins)
    flux = np.ones(wavelength_bins)
    
    # 5-Gas Mathematical Absorption Profiles (Valleys centered on real infrared bands)
    flux -= o2  * 0.15 * np.exp(-((wavelengths - 1.27) / 0.08)**2) # O2 feature at 1.27um
    flux -= h2o * 0.20 * np.exp(-((wavelengths - 2.70) / 0.25)**2) # H2O feature at 2.7um
    flux -= ch4 * 0.30 * np.exp(-((wavelengths - 3.30) / 0.15)**2) # CH4 feature at 3.3um
    flux -= co2 * 0.40 * np.exp(-((wavelengths - 4.30) / 0.12)**2) # CO2 feature at 4.3um
    flux -= o3  * 0.25 * np.exp(-((wavelengths - 4.70) / 0.10)**2) # O3 feature at 4.7um
    
    flux += np.random.normal(0, 0.01, wavelength_bins) # Atmospheric instrumentation noise
    return wavelengths, flux

def parse_planet_template(uploaded_file_bytes):
    """Parses custom uploaded text configurations into 5 active chemical abundances."""
    gas_mix = {"CH4": 0.25, "CO2": 0.60, "H2O": 0.40, "O2": 0.20, "O3": 0.10}
    try:
        decoded_content = uploaded_file_bytes.decode("utf-8")
        for line in decoded_content.splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.split("=")
                key = key.strip().upper()
                if key in gas_mix:
                    gas_mix[key] = float(val.strip())
        return gas_mix
    except:
        return gas_mix