"""Streamlit interface for the repository's two synthetic demonstrations."""

import pandas as pd
import streamlit as st

import exominer.preprocessing as transit_demo
import spectronet.preprocessing as spectrum_demo


st.set_page_config(page_title="Astrobiology Prototype", layout="wide")
st.title("Astrobiology Synthetic Data Prototype")
st.warning(
    "This application currently generates toy synthetic data. It does not query "
    "Kepler/TESS, run a trained classifier, or perform atmospheric retrieval."
)
st.markdown("---")

if "transit_result" not in st.session_state:
    st.session_state.transit_result = None
if "spectrum_result" not in st.session_state:
    st.session_state.spectrum_result = None

panel_left, panel_right = st.columns(2)

with panel_left:
    st.header("Synthetic Transit Prototype")
    st.caption("Generates toy global/local transit-shaped arrays for interface development.")

    target_label = st.text_input(
        "Prototype target label (not queried)",
        "KIC 10593626",
        help="This is metadata only; no catalog or light curve is accessed.",
    )
    period_input = st.slider("Orbital period (days)", 1.0, 300.0, 288.03)
    duration_input = st.slider("Transit duration (hours)", 0.5, 15.0, 11.12)
    transit_seed = int(st.number_input("Transit random seed", min_value=0, value=626, step=1))

    if st.button("Generate synthetic transit views"):
        try:
            global_view, local_view = transit_demo.generate_folded_views(
                period_input, duration_input, seed=transit_seed
            )
            similarity_score = max(
                0.0,
                min(
                    100.0,
                    100.0
                    - (
                        abs(period_input - 288.03) / 300.0
                        + abs(duration_input - 11.12) / 15.0
                    )
                    * 25.0,
                ),
            )
            st.session_state.transit_result = {
                "global": global_view,
                "local": local_view,
                "target_label": target_label,
                "period": period_input,
                "duration": duration_input,
                "seed": transit_seed,
                "score": similarity_score,
            }
        except (TypeError, ValueError) as exc:
            st.error(f"Could not generate the synthetic transit: {exc}")

    result = st.session_state.transit_result
    if result is not None:
        st.success("Synthetic transit views generated.")
        st.caption(
            f"Generated for {result['target_label']} with period={result['period']:.2f} d, "
            f"duration={result['duration']:.2f} h, seed={result['seed']}."
        )
        st.write("**Global synthetic view** (2000 bins)")
        st.line_chart(result["global"])
        st.write("**Local synthetic view** (200 bins)")
        st.line_chart(result["local"])
        st.metric("Prototype similarity score (not a probability)", f"{result['score']:.2f}/100")
        st.info(
            "This hand-written score measures proximity to the original demo defaults. "
            "It is not ML inference and has no planet-validation meaning."
        )

with panel_right:
    st.header("Synthetic Spectrum Prototype")
    st.caption("Generates Gaussian toy absorption features with additive white noise.")

    uploaded_file = st.file_uploader("Upload gas template (.txt / .ini)", type=["txt", "ini"])
    gas_profile = spectrum_demo.GAS_DEFAULTS.copy()
    if uploaded_file is not None:
        try:
            gas_profile = spectrum_demo.parse_planet_template(uploaded_file.getvalue())
            st.info("Template loaded. Values remain synthetic generator inputs.")
        except (TypeError, ValueError) as exc:
            st.error(f"Invalid template: {exc}")

    ch4_val = st.slider("Methane (CH4) toy abundance", 0.0, 1.0, gas_profile["CH4"])
    co2_val = st.slider("Carbon dioxide (CO2) toy abundance", 0.0, 1.0, gas_profile["CO2"])
    h2o_val = st.slider("Water vapor (H2O) toy abundance", 0.0, 1.0, gas_profile["H2O"])
    o2_val = st.slider("Oxygen (O2) toy abundance", 0.0, 1.0, gas_profile["O2"])
    o3_val = st.slider("Ozone (O3) toy abundance", 0.0, 1.0, gas_profile["O3"])
    spectrum_seed = int(st.number_input("Spectrum random seed", min_value=0, value=39, step=1))

    if st.button("Generate Gaussian toy spectrum"):
        abundances = {
            "CH4": ch4_val,
            "CO2": co2_val,
            "H2O": h2o_val,
            "O2": o2_val,
            "O3": o3_val,
        }
        try:
            wavelengths, flux = spectrum_demo.generate_jwst_spectrum(
                ch4_val, co2_val, h2o_val, o2_val, o3_val, seed=spectrum_seed
            )
            st.session_state.spectrum_result = {
                "wavelengths": wavelengths,
                "flux": flux,
                "abundances": abundances,
                "seed": spectrum_seed,
            }
        except (TypeError, ValueError) as exc:
            st.error(f"Could not generate the toy spectrum: {exc}")

    result = st.session_state.spectrum_result
    if result is not None:
        st.success("Gaussian toy spectrum generated.")
        chart_data = pd.DataFrame(
            {
                "Wavelength (micrometres)": result["wavelengths"],
                "Relative flux": result["flux"],
            }
        ).set_index("Wavelength (micrometres)")
        st.line_chart(chart_data)
        values = ", ".join(
            f"{gas}={value:.2f}" for gas, value in result["abundances"].items()
        )
        st.caption(f"Inputs used: {values}; seed={result['seed']}.")
        st.info(
            "These displayed values are generator inputs, not retrieved abundances. "
            "No ResNet or other trained model is present."
        )
