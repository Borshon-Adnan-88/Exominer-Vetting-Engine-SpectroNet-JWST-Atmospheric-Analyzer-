"""Generate the compact label-blind pilot review sheet."""

from kepler_lightcurves.diagnostics import plot_overview


if __name__ == "__main__":
    plot_overview(
        "data/lightcurve_processing/kepler_pilot_processing_manifest.csv",
        "data/lightcurves/processed",
        "docs/figures/kepler_pilot_diagnostic_overview.png",
    )
