# 🌌 Astrobiology Data Science & Deep Learning Platform

An advanced, multi-view machine learning platform designed to automate the detection of exoplanets and classify atmospheric biosignatures. This repository integrates two distinct astrobiology-focused frameworks utilizing optimized local pipelines for deployment.

---

## 🛠️ System Architecture & Portfolios

### 1. ExoMiner Vetting Engine (Transit Diagnostics)
Inspired by NASA’s automated exoplanet validation systems, this module processes stellar light curves to isolate real planetary transit events from astrophysical false positives (e.g., eclipsing binary stars or stellar rotation noise).

*   *Data Representation:* Raw stellar timelines are phase-folded around candidate orbital parameters and split into two separate 1D vector tensors:
    *   *Global Folded View (2000 bins):* Captures the full orbital period baseline to evaluate out-of-transit stellar stability and periodicity.
    *   *Local Folded View (200 bins):* Crops and zooms directly into the transit ingress/egress window to assess the depth and geometric shape of the eclipse.
*   *Target Machine Learning Model:* A Multi-Input Convolutional Neural Network (CNN) built in TensorFlow. The model uses separate feature extraction blocks to analyze the local and global structures in parallel before fusing them into a dense classification layer.

### 2. SpectroNet (JWST Atmospheric Analyzer)
An atmospheric inversion model framework designed to extract chemical abundances from transit spectroscopy datasets, mimicking measurements taken by instruments like the James Webb Space Telescope (JWST) NIRSpec.

*   *Data Representation:* Features consist of a 1D vector tracking relative transit depths across the near-to-mid-infrared spectrum (1.0µm to 5.0µm).
*   *Target Machine Learning Model:* A 1D Convolutional Deep Residual Network (ResNet). The architecture utilizes skip-connections to retain high-frequency molecular signal variations across network layers, feeding into a multi-head regression output layer.
*   *Biosignature Identification:* Targets specific absorption valleys corresponding to biogenic gases and non-equilibrium chemistry:
    *   *Water Vapor (H₂O):* Absorbs heavily at 2.7µm.
    *   *Methane (CH₄):* Clear biogenic proxy dropping at 3.3µm.
    *   *Carbon Dioxide (CO₂):* Vital carbon-cycle tracer creating sharp dips at 4.3µm.

---

## 💻 Repository Directory Layout

text
├── 📂 exominer/
│    └── 📄 preprocessing.py  # Phase-folding, index-masking, and Min-Max scaling
├── 📂 spectronet/
│    └── 📄 preprocessing.py  # Spectroscopic gas simulation and instrument noise injection
├── 📄 app.py                # Unified Streamlit Control Center application
├── 📄 requirements.txt      # Modular dependencies (Numpy, Pandas, Streamlit, Scipy)
└── 📄 setup.py              # Automated, offline workspace generation script

## 🚀 Deployment Instructions

### Integrated Side-by-Side Control Center
The platform features an open-workspace architecture that runs both machine learning pipelines simultaneously. Using `st.session_state` caching, the dashboard locks data in memory. This allows you to run independent planet searches on the left and simulate gas profiles on the right without erasing active outputs.

1. Install the required data structures and framework dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the graphical user interface via the local web application runner:
   ```bash
   python -m streamlit run app.py
   ```

### 📂 Custom Configuration File Templates

Users can bypass the dashboard's manual sliders by uploading custom planetary text configurations (`.txt` or `.ini` format). The dynamic data script reads custom columns instantly to compile corresponding target spectroscopic absorption charts.

Create a blank text document named `exo_earth.txt` and paste this block inside to use as a custom template:

```text
# Custom Planetary Atmospheric Signature Template
# Abundances map from 0.0 (None) to 1.0 (Maximum Saturated Valley)
O2 = 0.65
H2O = 0.80
CH4 = 0.15
CO2 = 0.05
O3 = 0.45
```
