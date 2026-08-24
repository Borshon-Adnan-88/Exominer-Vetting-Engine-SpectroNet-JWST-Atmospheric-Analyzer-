# Astrobiology Synthetic Data Prototype

This repository is an early interface and synthetic-data prototype for two future
astrobiology machine-learning projects. **It does not currently contain trained
machine-learning models or perform inference on observed data.**

## Current components

### Synthetic transit prototype

The left Streamlit panel generates deterministic toy global and local
transit-shaped relative-flux arrays. A target label can be entered for display,
but the application does not query Kepler or TESS and does not download or
phase-fold observed light curves.

The displayed prototype similarity score is a hand-written comparison with the
original demonstration defaults. It is not a probability, classifier output,
planet validation, or planet confirmation.

Transit epoch is intentionally absent from the synthetic generator because it
would have no effect on generated arrays. Epoch will be reintroduced when the
project implements genuine phase-folding of observed Kepler/TESS light curves.

### Synthetic spectrum prototype

The right panel adds five analytic Gaussian dips and white noise to a unit
baseline. Its gas abundance values are generator inputs. This is a toy spectrum,
not a JWST instrument simulator, radiative-transfer model, biosignature
classifier, or atmospheric retrieval. No ResNet is implemented or executed.

Supported toy inputs are CH4, CO2, H2O, O2, and O3, each constrained to the
dimensionless interval `[0, 1]`.

## Run the application

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Both generators expose explicit random seeds and use local NumPy random-number
generators, allowing an output to be reproduced without modifying global NumPy
random state.

## Run tests

Install the test dependency and run:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Repository layout

```text
exominer/             Synthetic transit generator
spectronet/           Gaussian toy spectrum generator and template parser
sample_planet_upload/ Example toy abundance templates
docs/history/         Clearly labelled screenshots from the superseded prototype
tests/                Unit tests for generation, validation, and reproducibility
app.py                Two-panel Streamlit application
```

## Research direction

The planned research question is: **How robust are machine-learning exoplanet
classifiers to realistic dataset splitting and changes in observational domain?**

Future work may add versioned Kepler/TESS TCE data, reproducible preprocessing,
host-star splits, classical baselines, a real global/local 1D CNN, repeated
evaluation, and cross-mission generalisation. None of those research stages is
implemented in this Stage 0 prototype.
