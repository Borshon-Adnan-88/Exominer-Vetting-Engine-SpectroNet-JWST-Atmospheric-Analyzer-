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

## Stage 1: Kepler catalogue foundation

The repository includes an offline-reproducible metadata pipeline for the NASA
Exoplanet Archive `q1_q17_dr25_koi` delivery. It validates the live TAP schema
before retrieval, hashes raw bytes before parsing, preserves all selected KOIs in
a deterministic manifest, and applies the versioned
`confirmed_vs_false_positive_v1` policy with conflict auditing.

This remains KOI-centred: it is not yet a complete DR25 TCE dataset, defines no
train/test splits, and trains no models. See
`docs/data_card_kepler_dr25.md` for scientific scope and limitations.

The catalogue workflow is:

```bash
python -m scripts.fetch_kepler_catalog
python -m scripts.build_kepler_manifest
python -m pytest -m "not network"
```

The fetch command is the only normal workflow step requiring network access.
The live integration test is explicitly opt-in:

```bash
RUN_NETWORK_TESTS=1 python -m pytest tests/test_catalog_network.py -m network
```

## Stage 2: Kepler light-curve pilot

Stage 2 freezes official MAST Kepler Q1–Q17 DR25 long-cadence products for a
deterministic 16-KOI pilot and constructs label-blind global and local transit
views from PDCSAP flux. Raw FITS and generated arrays are ignored; selection,
product inventory, provenance, checksums, processing metadata, and compact
diagnostics are versioned. The QC thresholds are pilot review flags, not final
scientific exclusion rules. See `docs/kepler_lightcurve_preprocessing.md`.

No models, dataset splits, TESS processing, or Streamlit integration are part of
Stage 2.

## Stage 3: deterministic leakage-study splits

Stage 3 freezes ten paired 70/15/15 object-stratified and host-group-stratified
split repeats for the conflict-free Stage 1 binary KOI cohort. Grouped splits
keep every KOI sharing a `kepid` in one partition. Split assignments and audits
contain no model training or full-cohort photometry preprocessing. See
`docs/kepler_split_policy.md`.

## Stage 4A: low-resource full-cohort pipeline

Stage 4A provides one cross-platform, external-storage-aware pipeline for the
frozen cohort. It validates source hashes, batches hosts by `kepid`, checkpoints
MAST discovery, freezes product metadata separately from downloads, and supports
read-only preflight capacity analysis. Large runtime data live under a CLI or
`EXOMINER_DATA_ROOT` location. See `docs/kepler_stage4a_pipeline.md` and
`docs/external_data_root.md`.

Low-bandwidth operation additionally supports an operational-only
`EXOMINER_SCRATCH_ROOT` and a `public_wifi` profile with one-host scheduling and
a 100 MiB per-run transfer ceiling. These settings do not change scientific
configuration, assignments, or canonical outputs.

The operational-only `home_wifi` profile uses 10-host batches with one network
and one CPU worker. FITS are staged in scratch, canonical NPZ files are retained
under the data root, and verified raw files are not archived to cloud storage.

The completed Stage 4A cohort has 6,637 independently verified canonical NPZ
files. The FITS checksum discrepancy was traced to Astropy's handling of
CHECKSUM-only headers; raw-byte checksum checks and eight fresh MAST comparisons
passed. See the [validation policy](docs/kepler_stage4a_validation_policy.md) and
[certification report](docs/kepler_stage4a_certification_report.md) for evidence,
final hashes, QC warnings, and retained-source accounting. Stage 4B has not started.
