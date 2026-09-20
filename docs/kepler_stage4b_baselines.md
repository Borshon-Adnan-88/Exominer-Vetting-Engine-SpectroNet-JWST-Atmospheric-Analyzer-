# Stage 4B fixed supervised baselines

Stage 4B compares object-level and host-group test performance by the existing
ten paired repeat IDs. Test ROC-AUC is primary. Average precision, Brier score,
precision, recall, F1 and the confusion matrix are key secondary outcomes.
Accuracy and trapezoidal PR-AUC are also reported. No tuning or new splitting is
part of this experiment.

## Frozen cohort and representation

`configs/kepler_stage4b_baselines_v1.json` anchors expected input SHA-256 values
to certified Stage 4A commit `b2d20c6`. Before fitting, the runner verifies the
Stage 1 cohort/labels, Stage 3 assignments/policy, Stage 2/4A scientific configs,
frozen inventory, and certified Stage 4A metadata. All 6,637 canonical NPZ files
and 39,822 logical-array hashes, expected dtypes/shapes, finite values and
identity joins must pass. Missing, duplicate or orphan canonical arrays stop
execution. Frozen inputs are read-only and never repaired.

Eligibility is `all_model_available_v1`: all 6,637 KOIs / 5,804 hosts, including
6,354 QC passes and 283 QC failures. Label 1 is confirmed planet; label 0 is
false positive. Features concatenate exactly 2,000 global flux values followed
by 200 local flux values. The explicit two-array allowlist excludes all
catalogue metadata, IDs, labels, QC flags, masks and counts from features.
Identities and flags are retained only in separate auditing/reporting tables.
Conversion of the frozen float32 fluxes to float64 is exact; no new photometry
preprocessing or population transformation occurs in feature loading.

## Models and fitting boundary

Logistic regression uses one sklearn Pipeline:

`StandardScaler -> LogisticRegression`

The pipeline is fitted once on `X[partition == "train"]` and its corresponding
labels for each strategy/repeat. Prediction invokes only the already-fitted
pipeline. There is no standalone scaler fitting, full-cohort fitting,
validation/test fitting, or train-plus-validation refit. Tests spy on both
scaler and classifier fit calls, assert exact training rows and sample count,
and prove that extreme holdout values and holdout predictions do not refit or
change the scaler.

Logistic settings: L2 (`l1_ratio=0` in sklearn 1.9), C=1, lbfgs, max_iter=2000,
tol=1e-4, fitted intercept, no class weights. Convergence warnings or reaching
the iteration cap stop the experiment for review.

Random Forest: 200 trees, maximum depth 12, minimum leaf size 5, sqrt feature
subsampling, Gini criterion, bootstrap enabled, no class weights, n_jobs=1.
It consumes the same flux vector without scaling. Its random seed is the
unchanged Stage 3 seed for the repeat, shared across the two strategies.
Logistic lbfgs is deterministic and has no stochastic seed operation.

Both model configurations are fixed before observing scores. No PCA,
imputation, feature selection, calibration fitting, resampling outside the
forest's training bootstrap, threshold optimization or parameter search is
performed. Forty fits are scheduled sequentially on CPU. Numerical libraries
are limited to one thread before import and during fitting/prediction.

## Evaluation

The decision rule is `p(confirmed) >= 0.5`, including ties. Confusion-matrix
columns are TN, FP, FN, TP. Average precision is non-trapezoidal; trapezoidal
PR-AUC is an additional separately labelled column. Brier is mean squared
probability error on the binary label, on the 0-to-1 scale.

Every model/strategy/repeat has train, validation and test metrics. Training
scores are in-sample diagnostics. Each partition has two evaluation tracks:

- Primary: all assigned, model-available KOIs.
- QC sensitivity: only its QC-pass rows, using the identical fitted model and
  probabilities. This does not test QC-pass-only training.

Assigned/evaluated/QC-excluded, positive and negative counts and prevalence are
explicit. Undefined ranking metrics preserve rows with a reason; zero-division
precision, recall or F1 is reported as zero with a flag. No KOI is reassigned.

Reliability tables use ten fixed equal-width probability bins on [0,1]; empty
bins remain present and probability 1 belongs to the last bin. Brier plus raw
calibration curves assess probability quality without post-hoc calibration.

For each model/partition/track/metric, retain all ten paired values and calculate
`object - grouped` by repeat ID. Report mean, median, sample standard deviation,
minimum, maximum and range. Lower Brier is better, so a negative difference
favours object-level splitting. Repeats reuse the cohort; summaries are
descriptive, without independence-based confidence intervals or p-values.

Host seen/unseen reporting is **secondary only**, within object-split test rows,
where "seen" means the host appears in that repeat's training partition. It
cannot affect fitting, thresholds, partitions or primary conclusions. A test
proves that changing the diagnostic grouping leaves primary and sensitivity
metrics unchanged. Host overlap is also recorded for each frozen split.

The split comparison is not a pure causal estimate of leakage: training and
test membership and host/class composition differ. QC flags remain scientific
review flags; this experiment does not certify astrophysical labels.

## Execution and artifacts

Run from the repository root with the already certified external dataset:

```text
python -m scripts.stage4b_run_baselines --dataset "<data-root>/kepler_dr25"
python -m scripts.stage4b_run_baselines --verify-predictions
```

The first command validates inputs and executes the 40-fit loop directly in the
terminal; no agent schedules individual fits. The second command is a new
process and loads stored models to verify all prediction probabilities bit for
bit, without any fitting. Repeated prediction, model serialization/reloading
and CSV float roundtrip are also verified during the initial run. A mismatch
stops for review. Reproducibility is scoped to the pinned environment and
single-thread settings; cross-platform estimator-file identity is not promised.

Existing output directories are not overwritten or resumed automatically.
An incomplete experiment writes `stopped_for_review.json` and requires review;
no seed is skipped or searched and no nonconverged model is accepted.

Compact results live in `data/stage4b/`: input validation, object identities,
metrics, calibration bins, paired values and summaries, host diagnostics,
runtime, run manifest, provenance and checksum manifests. Four aggregate figures
show primary test pairs, raw calibration, QC evaluation sensitivity and the
secondary host diagnostic.

Large artifacts live in ignored `data/stage4b_runtime/baseline_v1/`: one validated
feature cache, forty model artifacts, and forty prediction tables. Each run has
one probability per KOI; the QC track reuses these probabilities. Runtime paths
in manifests are relative to this root. The feature cache and every trained
model/prediction table are hashed. Do not load untrusted joblib files.

Runtime reporting distinguishes validation, fitting by family, and
fit/predict/serialization by family. RAM is the process peak working set/RSS,
including the interpreter and feature matrix; family peaks are sampled every
50 ms and are not incremental allocation estimates. The new-process replay is
timed separately. Model artifact sizes are actual compressed joblib bytes.

The Streamlit application remains a synthetic demonstration. It does not load
these models or perform inference on observed data. No CNN, Stage 5, TESS,
Streamlit model integration or automatic commit is included.
