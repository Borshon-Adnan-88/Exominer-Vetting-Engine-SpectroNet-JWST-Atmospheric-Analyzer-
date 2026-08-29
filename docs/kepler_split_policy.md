# Stage 3 Kepler KOI split policy

Stage 3 consumes only the frozen, conflict-free Stage 1 binary cohort. It creates
paired object-stratified and host-group-stratified assignments for ten fixed
seeds. It does not inspect photometry or train models.

Each strategy creates 20 folds: IDs 0–13 map to train, 14–16 to validation,
and 17–19 to test. **Fold IDs are arbitrary deterministic labels produced by the
splitting algorithm. They have no ordinal or scientific meaning.** They are not
reordered or optimized after inspecting class balance, hosts, or metadata.

Object splits use `StratifiedKFold`; grouped splits use
`StratifiedGroupKFold(groups=kepid)`. A grouped split must keep each host wholly
within one partition, including mixed-label systems. Each partition must be
within one percentage point of its 70/15/15 row target, within 1.5 percentage
points of full-cohort class prevalence, and contain both classes. Failure stops
the workflow: seeds are not searched, folds are not reordered, assignments are
not optimized, and tolerances are not relaxed.

Only `label`, `kepid`, and stable identities participate in splitting. Period,
duration, S/N, and host multiplicity are audit-only. Dispositions, confirmation
names, Robovetter scores/flags, comments, and provenance fields are forbidden in
model-related paths.

Split assignments must precede full-cohort scaling. Fixed independent per-object
preprocessing may subsequently run without cross-object fitting. Population
normalization, imputation, learned detrending, feature selection, augmentation,
resampling, calibration, and decision thresholds must be fit using the training
partition of each repeat only. Preprocessing failures cause reported attrition;
they never cause reassignment.
