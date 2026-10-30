# Manuscript outline after Stage 5D

Working framing: **A frozen host-aware Kepler classification benchmark and a diagnostic study of dual-view CNN limitations.** This is a framing proposal, not a literature-verified novelty claim. No abstract is drafted.

The paper should follow the scientific argument: define the labelled population and generalization target; establish a strong baseline; diagnose the CNN shortfall; test two constrained explanations; distinguish improved representation accessibility from improved held-out prediction. Stage labels can remain in an experiment map or supplement rather than drive the paper's chronology.

## 1. Introduction

- **Existing evidence:** A common flux-only benchmark, paired host-aware evaluation, a strong Random Forest reference, and two controlled CNN interventions. The final intervention preserves probe-accessible discrimination much more than it improves test prediction.
- **Needs writing/research:** Explain the scientific importance of the specific confirmed-versus-FP task and host-level generalization. State research questions without claiming a new planet detector or universal model ranking. Verify the contribution against literature.
- **Likely displays:** One conceptual study diagram assembled from the completed design; no new data analysis is needed.

## 2. Related Work

- **Existing evidence:** Exact internal claims and comparison boundaries are listed in [the evidence checkpoint](manuscript_evidence_after_stage5d.md#what-may-be-publishable-or-novel).
- **Needs writing/research:** Verify primary literature on DR25 KOI/TCE label definitions, global/local transit classifiers, classical flux baselines, grouped evaluation, inactive-ReLU behavior, and representation probes. Compare dataset, label, split, feature and metric definitions before comparing numerical scores. No novelty claim is yet verified.
- **Likely displays:** A literature-comparison matrix with task, cohort, split unit, model inputs, evaluation protocol, and available code/data; currently unwritten.

## 3. Dataset

- **Existing evidence:** Stage 1 [data card](data_card_kepler_dr25.md), [label policy](label_policy.md), manifest and conflict audit. Final cohort: 6,637 KOIs, 5,804 hosts; 2,674 confirmed and 3,963 false positives. Candidates/conflicts are excluded under the frozen policy.
- **Needs writing/research:** Present confirmation-selection bias and distinguish a KOI cohort from all TCEs or a uniform Robovetter PC-versus-FP task. Transcribe the exact snapshot provenance and exclusions from the frozen metadata.
- **Likely displays:** Cohort accounting table; host multiplicity and label-policy schematic using already recorded counts.

## 4. Preprocessing

- **Existing evidence:** Stage 2 [preprocessing specification](kepler_lightcurve_preprocessing.md) and Stage 4A [certification report](kepler_stage4a_certification_report.md). Real PDCSAP photometry, quality mask 1130799, quarter median normalization, ephemeris folding, 2,000/200 global/local bins, and deterministic interpolation. All objects constructible; 6,354 QC pass and 283 QC fail.
- **Needs writing/research:** Describe official ephemeris dependence, per-object processing boundaries, QC sensitivity, morphology/centering warnings, and the limits of source-integrity certification after raw-file purging. Separate scientific warnings from software-integrity acceptance.
- **Likely displays:** Existing [pilot diagnostic overview](figures/kepler_pilot_diagnostic_overview.png); compact cohort/QC table. Curate an existing representative illustration without new selection-based claims.

## 5. Split methodology

- **Existing evidence:** Frozen Stage 3 [policy](kepler_split_policy.md) and assignments. Ten seeds; 20 deterministic folds mapped to 70/15/15 train/validation/test; paired object-stratified and `kepid`-grouped strategies. Grouped hosts remain intact.
- **Needs writing/research:** Define each protocol's prediction target, pairing unit and denominator. Explain overlapping repeats, changing population composition, and why object-minus-group differences are descriptive rather than causal leakage estimates. Document that test partitions were reused across model-development stages.
- **Likely displays:** Split-design schematic and certified host-overlap accounting; cross-model split-sensitivity table.

## 6. Classical baselines

- **Existing evidence:** Stage 4B [specification](kepler_stage4b_baselines.md) and [report](kepler_stage4b_report.md). Same 2,200 flux values; training-only StandardScaler/L2 Logistic Regression; fixed 200-tree RF with depth 12 and minimum leaf size 5. No tuning or recalibration.
- **Needs writing/research:** Explain why these baselines test linear versus nonlinear discrimination, while acknowledging that fixed settings do not establish optimal family-level rankings. Preserve input and fitting boundaries.
- **Likely displays:** Model-setting table and the held-out master table in the Results draft; existing baseline calibration figure if space allows.

## 7. CNN architecture

- **Existing evidence:** Stage 5 [architecture/training specification](kepler_stage5_cnn.md). Separate global/local convolutional branches, concatenate 4,160 features, 32-unit ReLU fusion, dropout 0.2 and sigmoid; 138,561 parameters. Adam/BCE, fixed threshold 0.5, validation-BCE checkpoint selection, patience 10 and cap 100.
- **Needs writing/research:** Explain why this architecture was selected without implying equivalence to a published optimized system. Clarify earliest-minimum-BCE weights, paired initialization, inference-mode histories and absence of test-based checkpoint selection.
- **Likely displays:** Architecture schematic and an intervention table identifying the sole changed factor at each ablation.

## 8. Controlled ablations

- **Existing evidence:** Stage 5B [activation report](kepler_stage5b_activation_ablation.md), Stage 5D [validation report](kepler_stage5d_validation_analysis.md) and [test report](kepler_stage5d_test_evaluation.md). LeakyReLU(0.01) replaces fusion ReLU at width 32; widening then changes width 32 to 128 (538,113 parameters). Other scientific settings remain frozen.
- **Needs writing/research:** State intervention hypotheses and decision timing faithfully. Do not retrospectively call the entire sequential development program a preregistered study. Explain that differently shaped heads cannot share identical full parameter arrays and that branches train end-to-end.
- **Likely displays:** Results-draft ablation table; existing [collapse comparison](../data/stage5b/figures/collapse_stability.png) and [width paired differences](../data/stage5d/figures/paired_differences.png).

## 9. Representation diagnostics

- **Existing evidence:** Stage 5A [endpoint diagnostics](kepler_stage5a_failure_analysis.md) and Stage 5C [linear probes](kepler_stage5c_linear_probes.md). Six fixed probes; scalers/classifiers trained only on training rows, validation-only evaluation. Stage 5D reuses compatible raw probes and recomputes four learned probes per fit.
- **Needs writing/research:** Distinguish activation endpoint evidence, fixed linear accessibility, and irreversible information loss. Note dimensionality/regularization differences and absence of historical gradient trajectories. Do not present validation probes as held-out test scores.
- **Likely displays:** Validation-only representation table and existing [fusion-gap comparison](../data/stage5d/figures/fusion_gap.png).

## 10. Results

- **Existing evidence:** [Publication-style Results draft](manuscript_results_stage1_to_stage5d.md), including the ten-row held-out master table, split comparison, ablation table, validation-only probe table, and paired width128 test outcomes.
- **Needs writing/research:** Fit the prose/table numbering to the eventual journal format; ensure every reported number retains its partition, primary/QC track and unit of aggregation. No further fitting, prediction scoring, or inferential threshold selection is needed for this draft.
- **Likely displays:** Master AUC/AP/Brier table; intervention table; paired test differences; compact QC/calibration summaries from existing artifacts.

## 11. Discussion

- **Existing evidence:** RF remains strongest among evaluated models; LeakyReLU removes observed collapse; widening substantially improves probes but modestly improves classification. Classical split differences are small, while CNN differences vary.
- **Needs writing/research:** Separate strong, negative, suggestive and unresolved findings using the [evidence taxonomy](manuscript_evidence_after_stage5d.md#defensible-conclusions-and-evidence-strength). Compare explanations to verified literature. Explain why preserving probe discrimination is not sufficient for a competitive CNN output.
- **Likely displays:** Optional concise claim/evidence/limitation table; reuse the ablation table rather than add exploratory analyses.

## 12. Limitations

- **Existing evidence:** Confirmation-selected labels; ephemeris-conditioned views; retained QC failures and scientific preprocessing warnings; fixed unoptimized configurations; overlapping repeated splits; sequential validation and held-out partition reuse; endpoint-only probes; no external-domain test.
- **Needs writing/research:** Explain scope without converting absence of an observed failure into a universal guarantee. Explicitly avoid independent-observation significance claims, equivalence claims and a sole-cause compression claim. Distinguish same-environment replay from new training replications.
- **Likely displays:** A compact limitations table if needed; no new figures required.

## 13. Future work

- **Existing evidence:** [Research decision](post_stage5d_research_decision.md) comparing further CNN architecture work, consolidation of the Kepler paper, and Kepler-to-TESS robustness work.
- **Needs writing/research:** Recommend manuscript consolidation first. Describe a separately designed future domain-robustness study only as a proposal, with independent labels, mission-specific preprocessing, host-aware separation and a newly frozen evaluation plan.
- **Likely displays:** None required. No new experiment or transfer study is authorized by this outline.

## 14. Conclusion

- **Existing evidence:** A certified common-input benchmark and a controlled failure-diagnosis chain support a bounded conclusion: fixing observed collapse and preserving linear-probe discrimination do not by themselves yield competitive held-out CNN performance.
- **Needs writing/research:** Write a concise conclusion after literature comparison and editorial revision; retain the modest test gain and strong RF benchmark. Do not claim deployment readiness, general deep-learning failure, or literature priority.
- **Likely displays:** None; refer to the principal results table.
