# Results: frozen Kepler cohort through Stage 5D

This Results draft reports completed certified experiments only. It contains no abstract. Values are transcribed from certified aggregate tables; no predictions were rescored, models retrained, probes rerun, or scientific settings changed for this synthesis.

## Cohort, representation and evaluation scope

The frozen confirmation-selected Kepler DR25 cohort comprised 6,637 KOIs from 5,804 hosts, including 2,674 confirmed objects and 3,963 false positives. All objects had constructible global and local flux views. The primary analyses retained 6,354 QC-passing objects and 283 objects failing predefined QC thresholds; QC-pass analyses were sensitivity analyses rather than replacement primary cohorts. Global and local inputs contained 2,000 and 200 phase-folded flux values, respectively. The classical models used their concatenation and the CNNs used separate global and local branches. Archive dispositions and other target-encoding metadata were excluded from model features, although the views were constructed using frozen catalogue ephemerides.

Each model configuration was evaluated under ten paired object-stratified and host-group-stratified assignments, with fixed 70/15/15 training/validation/test allocation. Host-group assignments kept every target host within one partition. CNN checkpoints were selected by the earliest minimum validation BCE, rather than by maximum validation AUC or test performance. The test set remained outside gradient updates and checkpoint selection. The held-out partitions were reused across completed model-development stages, however; the Stage 5D variant's first frozen test evaluation should not be interpreted as a new independent confirmation dataset for the entire research program. [Cohort and preprocessing evidence](kepler_stage4a_certification_report.md); [split policy](kepler_split_policy.md).

## Classical baselines and principal held-out comparison

Random Forest achieved mean test ROC-AUC of 0.880196 under object splitting and 0.879338 under host-group splitting (Table 1). Logistic Regression attained 0.751890 and 0.752666, respectively. Random Forest also had higher average precision and lower Brier scores than the other evaluated models under both strategies. Its advantage therefore extended beyond ROC ranking in this fixed protocol. The CNN variants did not match either classical model in ROC-AUC or average precision. Logistic Regression did not dominate the later CNNs in Brier score, illustrating that discrimination and squared probability error need not rank models identically.

**Table 1. Principal held-out test results.** Values are means ± sample SD across ten repeats for each model/strategy on the complete assigned test population (`track=primary`). The SD describes between-repeat dispersion, not a confidence interval or a standard error from independent datasets. Average precision is for label 1 (confirmed); it is not trapezoidal PR-AUC. Higher AUC/AP and lower Brier are better. No validation or probe result appears in this table.

| Model | Split strategy | Test ROC-AUC | Test average precision | Test Brier |
| --- | --- | --- | --- | --- |
| Logistic Regression | Object | 0.751890 ± 0.027077 | 0.583295 ± 0.032360 | 0.207517 ± 0.010824 |
| Logistic Regression | Host-group | 0.752666 ± 0.017619 | 0.584336 ± 0.023644 | 0.207052 ± 0.006859 |
| Random Forest | Object | 0.880196 ± 0.011750 | 0.812993 ± 0.020434 | 0.145925 ± 0.004441 |
| Random Forest | Host-group | 0.879338 ± 0.010507 | 0.808660 ± 0.015470 | 0.146461 ± 0.004587 |
| Stage 5: ReLU32 | Object | 0.609158 ± 0.082566 | 0.461919 ± 0.055816 | 0.213849 ± 0.018426 |
| Stage 5: ReLU32 | Host-group | 0.600290 ± 0.088651 | 0.459597 ± 0.054054 | 0.217524 ± 0.019982 |
| Stage 5B: LeakyReLU32 | Object | 0.641803 ± 0.031751 | 0.469417 ± 0.040364 | 0.202710 ± 0.003216 |
| Stage 5B: LeakyReLU32 | Host-group | 0.639775 ± 0.020623 | 0.467661 ± 0.021775 | 0.202149 ± 0.002631 |
| Stage 5D: LeakyReLU128 | Object | 0.646314 ± 0.027160 | 0.475296 ± 0.031608 | 0.202174 ± 0.003018 |
| Stage 5D: LeakyReLU128 | Host-group | 0.668214 ± 0.040078 | 0.502270 ± 0.037547 | 0.200790 ± 0.004097 |

Sources: Stage 4B [repeat summaries](../data/stage4b/repeat_summary.csv), Stage 5 [repeat summaries](../data/stage5/repeat_summary.csv), Stage 5B [repeat summaries](../data/stage5b/repeat_summary.csv), and Stage 5D [test summaries](../data/stage5d_test/test_summary.csv). These are existing certified aggregate results; each displayed mean and sample SD is a transcription, not a new pooled prediction estimate.

## Split sensitivity

Object-minus-host-group test ROC-AUC averaged -0.000775 for Logistic Regression and +0.000858 for Random Forest. The respective sample SDs of paired differences were 0.029217 and 0.016787. Thus the classical models showed small mean differences without a common direction of benefit for object splitting. The original CNN difference was +0.008868 (SD 0.117130), falling to +0.002028 (SD 0.043522) after the activation intervention. Width128 instead yielded -0.021900 (SD 0.056905), favoring host-group splitting on average; object splitting was higher in four of ten width128 repeats.

These comparisons concern different partitioning protocols with changing membership and composition. Small mean differences do not demonstrate equivalence, and larger differences do not isolate the causal effect of host overlap. The repeats share a cohort and are not independent datasets. Consequently, these findings are reported descriptively without independence-based confidence intervals or post-hoc significance thresholds. Sources: [Stage 4B split comparison](kepler_stage4b_report.md), [original CNN paired summary](../data/stage5/paired_summary.csv), [Stage 5B paired summary](../data/stage5b/paired_summary.csv), and [Stage 5D split summary](../data/stage5d_test/object_group_summary.csv).

## Original CNN collapse and the activation intervention

The original dual-view CNN had a 32-unit ReLU fusion representation after concatenating 4,160 branch features. Its mean test ROC-AUC was 0.609158 under object splitting and 0.600290 under host-group splitting, with substantial repeat-to-repeat dispersion. Endpoint diagnostics identified four severe probability-collapse runs and seven runs with at most three distinct predicted probabilities, including the four severe cases. These overlapping counts must not be added. In all seven low-diversity cases, the inspected training sample had an entirely zero hidden representation. Variable upstream representations, constant sampled logits, finite nonzero stored weights, near-prior probabilities, and training histories supported localization to the fusion activation endpoint. They did not establish a complete causal history of optimization or prove that every full-cohort hidden vector was zero.

Replacing fusion ReLU with LeakyReLU(0.01), while retaining width 32 and the other frozen scientific settings, removed the observed severe, low-diversity and entirely zero sampled fusion flags in all twenty Stage 5B fits. The directly comparable full-cohort output checks also showed zero severe and low-diversity runs. Mean test ROC-AUC rose to 0.641803 and 0.639775 for object and host-group splitting, respectively. Dispersion was lower than for the original CNN, but the model remained substantially below Random Forest. The intervention therefore resolved an observed stability failure without establishing that instability alone explained the original performance deficit. [Endpoint diagnosis](kepler_stage5a_failure_analysis.md); [activation intervention](kepler_stage5b_activation_ablation.md).

**Table 2. Ablation and diagnostic chain.** Stages 5A and 5C are diagnostic analyses, not independently trained competing classifiers. Probe results are validation-only.

| Stage | Intervention or role | Principal observation | Bounded interpretation |
| --- | --- | --- | --- |
| 5 | Original dual-view CNN; 32-unit ReLU fusion | Four severe and seven overlapping low-diversity runs; test AUC 0.609158 object / 0.600290 host-group | Poor performance with an identifiable endpoint failure in a subset of fits |
| 5A | Read-only collapse diagnosis | Sampled zero fusion states despite variable upstream features and finite stored weights | Activation-related endpoint localization; training-path mechanism remains unresolved |
| 5B | Fusion LeakyReLU(0.01), width 32 | Observed collapse flags eliminated in 20/20 fits; test AUC 0.641803 / 0.639775 | Stability improves; predictive gains remain modest relative to the benchmark gap |
| 5C | Fixed training-fitted linear probes; no CNN retraining or test evaluation | Concatenated branches retain strong validation discrimination; mean concat-minus-hidden probe AUC gap 0.105972 | Major loss of linear accessibility localized primarily to fusion/compression under the fixed probe protocol |
| 5D | Fusion width 32 to 128 with LeakyReLU retained | Probe gap reduced by 97.9%, improving 19/20 pairs; test AUC 0.646314 / 0.668214 | Representation preservation improves strongly, held-out classification modestly; RF remains substantially stronger |

## Representation diagnostics and fusion width

Stage 5C evaluated fixed linear probes on frozen Stage 5B representations. Probe standardization and classifier fitting used only training rows, and scoring used validation rows. The local learned branch and concatenated branches retained discriminative structure: mean concatenated-probe ROC-AUC across twenty runs was 0.828246, compared with 0.722274 after the 32-unit hidden representation. The signed concat-minus-hidden gap averaged 0.105972. This was evidence for a substantial loss of linear accessibility at fusion rather than a general absence of useful learned branch features. It did not demonstrate irreversible information destruction; differing dimensionalities and fixed regularization also affect probe performance.

Stage 5D widened fusion to 128 units, increasing the parameter count from 138,561 to 538,113. Branch design, activation, dropout, optimizer, learning rate, clipping, batch size, early stopping, seeds, splits and QC policy were retained. Both branches continued to train end-to-end. Mean hidden-probe validation ROC-AUC increased to 0.826806, near the width128 concatenated-probe value of 0.829044. The signed gap decreased to 0.002238, corresponding to a mean paired reduction of 0.103734, or 97.9% of the original mean gap; nineteen of twenty pairs improved. No new severe or low-diversity outputs were observed. Seven width128 runs reached the epoch cap, compared with four width32 LeakyReLU runs; reaching the cap is not a claim of convergence.

**Table 3. Validation-only representation diagnostic.** These are means over twenty matched runs, not held-out test metrics. A negative per-run signed gap is possible if a hidden probe outperforms its concatenated probe. The percent reduction is a ratio of aggregate mean gaps, not a universal per-run percentage or an information-theoretic quantity.

| Probe quantity | Stage 5B, width32 | Stage 5D, width128 |
| --- | --- | --- |
| Concatenated-branch validation ROC-AUC | 0.828246 | 0.829044 |
| Hidden validation ROC-AUC | 0.722274 | 0.826806 |
| Mean signed concat-minus-hidden AUC gap | 0.105972 | 0.002238 |

Source: [certified representation summary](../data/stage5d/representation_probe_summary.csv) and [paired fusion-gap comparison](../data/stage5d/fusion_gap_comparison.csv). No probe was rerun for the held-out test analysis or this draft.

## Frozen width128 held-out evaluation

The twenty predeclared width128 models were evaluated once on their original test partitions after a validation-only GO decision. An independent fresh-process replay verified saved identities, labels, probabilities, classifications and metrics; replay was a technical verification of the same evaluation, not an additional scientific replicate. The threshold remained 0.5, with no recalibration, test-driven selection or post-test adjustment.

Width128 improved mean test ROC-AUC over Stage 5B by +0.004511 under object splitting and +0.028439 under host-group splitting. The overall mean paired difference was +0.016475; twelve runs improved, eight worsened and none tied. The respective strategy counts were 5/5/0 and 7/3/0. The object paired median was -0.002194 despite its positive mean, whereas the host-group median was +0.030428. Thus the small object gain should not be described as consistent across repeats. Table 4 reports the activation and width contrasts from the certified paired summaries.

**Table 4. Paired held-out ROC-AUC differences.** Counts refer to actual AUC improvements; contrasts retain every matched run and do not substitute maximum-AUC epochs.

| Intervention contrast | Strategy | Mean paired test AUC difference | Sample SD | Improved/worsened/tied |
| --- | --- | --- | --- | --- |
| 5B minus 5 | Object | +0.032645 | 0.084090 | 6/4/0 |
| 5B minus 5 | Host-group | +0.039485 | 0.088189 | 5/5/0 |
| 5D minus 5B | Object | +0.004511 | 0.030107 | 5/5/0 |
| 5D minus 5B | Host-group | +0.028439 | 0.035524 | 7/3/0 |

The direction of the mean validation gains reproduced on test under both strategies. Object validation gain was +0.003958 versus test +0.004511; host-group gain was +0.030221 versus test +0.028439. Overall gain decreased slightly from +0.017089 on validation to +0.016475 on test. This supports a modest generalization of the width intervention, especially for host-group splits, rather than a large recovery in end-to-end predictive quality. [Validation-to-test comparison](../data/stage5d_test/validation_to_test.csv).

Random Forest remained stronger in every matched width128 test AUC comparison. The overall width128-minus-RF difference was -0.222503, comprising -0.233881 for object and -0.211124 for host-group splitting. Width128 also remained below Logistic Regression in test AUC in all twenty comparisons, with an overall difference of -0.095014. These comparisons use previously certified classical results without refitting the baselines. Together, the results support interpretation category B in the frozen test report: representation preservation improved strongly, but predictive improvement was modest. The fusion bottleneck is not established as the sole cause of CNN underperformance. [Frozen test evaluation](kepler_stage5d_test_evaluation.md).

## QC sensitivity and probability quality

For width128, the mean object test denominator fell from 993 to 950 under the predefined QC-pass subset, excluding 43 objects per repeat on average. Mean host-group denominator fell from 994.5 to 954.7, excluding 39.8. Mean positive prevalence rose from approximately 0.402 to 0.415 in both strategies. QC-pass ROC-AUC was 0.635295 object and 0.658961 host-group, compared with primary values 0.646314 and 0.668214. Excluding QC failures therefore did not improve mean AUC in this fixed sensitivity analysis; population and prevalence changes must be retained when interpreting it.

Primary mean Brier scores were 0.202174 object and 0.200790 host-group; QC-pass values were 0.205566 and 0.204143. These remained above the Random Forest primary scores, 0.145925 and 0.146461. Brier combines aspects of discrimination and calibration, so this ranking alone does not isolate a calibration mechanism. Reliability summaries used ten fixed equal-width bins without fitting a recalibration model. Repeated objects can contribute across repeats to the pooled descriptive reliability figure. [QC sensitivity](../data/stage5d_test/qc_sensitivity.csv); [calibration bins](../data/stage5d_test/calibration_bins.csv).

## Scope of inference

The completed experiments support a controlled distinction between numerical/activation stability, linear accessibility of learned representations, and actual classification performance. They do not establish a causal host-leakage effect, a universal architectural failure, optimal performance by model family, or external-domain robustness. All conclusions concern the frozen confirmation-selected DR25 KOI cohort, ephemeris-conditioned flux views, fixed models, and reused partition framework. Statistical and literature novelty claims require separate justification; no such claim is introduced by this draft.
