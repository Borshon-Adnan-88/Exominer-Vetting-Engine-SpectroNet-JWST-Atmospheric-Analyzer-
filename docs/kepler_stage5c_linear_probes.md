# Stage 5C: short linear-probe localization

**Conclusion: B: fusion/compression.** This concerns validation-set linear accessibility under one fixed probe protocol, not loss of all possible information.

Six probes only, across the same 2 strategies x 10 repeats: raw global (2,000), raw local (200), global Flatten (2,560), local Flatten (1,600), concatenated branches (4,160), and post-LeakyReLU fusion hidden (32). CNNs are frozen; inference disables dropout. Every scaler and logistic regression is fitted on its run's training rows only. No test rows participate.

StandardScaler plus the existing Stage 4B L2 logistic-regression settings (C=1, lbfgs, tolerance 1e-4, max_iter=2,000, no class weighting); no tuning or extra probes. The already completed raw-concatenated Stage 4B logistic-regression validation score is reused as a like-for-like reference with identical seeds, training rows, scaling and regularization settings.

## Six-probe validation ROC-AUC

| strategy | probe | count | mean | median | std | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| host_group_stratified | concatenated_branch | 10 | 0.843062 | 0.842074 | 0.014221 | 0.821773 | 0.869697 |
| host_group_stratified | fusion_hidden | 10 | 0.734853 | 0.719154 | 0.073640 | 0.644666 | 0.875643 |
| host_group_stratified | global_branch | 10 | 0.519388 | 0.509664 | 0.031895 | 0.502525 | 0.608937 |
| host_group_stratified | local_branch | 10 | 0.841841 | 0.839631 | 0.014309 | 0.821886 | 0.869710 |
| host_group_stratified | raw_global | 10 | 0.660717 | 0.659404 | 0.014919 | 0.637786 | 0.679246 |
| host_group_stratified | raw_local | 10 | 0.787969 | 0.787804 | 0.020002 | 0.754102 | 0.824328 |
| object_stratified | concatenated_branch | 10 | 0.813430 | 0.823694 | 0.043822 | 0.697111 | 0.858776 |
| object_stratified | fusion_hidden | 10 | 0.709696 | 0.710484 | 0.047896 | 0.648359 | 0.772308 |
| object_stratified | global_branch | 10 | 0.527683 | 0.510888 | 0.049975 | 0.504188 | 0.668384 |
| object_stratified | local_branch | 10 | 0.800613 | 0.823802 | 0.080962 | 0.574607 | 0.858772 |
| object_stratified | raw_global | 10 | 0.668648 | 0.672485 | 0.017302 | 0.632719 | 0.685577 |
| object_stratified | raw_local | 10 | 0.782615 | 0.792375 | 0.020380 | 0.749256 | 0.805569 |

## Paired loss localization

Positive loss means lower ROC-AUC downstream. Branch loss is existing raw-concatenated LR minus concatenated-branch LR. Fusion loss is concatenated-branch LR minus hidden LR. Their difference compares the two losses without selecting the better raw branch after seeing results. Global/local losses are supporting checks. All 20 comparisons are retained.

| strategy | repeat_id | global_branch_loss | local_branch_loss | joint_branch_loss | fusion_loss | branch_minus_fusion_loss |
| --- | --- | --- | --- | --- | --- | --- |
| host_group_stratified | 0 | 0.153451 | -0.035299 | -0.068490 | -0.015651 | -0.052839 |
| host_group_stratified | 1 | 0.164750 | -0.079549 | -0.104856 | 0.188864 | -0.293719 |
| host_group_stratified | 2 | 0.137121 | -0.047336 | -0.069680 | 0.160859 | -0.230539 |
| host_group_stratified | 3 | 0.028849 | -0.042332 | -0.144259 | 0.128607 | -0.272866 |
| host_group_stratified | 4 | 0.162024 | -0.036514 | -0.039142 | 0.086393 | -0.125535 |
| host_group_stratified | 5 | 0.158237 | -0.048622 | -0.085895 | 0.108693 | -0.194588 |
| host_group_stratified | 6 | 0.163799 | -0.042151 | -0.058977 | 0.038914 | -0.097891 |
| host_group_stratified | 7 | 0.139970 | -0.064102 | -0.122694 | 0.130135 | -0.252829 |
| host_group_stratified | 8 | 0.155281 | -0.080536 | -0.121438 | 0.193553 | -0.314992 |
| host_group_stratified | 9 | 0.149806 | -0.062281 | -0.138788 | 0.061721 | -0.200509 |
| object_stratified | 0 | 0.178877 | -0.038748 | -0.054099 | 0.151992 | -0.206091 |
| object_stratified | 1 | -0.020810 | 0.223398 | 0.070301 | 0.046758 | 0.023543 |
| object_stratified | 2 | 0.125181 | -0.020873 | -0.079470 | 0.111405 | -0.190875 |
| object_stratified | 3 | 0.152187 | -0.032691 | -0.027498 | 0.077216 | -0.104713 |
| object_stratified | 4 | 0.178965 | -0.081061 | -0.114323 | 0.058387 | -0.172710 |
| object_stratified | 5 | 0.154566 | -0.040188 | -0.084235 | 0.099919 | -0.184154 |
| object_stratified | 6 | 0.165527 | -0.027170 | -0.085960 | 0.118156 | -0.204116 |
| object_stratified | 7 | 0.146167 | -0.078782 | -0.127874 | 0.113609 | -0.241483 |
| object_stratified | 8 | 0.166222 | -0.030667 | -0.075767 | 0.173432 | -0.249199 |
| object_stratified | 9 | 0.162767 | -0.053203 | -0.127291 | 0.086468 | -0.213759 |

| strategy | comparison | count | mean | median | std | min | max | positive | negative | tied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| host_group_stratified | branch_minus_fusion_loss | 10 | -0.203631 | -0.215524 | 0.087283 | -0.314992 | -0.052839 | 0 | 10 | 0 |
| host_group_stratified | fusion_loss | 10 | 0.108209 | 0.118650 | 0.066865 | -0.015651 | 0.193553 | 9 | 1 | 0 |
| host_group_stratified | global_branch_loss | 10 | 0.141329 | 0.154366 | 0.040623 | 0.028849 | 0.164750 | 10 | 0 | 0 |
| host_group_stratified | joint_branch_loss | 10 | -0.095422 | -0.095375 | 0.036132 | -0.144259 | -0.039142 | 0 | 10 | 0 |
| host_group_stratified | local_branch_loss | 10 | -0.053872 | -0.047979 | 0.016776 | -0.080536 | -0.035299 | 0 | 10 | 0 |
| object_stratified | branch_minus_fusion_loss | 10 | -0.174356 | -0.197495 | 0.080235 | -0.249199 | 0.023543 | 1 | 9 | 0 |
| object_stratified | fusion_loss | 10 | 0.103734 | 0.105662 | 0.039256 | 0.046758 | 0.173432 | 10 | 0 | 0 |
| object_stratified | global_branch_loss | 10 | 0.140965 | 0.158666 | 0.059025 | -0.020810 | 0.178965 | 9 | 1 | 0 |
| object_stratified | joint_branch_loss | 10 | -0.070621 | -0.081852 | 0.058610 | -0.127874 | 0.070301 | 1 | 9 | 0 |
| object_stratified | local_branch_loss | 10 | -0.017999 | -0.035720 | 0.087289 | -0.081061 | 0.223398 | 1 | 9 | 0 |

Branch loss exceeds fusion loss in 1/20 pairs; the reverse holds in 19/20. The predeclared descriptive rule requires the same mean direction and a majority of pairs within each strategy; it is not a statistical significance claim. Repeated splits overlap, so no independent-observation confidence interval is claimed.

## One next experiment

Retain the branches and test one wider fusion hidden layer, leaving the rest of the protocol frozen. This is a recommendation only; no architecture experiment was started.

## Limits and stop

Frozen representations, different feature dimensions and fixed regularization can change linear accessibility without proving irreversible information destruction. Standardization removes marginal scale differences but does not make the hypothesis classes identical. Probes diagnose the learned endpoints and do not establish when a loss arose. Only validation results were used to support the next hypothesis; a future experiment needs a predeclared evaluation plan.

No Random Forest probes, PCA, correlation matrices, morphology analysis or secondary study. Stage 5C stops here.

Evidence: [protocol](../data/stage5c/protocol.json), [all probe metrics](../data/stage5c/metrics.csv), [all paired comparisons](../data/stage5c/paired_comparisons.csv), [conclusion](../data/stage5c/conclusion.json). Per-run receipts and compressed validation predictions permit audit without rerunning fits.
