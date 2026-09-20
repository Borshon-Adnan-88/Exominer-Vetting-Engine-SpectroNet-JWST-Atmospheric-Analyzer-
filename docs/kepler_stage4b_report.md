# Stage 4B baseline experiment report

This is a supervised command-line experiment on the certified Kepler DR25 cohort. The Streamlit application remains a separate synthetic prototype.

**Status: complete; 40 fits and independent prediction replay verified.**

Frozen-input validation passed: **6,637 KOIs, 5,804 hosts, 2,200 flux features**. The primary cohort retains all 6,354 QC passes and 283 QC failures. All canonical file hashes, 39,822 logical hashes, shapes/dtypes, finite values and identity joins passed. All 132,740 Stage 3 assignment rows remain unchanged; grouped host separation passed.

## Primary scientific comparison

The primary outcome is object-level minus host-group **test ROC-AUC**, paired by the ten frozen repeat IDs. Means below are across repeats, not metrics pooled over repeated observations. Host seen/unseen results do not enter this comparison.

| model | count | mean | median | std | min | max | range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 10 | -0.00077545702 | 0.0091455333 | 0.029216532 | -0.062183573 | 0.028277425 | 0.090460998 |
| random_forest | 10 | 0.00085783765 | -0.0025413755 | 0.016786878 | -0.020472933 | 0.023161814 | 0.043634747 |

`logistic_regression`: object-level test ROC-AUC is higher in 7/10 repeats and lower in 3/10; mean paired difference -0.000775.

`random_forest`: object-level test ROC-AUC is higher in 5/10 repeats and lower in 5/10; mean paired difference +0.000858.

These are descriptive split-protocol comparisons, not a causal estimate of leakage. Training/test membership, host composition and class prevalence differ. Repeats share the same cohort and are not independent datasets; no independence-based confidence interval or significance claim is made. The secondary host diagnostic cannot establish or override the primary conclusion.

## Fixed models and runtime

Logistic regression is a single training-only StandardScaler → LogisticRegression Pipeline. Scaler fit-call tests prove that validation/test and full-cohort fitting do not occur. Random Forest uses the identical frozen flux representation without scaling. All settings below were frozen before fitting; no tuning occurred.

```json
{
  "logistic_regression": {
    "C": 1.0,
    "class_weight": null,
    "fit_intercept": true,
    "l1_ratio": 0.0,
    "max_iter": 2000,
    "solver": "lbfgs",
    "tol": 0.0001
  },
  "random_forest": {
    "bootstrap": true,
    "class_weight": null,
    "criterion": "gini",
    "max_depth": 12,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_estimators": 200,
    "n_jobs": 1
  }
}
```

Forest random states are the Stage 3 seeds, shared between strategies within each repeat. The deterministic lbfgs solver has no stochastic seed operation. Decision threshold is p(confirmed) >= 0.5. No class-weight search, PCA, imputation, threshold tuning or post-hoc calibration is used.

All 20 logistic fits converged; iteration counts range from 212 to 336. Forest convergence is not applicable.

Total experiment runtime: **1070.509 seconds**, including 344.696 seconds for input validation, plus **19.320 seconds** for the separate independent replay. Timers start inside the commands after library imports; test runtime is separate.

Process peak RAM: **558,800,896 bytes (532.91 MiB)**. CPU execution was sequential with one numerical thread. Family RAM peaks are 50-ms samples of total process RSS, not incremental estimator allocations.

| model | fit_predict_serialize_seconds | fit_seconds | fits | model_artifact_bytes | sampled_peak_rss_bytes |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 163.06901 | 143.20094 | 20 | 1301790 | 558800896 |
| random_forest | 547.47289 | 498.45254 | 20 | 36385206 | 511971328 |

Iteration counts, fit runtimes, model sizes and per-run hashes are recorded in [run_manifest.json](../data/stage4b/run_manifest.json).

## Test metrics for every model, strategy and repeat

Positive label is confirmed planet. Confusion columns are TN, FP, FN, TP. Average precision is non-trapezoidal; trapezoidal PR-AUC is separate. Brier is mean squared binary probability error (lower is better).

| model | strategy | repeat_id | sample_count | positive_count | negative_count | positive_prevalence | roc_auc | average_precision | pr_auc_trapezoidal | precision | recall | f1 | accuracy | brier | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | object_stratified | 0 | 993 | 399 | 594 | 0.40181269 | 0.72472849 | 0.54408847 | 0.54222215 | 0.58552632 | 0.66917293 | 0.6245614 | 0.67673716 | 0.21985243 | 405 | 189 | 132 | 267 |
| logistic_regression | host_group_stratified | 0 | 994 | 400 | 594 | 0.40241449 | 0.76145202 | 0.59700682 | 0.59485288 | 0.61870504 | 0.645 | 0.63157895 | 0.6971831 | 0.20474083 | 435 | 159 | 142 | 258 |
| logistic_regression | object_stratified | 1 | 993 | 399 | 594 | 0.40181269 | 0.76719999 | 0.60955673 | 0.60758687 | 0.64130435 | 0.5914787 | 0.61538462 | 0.70292044 | 0.20269505 | 462 | 132 | 163 | 236 |
| logistic_regression | host_group_stratified | 1 | 993 | 400 | 593 | 0.40281974 | 0.74876476 | 0.57712134 | 0.57518613 | 0.6019656 | 0.6125 | 0.60718711 | 0.68076536 | 0.20602623 | 431 | 162 | 155 | 245 |
| logistic_regression | object_stratified | 2 | 993 | 399 | 594 | 0.40181269 | 0.79157068 | 0.63026721 | 0.62964555 | 0.66903073 | 0.70927318 | 0.68856448 | 0.74219537 | 0.19137938 | 454 | 140 | 116 | 283 |
| logistic_regression | host_group_stratified | 2 | 994 | 400 | 594 | 0.40241449 | 0.78522306 | 0.62556636 | 0.62374292 | 0.6761658 | 0.6525 | 0.66412214 | 0.73440644 | 0.19630126 | 469 | 125 | 139 | 261 |
| logistic_regression | object_stratified | 3 | 993 | 399 | 594 | 0.40181269 | 0.77650355 | 0.61095612 | 0.60900373 | 0.66997519 | 0.67669173 | 0.67331671 | 0.73615307 | 0.20122402 | 461 | 133 | 129 | 270 |
| logistic_regression | host_group_stratified | 3 | 995 | 401 | 594 | 0.40301508 | 0.75503581 | 0.59070311 | 0.58874641 | 0.60655738 | 0.64588529 | 0.62560386 | 0.68844221 | 0.20005155 | 426 | 168 | 142 | 259 |
| logistic_regression | object_stratified | 4 | 993 | 399 | 594 | 0.40181269 | 0.77506899 | 0.61039911 | 0.60845561 | 0.63205418 | 0.70175439 | 0.66508314 | 0.71601208 | 0.19967579 | 431 | 163 | 119 | 280 |
| logistic_regression | host_group_stratified | 4 | 995 | 401 | 594 | 0.40301508 | 0.75757996 | 0.59250513 | 0.59060563 | 0.63701923 | 0.66084788 | 0.64871481 | 0.71155779 | 0.21048801 | 443 | 151 | 136 | 265 |
| logistic_regression | object_stratified | 5 | 993 | 399 | 594 | 0.40181269 | 0.74208459 | 0.56404066 | 0.56291827 | 0.59912854 | 0.68922306 | 0.64102564 | 0.6898288 | 0.21353533 | 410 | 184 | 124 | 275 |
| logistic_regression | host_group_stratified | 5 | 996 | 401 | 595 | 0.40261044 | 0.76077453 | 0.58575408 | 0.58392286 | 0.62124711 | 0.67082294 | 0.64508393 | 0.70281124 | 0.2043356 | 431 | 164 | 132 | 269 |
| logistic_regression | object_stratified | 6 | 993 | 399 | 594 | 0.40181269 | 0.70232821 | 0.53794415 | 0.53593032 | 0.54123711 | 0.52631579 | 0.53367217 | 0.63041289 | 0.22471636 | 416 | 178 | 189 | 210 |
| logistic_regression | host_group_stratified | 6 | 994 | 400 | 594 | 0.40241449 | 0.76451178 | 0.6069175 | 0.60452402 | 0.63613861 | 0.6425 | 0.63930348 | 0.7082495 | 0.20465627 | 447 | 147 | 143 | 257 |
| logistic_regression | object_stratified | 7 | 993 | 399 | 594 | 0.40181269 | 0.74049602 | 0.56215045 | 0.56020775 | 0.58695652 | 0.67669173 | 0.62863795 | 0.67875126 | 0.21098452 | 404 | 190 | 129 | 270 |
| logistic_regression | host_group_stratified | 7 | 996 | 401 | 595 | 0.40261044 | 0.72855257 | 0.54908183 | 0.54738445 | 0.59574468 | 0.62842893 | 0.61165049 | 0.67871486 | 0.22049588 | 424 | 171 | 149 | 252 |
| logistic_regression | object_stratified | 8 | 993 | 399 | 594 | 0.40181269 | 0.75982043 | 0.59927371 | 0.5971066 | 0.60784314 | 0.62155388 | 0.61462206 | 0.68680765 | 0.19645279 | 434 | 160 | 151 | 248 |
| logistic_regression | host_group_stratified | 8 | 993 | 400 | 593 | 0.40281974 | 0.731543 | 0.56136287 | 0.5595864 | 0.57494407 | 0.6425 | 0.6068477 | 0.66465257 | 0.21041854 | 403 | 190 | 143 | 257 |
| logistic_regression | object_stratified | 9 | 993 | 399 | 594 | 0.40181269 | 0.73910365 | 0.56427582 | 0.56254373 | 0.60880196 | 0.62406015 | 0.61633663 | 0.6878147 | 0.21465653 | 434 | 160 | 150 | 249 |
| logistic_regression | host_group_stratified | 9 | 995 | 401 | 594 | 0.40301508 | 0.73322166 | 0.55733832 | 0.55557284 | 0.59684685 | 0.66084788 | 0.62721893 | 0.68341709 | 0.21300949 | 415 | 179 | 136 | 265 |
| random_forest | object_stratified | 0 | 993 | 399 | 594 | 0.40181269 | 0.87974144 | 0.81939287 | 0.81906776 | 0.735 | 0.73684211 | 0.7359199 | 0.78751259 | 0.14620864 | 488 | 106 | 105 | 294 |
| random_forest | host_group_stratified | 0 | 994 | 400 | 594 | 0.40241449 | 0.87727273 | 0.81985741 | 0.81956983 | 0.72705882 | 0.7725 | 0.74909091 | 0.7917505 | 0.14871924 | 478 | 116 | 91 | 309 |
| random_forest | object_stratified | 1 | 993 | 399 | 594 | 0.40181269 | 0.85989806 | 0.77219052 | 0.77171468 | 0.70343137 | 0.71929825 | 0.71127633 | 0.7653575 | 0.15161889 | 473 | 121 | 112 | 287 |
| random_forest | host_group_stratified | 1 | 993 | 400 | 593 | 0.40281974 | 0.88037099 | 0.81485736 | 0.81444108 | 0.74623116 | 0.7425 | 0.7443609 | 0.79456193 | 0.14529764 | 492 | 101 | 103 | 297 |
| random_forest | object_stratified | 2 | 993 | 399 | 594 | 0.40181269 | 0.89704058 | 0.83060506 | 0.83026286 | 0.76201923 | 0.79448622 | 0.77791411 | 0.81772407 | 0.13788428 | 495 | 99 | 82 | 317 |
| random_forest | host_group_stratified | 2 | 994 | 400 | 594 | 0.40241449 | 0.8785101 | 0.80094262 | 0.80045673 | 0.71719457 | 0.7925 | 0.75296912 | 0.79074447 | 0.14565894 | 469 | 125 | 83 | 317 |
| random_forest | object_stratified | 3 | 993 | 399 | 594 | 0.40181269 | 0.88659781 | 0.81066835 | 0.81027162 | 0.73095238 | 0.76942356 | 0.74969475 | 0.79355488 | 0.14488175 | 481 | 113 | 92 | 307 |
| random_forest | host_group_stratified | 3 | 995 | 401 | 594 | 0.40301508 | 0.86586984 | 0.79176931 | 0.79132337 | 0.70412844 | 0.76558603 | 0.73357228 | 0.7758794 | 0.15160823 | 465 | 129 | 94 | 307 |
| random_forest | object_stratified | 4 | 993 | 399 | 594 | 0.40181269 | 0.88720961 | 0.82433113 | 0.82400897 | 0.76485788 | 0.74185464 | 0.75318066 | 0.80463243 | 0.14411102 | 503 | 91 | 103 | 296 |
| random_forest | host_group_stratified | 4 | 995 | 401 | 594 | 0.40301508 | 0.86404779 | 0.79874565 | 0.79837136 | 0.69905213 | 0.73566085 | 0.71688943 | 0.76582915 | 0.15409097 | 467 | 127 | 106 | 295 |
| random_forest | object_stratified | 5 | 993 | 399 | 594 | 0.40181269 | 0.88256415 | 0.8268465 | 0.82657072 | 0.74754902 | 0.76441103 | 0.755886 | 0.80161128 | 0.14531098 | 491 | 103 | 94 | 305 |
| random_forest | host_group_stratified | 5 | 996 | 401 | 595 | 0.40261044 | 0.89451581 | 0.82692014 | 0.82645651 | 0.7453271 | 0.79551122 | 0.76960193 | 0.80823293 | 0.14090808 | 486 | 109 | 82 | 319 |
| random_forest | object_stratified | 6 | 993 | 399 | 594 | 0.40181269 | 0.86661941 | 0.78680889 | 0.78634986 | 0.70601852 | 0.76441103 | 0.73405535 | 0.77744209 | 0.15216136 | 467 | 127 | 94 | 305 |
| random_forest | host_group_stratified | 6 | 994 | 400 | 594 | 0.40241449 | 0.87417088 | 0.79646573 | 0.7959049 | 0.72069825 | 0.7225 | 0.721598 | 0.77565392 | 0.14835888 | 482 | 112 | 111 | 289 |
| random_forest | object_stratified | 7 | 993 | 399 | 594 | 0.40181269 | 0.89051332 | 0.83060967 | 0.83029575 | 0.75294118 | 0.80200501 | 0.77669903 | 0.81470292 | 0.14082117 | 489 | 105 | 79 | 320 |
| random_forest | host_group_stratified | 7 | 996 | 401 | 595 | 0.40261044 | 0.87753306 | 0.78707762 | 0.78650799 | 0.73631841 | 0.73815461 | 0.73723537 | 0.78815261 | 0.14776883 | 489 | 106 | 105 | 296 |
| random_forest | object_stratified | 8 | 993 | 399 | 594 | 0.40181269 | 0.86804975 | 0.79979623 | 0.79944387 | 0.70544554 | 0.71428571 | 0.70983811 | 0.7653575 | 0.14882804 | 475 | 119 | 114 | 285 |
| random_forest | host_group_stratified | 8 | 993 | 400 | 593 | 0.40281974 | 0.88502951 | 0.81923676 | 0.81890817 | 0.7373494 | 0.765 | 0.75092025 | 0.79556898 | 0.14249588 | 484 | 109 | 94 | 306 |
| random_forest | object_stratified | 9 | 993 | 399 | 594 | 0.40181269 | 0.88372446 | 0.82867688 | 0.82821145 | 0.70975057 | 0.78446115 | 0.7452381 | 0.78449144 | 0.1474286 | 466 | 128 | 86 | 313 |
| random_forest | host_group_stratified | 9 | 995 | 401 | 594 | 0.40301508 | 0.89605951 | 0.83073051 | 0.83032109 | 0.76097561 | 0.77805486 | 0.76942047 | 0.8120603 | 0.13970114 | 496 | 98 | 89 | 312 |

The [metrics CSV](../data/stage4b/metrics.csv) contains all 240 rows: 40 fitted models × three partitions × two reporting tracks. Training scores are in-sample diagnostics. Assigned/evaluated/excluded counts, prevalence and undefined-metric flags are retained.

## Repeat summaries and paired differences

Primary test mean/median/sample-standard-deviation/min/max/range:

| model | strategy | metric | count | mean | median | std | min | max | range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | host_group_stratified | accuracy | 10 | 0.69502001 | 0.69281265 | 0.020095108 | 0.66465257 | 0.73440644 | 0.069753871 |
| logistic_regression | host_group_stratified | average_precision | 10 | 0.58433574 | 0.5882286 | 0.023643768 | 0.54908183 | 0.62556636 | 0.076484533 |
| logistic_regression | host_group_stratified | brier | 10 | 0.20705237 | 0.20538353 | 0.0068594298 | 0.19630126 | 0.22049588 | 0.024194617 |
| logistic_regression | host_group_stratified | f1 | 10 | 0.63073114 | 0.62939894 | 0.018991266 | 0.6068477 | 0.66412214 | 0.05727444 |
| logistic_regression | host_group_stratified | pr_auc_trapezoidal | 10 | 0.58241245 | 0.58633463 | 0.023527751 | 0.54738445 | 0.62374292 | 0.076358474 |
| logistic_regression | host_group_stratified | precision | 10 | 0.61653344 | 0.61263121 | 0.028378665 | 0.57494407 | 0.6761658 | 0.10122173 |
| logistic_regression | host_group_stratified | recall | 10 | 0.64618329 | 0.64544264 | 0.016850708 | 0.6125 | 0.67082294 | 0.058322943 |
| logistic_regression | host_group_stratified | roc_auc | 10 | 0.75266592 | 0.75630788 | 0.017619217 | 0.72855257 | 0.78522306 | 0.056670496 |
| logistic_regression | object_stratified | accuracy | 10 | 0.69476334 | 0.68882175 | 0.032213163 | 0.63041289 | 0.74219537 | 0.11178248 |
| logistic_regression | object_stratified | average_precision | 10 | 0.58329524 | 0.58177477 | 0.032360056 | 0.53794415 | 0.63026721 | 0.09232306 |
| logistic_regression | object_stratified | brier | 10 | 0.20751722 | 0.20683979 | 0.010823713 | 0.19137938 | 0.22471636 | 0.033336983 |
| logistic_regression | object_stratified | f1 | 10 | 0.63012048 | 0.62659968 | 0.042954295 | 0.53367217 | 0.68856448 | 0.1548923 |
| logistic_regression | object_stratified | pr_auc_trapezoidal | 10 | 0.58156206 | 0.58001244 | 0.032493473 | 0.53593032 | 0.62964555 | 0.093715231 |
| logistic_regression | object_stratified | precision | 10 | 0.6141858 | 0.60832255 | 0.039937957 | 0.54123711 | 0.66997519 | 0.12873807 |
| logistic_regression | object_stratified | recall | 10 | 0.64862155 | 0.67293233 | 0.057453707 | 0.52631579 | 0.70927318 | 0.18295739 |
| logistic_regression | object_stratified | roc_auc | 10 | 0.75189046 | 0.75095251 | 0.027077079 | 0.70232821 | 0.79157068 | 0.089242466 |
| random_forest | host_group_stratified | accuracy | 10 | 0.78984342 | 0.79124748 | 0.014403469 | 0.76582915 | 0.8120603 | 0.046231156 |
| random_forest | host_group_stratified | average_precision | 10 | 0.80866031 | 0.80789999 | 0.015469843 | 0.78707762 | 0.83073051 | 0.043652887 |
| random_forest | host_group_stratified | brier | 10 | 0.14646078 | 0.14671389 | 0.0045870831 | 0.13970114 | 0.15409097 | 0.01438983 |
| random_forest | host_group_stratified | f1 | 10 | 0.74456587 | 0.74672591 | 0.017776919 | 0.71688943 | 0.76960193 | 0.052712501 |
| random_forest | host_group_stratified | pr_auc_trapezoidal | 10 | 0.8082261 | 0.80744891 | 0.015522811 | 0.78650799 | 0.83032109 | 0.043813099 |
| random_forest | host_group_stratified | precision | 10 | 0.72943339 | 0.73168862 | 0.019535849 | 0.69905213 | 0.76097561 | 0.061923477 |
| random_forest | host_group_stratified | recall | 10 | 0.76079676 | 0.76529302 | 0.025014431 | 0.7225 | 0.79551122 | 0.073011222 |
| random_forest | host_group_stratified | roc_auc | 10 | 0.87933802 | 0.87802158 | 0.010506744 | 0.86404779 | 0.89605951 | 0.032011722 |
| random_forest | object_stratified | accuracy | 10 | 0.79123867 | 0.79053374 | 0.018660177 | 0.7653575 | 0.81772407 | 0.052366566 |
| random_forest | object_stratified | average_precision | 10 | 0.81299261 | 0.821862 | 0.020434056 | 0.77219052 | 0.83060967 | 0.058419154 |
| random_forest | object_stratified | brier | 10 | 0.14592547 | 0.14575981 | 0.0044414667 | 0.13788428 | 0.15216136 | 0.01427708 |
| random_forest | object_stratified | f1 | 10 | 0.74497023 | 0.74746642 | 0.023265525 | 0.70983811 | 0.77791411 | 0.068076003 |
| random_forest | object_stratified | pr_auc_trapezoidal | 10 | 0.81261975 | 0.82153837 | 0.020480861 | 0.77171468 | 0.83029575 | 0.058581064 |
| random_forest | object_stratified | precision | 10 | 0.73179657 | 0.73297619 | 0.024408931 | 0.70343137 | 0.76485788 | 0.061426509 |
| random_forest | object_stratified | recall | 10 | 0.75914787 | 0.76441103 | 0.030339735 | 0.71428571 | 0.80200501 | 0.087719298 |
| random_forest | object_stratified | roc_auc | 10 | 0.88019586 | 0.88314431 | 0.011749643 | 0.85989806 | 0.89704058 | 0.03714252 |

Primary paired test differences (object minus grouped, unchanged direction for every metric; negative Brier difference favours object splitting):

| model | metric | count | mean | median | std | min | max | range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | accuracy | 10 | -0.00025667155 | 0.0044259566 | 0.033337215 | -0.077836607 | 0.04771086 | 0.12554747 |
| logistic_regression | average_precision | 10 | -0.0010404942 | 0.010003062 | 0.035713369 | -0.068973352 | 0.037910836 | 0.10688419 |
| logistic_regression | brier | 10 | 0.00046485539 | -0.0010793545 | 0.011328739 | -0.013965746 | 0.020060093 | 0.034025839 |
| logistic_regression | f1 | 10 | -0.00061066111 | 0.0079859309 | 0.04068369 | -0.10563131 | 0.047712843 | 0.15334415 |
| logistic_regression | pr_auc_trapezoidal | 10 | -0.00085039655 | 0.0098970946 | 0.035501917 | -0.068593701 | 0.037520196 | 0.1061139 |
| logistic_regression | precision | 10 | -0.0023476348 | -0.0060500625 | 0.044081567 | -0.0949015 | 0.063417809 | 0.15831931 |
| logistic_regression | recall | 10 | 0.0024382621 | 0.021286524 | 0.052462746 | -0.11618421 | 0.056773183 | 0.17295739 |
| logistic_regression | roc_auc | 10 | -0.00077545702 | 0.0091455333 | 0.029216532 | -0.062183573 | 0.028277425 | 0.090460998 |
| random_forest | accuracy | 10 | 0.0013952511 | -0.0012248719 | 0.02543433 | -0.03021148 | 0.038803281 | 0.069014762 |
| random_forest | average_precision | 10 | 0.0043322982 | -0.00026908886 | 0.02559316 | -0.042666841 | 0.04353205 | 0.086198891 |
| random_forest | brier | 10 | -0.00053530916 | 0.00064594112 | 0.0069170506 | -0.0099799509 | 0.0077274565 | 0.017707407 |
| random_forest | f1 | 10 | 0.00040436804 | -0.00035682836 | 0.029133584 | -0.041082138 | 0.039463662 | 0.0805458 |
| random_forest | pr_auc_trapezoidal | 10 | 0.0043936514 | -0.00019392945 | 0.025667692 | -0.0427264 | 0.043787758 | 0.086514158 |
| random_forest | precision | 10 | 0.0023631796 | 0.0050815466 | 0.038059627 | -0.051225043 | 0.065805748 | 0.11703079 |
| random_forest | recall | 10 | -0.0016488884 | 0.0029118698 | 0.035360465 | -0.050714286 | 0.063850399 | 0.11456468 |
| random_forest | roc_auc | 10 | 0.00085783765 | -0.0025413755 | 0.016786878 | -0.020472933 | 0.023161814 | 0.043634747 |

All ten object/grouped values and differences are preserved in [paired_differences.csv](../data/stage4b/paired_differences.csv), including validation, training and QC sensitivity tracks. No favourable repeats were selected.

## QC-pass evaluation sensitivity

The same 40 fitted models and probabilities are evaluated on QC-pass rows within each unchanged partition. No model is refitted on a QC-pass-only training subset. Original denominators and QC-excluded counts remain in metrics.csv; the primary cohort retains QC failures.

| model | strategy | metric | count | mean | median | std | min | max | range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | host_group_stratified | average_precision | 10 | 0.59383794 | 0.59902963 | 0.022710469 | 0.56486658 | 0.6352073 | 0.070340724 |
| logistic_regression | host_group_stratified | brier | 10 | 0.20927416 | 0.20867808 | 0.0069265939 | 0.19885414 | 0.22254951 | 0.023695375 |
| logistic_regression | host_group_stratified | roc_auc | 10 | 0.74817832 | 0.75184261 | 0.018882038 | 0.72353616 | 0.78178801 | 0.058251847 |
| logistic_regression | object_stratified | average_precision | 10 | 0.59200825 | 0.58976871 | 0.033788723 | 0.54396309 | 0.64845576 | 0.10449267 |
| logistic_regression | object_stratified | brier | 10 | 0.21029963 | 0.21164043 | 0.01075926 | 0.19264873 | 0.22742024 | 0.034771515 |
| logistic_regression | object_stratified | roc_auc | 10 | 0.74645435 | 0.74520349 | 0.027254228 | 0.69722833 | 0.78981769 | 0.092589359 |
| random_forest | host_group_stratified | average_precision | 10 | 0.81320334 | 0.81320919 | 0.013276947 | 0.79510327 | 0.83156096 | 0.036457688 |
| random_forest | host_group_stratified | brier | 10 | 0.14831748 | 0.14791695 | 0.004067887 | 0.14248861 | 0.15547764 | 0.012989035 |
| random_forest | host_group_stratified | roc_auc | 10 | 0.87765927 | 0.87755433 | 0.0092118 | 0.86347802 | 0.8918151 | 0.028337082 |
| random_forest | object_stratified | average_precision | 10 | 0.81665558 | 0.82636052 | 0.020153399 | 0.77570256 | 0.83362534 | 0.057922785 |
| random_forest | object_stratified | brier | 10 | 0.14833988 | 0.14814498 | 0.0039907752 | 0.14150978 | 0.15479323 | 0.013283449 |
| random_forest | object_stratified | roc_auc | 10 | 0.87723568 | 0.88068738 | 0.011477324 | 0.85519251 | 0.89066447 | 0.035471958 |

QC-pass paired test differences:

| model | metric | count | mean | median | std | min | max | range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | average_precision | 10 | -0.0018296868 | 0.011299601 | 0.034873237 | -0.066728133 | 0.035062649 | 0.10179078 |
| logistic_regression | brier | 10 | 0.0010254738 | -0.00099794788 | 0.011161169 | -0.013341013 | 0.022032679 | 0.035373691 |
| logistic_regression | roc_auc | 10 | -0.0017239759 | 0.0080672197 | 0.029909934 | -0.066800985 | 0.025373506 | 0.092174491 |
| random_forest | average_precision | 10 | 0.003452235 | 0.0004665841 | 0.024509024 | -0.043259871 | 0.038522071 | 0.081781942 |
| random_forest | brier | 10 | 2.2400023e-05 | 0.0011122392 | 0.0062797846 | -0.0085507731 | 0.0071992642 | 0.015750037 |
| random_forest | roc_auc | 10 | -0.00042358613 | -0.0040042694 | 0.015917565 | -0.022687466 | 0.022600436 | 0.045287902 |

## Calibration

Raw model Brier scores are in the tables above. The [calibration-bin table](../data/stage4b/calibration_bins.csv) contains ten fixed-width bins per partition/track/run, including empty bins, counts, mean probability and observed positive fraction. The [test calibration figure](../data/stage4b/figures/test_calibration.png) shows one raw curve per repeat; no calibrator was fitted. Brier assesses overall probability quality and is not interpreted as pure calibration error.

## Host seen/unseen: secondary diagnostic only

Within object-level test partitions, a host is seen if it appears in that repeat's training partition. This grouping did not alter fitting, assignments, eligibility, thresholds or the primary paired analysis. Class composition and sample sizes differ, so this is not a causal leakage estimate.

| model | repeat_id | host_group | sample_count | positive_prevalence | roc_auc | average_precision | brier | precision | recall | f1 | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0 | seen_in_training | 175 | 0.86285714 | 0.64679912 | 0.89660415 | 0.25249032 | 0.89830508 | 0.70198675 | 0.78810409 | 12 | 12 | 45 | 106 |
| logistic_regression | 0 | unseen_in_training | 818 | 0.30317848 | 0.72443407 | 0.44162465 | 0.21287 | 0.47633136 | 0.64919355 | 0.54948805 | 393 | 177 | 87 | 161 |
| logistic_regression | 1 | seen_in_training | 178 | 0.90449438 | 0.67848009 | 0.93133968 | 0.23944754 | 0.94392523 | 0.62732919 | 0.75373134 | 11 | 6 | 60 | 101 |
| logistic_regression | 1 | unseen_in_training | 815 | 0.29202454 | 0.76086102 | 0.48296488 | 0.19466812 | 0.51724138 | 0.56722689 | 0.54108216 | 451 | 126 | 103 | 135 |
| logistic_regression | 2 | seen_in_training | 169 | 0.8816568 | 0.72885906 | 0.912087 | 0.20327973 | 0.94957983 | 0.75838926 | 0.84328358 | 14 | 6 | 36 | 113 |
| logistic_regression | 2 | unseen_in_training | 824 | 0.30339806 | 0.78320209 | 0.52537262 | 0.18893865 | 0.55921053 | 0.68 | 0.61371841 | 440 | 134 | 80 | 170 |
| logistic_regression | 3 | seen_in_training | 154 | 0.9025974 | 0.57026379 | 0.90267195 | 0.25834343 | 0.92708333 | 0.64028777 | 0.75744681 | 8 | 7 | 50 | 89 |
| logistic_regression | 3 | unseen_in_training | 839 | 0.30989273 | 0.78686728 | 0.52945458 | 0.19073965 | 0.58957655 | 0.69615385 | 0.63844797 | 453 | 126 | 79 | 181 |
| logistic_regression | 4 | seen_in_training | 177 | 0.85875706 | 0.59263158 | 0.87916484 | 0.26485143 | 0.89473684 | 0.67105263 | 0.76691729 | 13 | 12 | 50 | 102 |
| logistic_regression | 4 | unseen_in_training | 816 | 0.30269608 | 0.79195691 | 0.5313784 | 0.18553843 | 0.54103343 | 0.72064777 | 0.61805556 | 418 | 151 | 69 | 178 |
| logistic_regression | 5 | seen_in_training | 162 | 0.9382716 | 0.63421053 | 0.95991599 | 0.27221187 | 0.96078431 | 0.64473684 | 0.77165354 | 6 | 4 | 54 | 98 |
| logistic_regression | 5 | unseen_in_training | 831 | 0.29723225 | 0.75786146 | 0.46860279 | 0.20209659 | 0.49579832 | 0.71659919 | 0.58609272 | 404 | 180 | 70 | 177 |
| logistic_regression | 6 | seen_in_training | 183 | 0.8579235 | 0.67736404 | 0.91670238 | 0.26460907 | 0.92222222 | 0.52866242 | 0.67206478 | 19 | 7 | 74 | 83 |
| logistic_regression | 6 | unseen_in_training | 810 | 0.29876543 | 0.70111163 | 0.42866191 | 0.21570356 | 0.4261745 | 0.52479339 | 0.47037037 | 397 | 171 | 115 | 127 |
| logistic_regression | 7 | seen_in_training | 168 | 0.9047619 | 0.66940789 | 0.93513569 | 0.24560497 | 0.93457944 | 0.65789474 | 0.77220077 | 9 | 7 | 52 | 100 |
| logistic_regression | 7 | unseen_in_training | 825 | 0.29939394 | 0.7468655 | 0.45735758 | 0.20393454 | 0.4815864 | 0.68825911 | 0.56666667 | 395 | 183 | 77 | 170 |
| logistic_regression | 8 | seen_in_training | 165 | 0.91515152 | 0.76773888 | 0.94795044 | 0.23456167 | 0.97169811 | 0.68211921 | 0.80155642 | 11 | 3 | 48 | 103 |
| logistic_regression | 8 | unseen_in_training | 828 | 0.29951691 | 0.75141129 | 0.49106314 | 0.18885863 | 0.48013245 | 0.58467742 | 0.52727273 | 423 | 157 | 103 | 145 |
| logistic_regression | 9 | seen_in_training | 172 | 0.90697674 | 0.65865385 | 0.94356207 | 0.27697486 | 0.94565217 | 0.55769231 | 0.7016129 | 11 | 5 | 69 | 87 |
| logistic_regression | 9 | unseen_in_training | 821 | 0.29598051 | 0.75506572 | 0.4676361 | 0.2016008 | 0.51104101 | 0.66666667 | 0.57857143 | 423 | 155 | 81 | 162 |
| random_forest | 0 | seen_in_training | 175 | 0.86285714 | 0.8388521 | 0.96815537 | 0.17231292 | 0.944 | 0.78145695 | 0.85507246 | 17 | 7 | 33 | 118 |
| random_forest | 0 | unseen_in_training | 818 | 0.30317848 | 0.87420062 | 0.74597914 | 0.14062398 | 0.64 | 0.70967742 | 0.67304015 | 471 | 99 | 72 | 176 |
| random_forest | 1 | seen_in_training | 178 | 0.90449438 | 0.85787358 | 0.9807781 | 0.17343069 | 0.9765625 | 0.77639752 | 0.8650519 | 14 | 3 | 36 | 125 |
| random_forest | 1 | unseen_in_training | 815 | 0.29202454 | 0.84968615 | 0.66836584 | 0.14685509 | 0.57857143 | 0.68067227 | 0.62548263 | 459 | 118 | 76 | 162 |
| random_forest | 2 | seen_in_training | 169 | 0.8816568 | 0.86879195 | 0.97528808 | 0.16308311 | 0.97619048 | 0.82550336 | 0.89454545 | 17 | 3 | 26 | 123 |
| random_forest | 2 | unseen_in_training | 824 | 0.30339806 | 0.89478746 | 0.76610771 | 0.13271608 | 0.66896552 | 0.776 | 0.71851852 | 478 | 96 | 56 | 194 |
| random_forest | 3 | seen_in_training | 154 | 0.9025974 | 0.86330935 | 0.98316681 | 0.16729607 | 0.95327103 | 0.73381295 | 0.82926829 | 10 | 5 | 37 | 102 |
| random_forest | 3 | unseen_in_training | 839 | 0.30989273 | 0.88450246 | 0.73663018 | 0.14076756 | 0.65495208 | 0.78846154 | 0.71553229 | 471 | 108 | 55 | 205 |
| random_forest | 4 | seen_in_training | 177 | 0.85875706 | 0.79578947 | 0.9571254 | 0.18166434 | 0.92366412 | 0.79605263 | 0.85512367 | 15 | 10 | 31 | 121 |
| random_forest | 4 | unseen_in_training | 816 | 0.30269608 | 0.8820361 | 0.75455922 | 0.13596526 | 0.68359375 | 0.70850202 | 0.69582505 | 488 | 81 | 72 | 175 |
| random_forest | 5 | seen_in_training | 162 | 0.9382716 | 0.85131579 | 0.98646074 | 0.17322317 | 0.975 | 0.76973684 | 0.86029412 | 7 | 3 | 35 | 117 |
| random_forest | 5 | unseen_in_training | 831 | 0.29723225 | 0.87766901 | 0.75789197 | 0.13986962 | 0.65277778 | 0.7611336 | 0.70280374 | 484 | 100 | 59 | 188 |
| random_forest | 6 | seen_in_training | 183 | 0.8579235 | 0.73346399 | 0.94248572 | 0.17491089 | 0.88356164 | 0.82165605 | 0.85148515 | 9 | 17 | 28 | 129 |
| random_forest | 6 | unseen_in_training | 810 | 0.29876543 | 0.86050809 | 0.69963279 | 0.14702166 | 0.61538462 | 0.72727273 | 0.66666667 | 458 | 110 | 66 | 176 |
| random_forest | 7 | seen_in_training | 168 | 0.9047619 | 0.8203125 | 0.97563282 | 0.16442472 | 0.94890511 | 0.85526316 | 0.89965398 | 9 | 7 | 22 | 130 |
| random_forest | 7 | unseen_in_training | 825 | 0.29939394 | 0.88458036 | 0.75286518 | 0.13601463 | 0.65972222 | 0.76923077 | 0.71028037 | 480 | 98 | 57 | 190 |
| random_forest | 8 | seen_in_training | 165 | 0.91515152 | 0.89687796 | 0.9888041 | 0.17304022 | 0.9826087 | 0.74834437 | 0.84962406 | 12 | 2 | 38 | 113 |
| random_forest | 8 | unseen_in_training | 828 | 0.29951691 | 0.85853031 | 0.70887731 | 0.14400315 | 0.59515571 | 0.69354839 | 0.6405959 | 463 | 117 | 76 | 172 |
| random_forest | 9 | seen_in_training | 172 | 0.90697674 | 0.79086538 | 0.9736693 | 0.17048637 | 0.94615385 | 0.78846154 | 0.86013986 | 9 | 7 | 33 | 123 |
| random_forest | 9 | unseen_in_training | 821 | 0.29598051 | 0.8835633 | 0.76327911 | 0.14259798 | 0.61093248 | 0.781893 | 0.68592058 | 457 | 121 | 53 | 190 |

[Host overlap counts](../data/stage4b/host_overlap.csv) accompany the diagnostic. Figures are compact aggregate plots; no per-target plots were generated.

## Determinism and hashes

Repeated predictions, model reloads and CSV roundtrips were bit-identical for every run. A fresh process replayed all **40 saved models / 265,480 probabilities**, with zero new fits. This verifies prediction replay in the pinned environment, not a second independent training run or cross-platform binary identity.

Every prediction artifact SHA-256 (runtime-root-relative paths):

```text
414be16898408207772968d5faf614fcfde1e83f3d2f7a926a6b7859c68bb9ab  predictions/logistic_regression/object_stratified/repeat-00.csv
fc5bcdde43101a2e943297c885a0a0ba58f21dd8fd5f70597b2f4a6a58e6c7c7  predictions/logistic_regression/host_group_stratified/repeat-00.csv
835566be3a22e948570da3e874ac5f965adf6747f63eee0ba20a4229ebe89e12  predictions/logistic_regression/object_stratified/repeat-01.csv
1a9dc8a2ab52e3d65a913e83ed43f148e6bce0ccb25388f0bef8617fdb8a3092  predictions/logistic_regression/host_group_stratified/repeat-01.csv
f768e282135abf21382f71fccb5bbcaff17c2c4ff7b347f7e78a50020b238b86  predictions/logistic_regression/object_stratified/repeat-02.csv
f4123042ae4f2b13e4510f402a90d9ee306d78538442f7f146c010563ff4d28e  predictions/logistic_regression/host_group_stratified/repeat-02.csv
81c3fa4a2fc4460b4d1c4bdbe4408cae0a149e49dc898e2ff2d3c339264653df  predictions/logistic_regression/object_stratified/repeat-03.csv
4b461b6fc96bc24a06f9d5dfa4e71e96f57b385cc0c59a06ed32f95b7ddf2b1e  predictions/logistic_regression/host_group_stratified/repeat-03.csv
a1366336e04ece040a89bcea2ee18132d77bf20d714c99c616738582c596fcac  predictions/logistic_regression/object_stratified/repeat-04.csv
7926326e558b7815e3bd42fadc668605a3fad9cf0c0c95b5a09abedd34a4d0f3  predictions/logistic_regression/host_group_stratified/repeat-04.csv
dcbacd9aaf87f679bd34095c12957e3298e19577ed4939e3a38d173a0e93b836  predictions/logistic_regression/object_stratified/repeat-05.csv
61cb9e74fff0eb0d18d8707102029325134a6750a6acbce0c1ac0ecf0564df66  predictions/logistic_regression/host_group_stratified/repeat-05.csv
76cb27ec35caba68f2ac430b9ffee72ee440dae3a27c258ebd29f52f4a0dd033  predictions/logistic_regression/object_stratified/repeat-06.csv
029d01f8b015e49c1e54468096c1d878466d8eee72402cb649a7cfa44c073a24  predictions/logistic_regression/host_group_stratified/repeat-06.csv
3af1aa89222cc4fb5f1693630ae86c469882e12972525b07a207a79b59da2ebc  predictions/logistic_regression/object_stratified/repeat-07.csv
29bb0a4c4d91b506b2c810760a5b93edfefaf02fb61ad40586e6f21c306c01c9  predictions/logistic_regression/host_group_stratified/repeat-07.csv
bbe0b7c4f80e2630f284145e946b40a1d327d4b51522a4ec9a699a6dab2a39df  predictions/logistic_regression/object_stratified/repeat-08.csv
e610f3b9aa159ad30fd1cce5e62fb21d40a84f895f20b784428bc8c07387a1e8  predictions/logistic_regression/host_group_stratified/repeat-08.csv
62ad56f3a3ef2175387da4c2dd4390b7ac6152a2b054b70b16d49e694564eda1  predictions/logistic_regression/object_stratified/repeat-09.csv
cd96ab902bd08747c5b1aeb13361d7c324288d6a5324c13c3347e59d901ba56e  predictions/logistic_regression/host_group_stratified/repeat-09.csv
8ed59bfbb414e8df4c50e35bf88673bb9206a377817904f0cc649fe68f2990f3  predictions/random_forest/object_stratified/repeat-00.csv
155459c8c8f0a3048451eafe7822bd77b8a616e162d13f946aa95b8ee4256927  predictions/random_forest/host_group_stratified/repeat-00.csv
9fe05dc4dd30226567eb67ef41f9361f50ca25abc9ab136626feb534c0ab34c0  predictions/random_forest/object_stratified/repeat-01.csv
4fafe27c3f6dd058216f86be3803d62fe72e4cfd48c9855784c234be36460616  predictions/random_forest/host_group_stratified/repeat-01.csv
cd8b6505112c6144f42277bf527cd3dbff3d8150cfff4de959446a793545c4e6  predictions/random_forest/object_stratified/repeat-02.csv
a1f0308ab869f19569e497ead795c285aa839efb7564a625a1b129998a4363de  predictions/random_forest/host_group_stratified/repeat-02.csv
8d425641224f3fcdff993a1c87781acb281786ee1b2e86ee4a0bba690e0c5a12  predictions/random_forest/object_stratified/repeat-03.csv
236347a23ca9898cbff6c6362a476ff4264a893e2e280fb52fc6dda89af7d00c  predictions/random_forest/host_group_stratified/repeat-03.csv
be4f6d374a5d4be5bf2d280e7ff580192022dccc0df1bd84f61092efee949e20  predictions/random_forest/object_stratified/repeat-04.csv
da55725021e3a194c3b7432eb7ba6e2e599d10f29989418511965528fecf835e  predictions/random_forest/host_group_stratified/repeat-04.csv
8bb7cce8d5d32136ed9bab8b81e8b81137dd481f4de063f24ec31fc6c1f39ab5  predictions/random_forest/object_stratified/repeat-05.csv
18fdeac646309022df06d436a123b890e3773d8deceed0452f980654bce1445b  predictions/random_forest/host_group_stratified/repeat-05.csv
324d5f0284fe5f5f397bc2eb2dccb526ce0016b629b8aed76c18564fb17e566f  predictions/random_forest/object_stratified/repeat-06.csv
108152d77d0b6d14232139803e1b9fd966d921193186c567bab65924a428c161  predictions/random_forest/host_group_stratified/repeat-06.csv
1608f4861b15af8585d89923befcf7e535b5cda5def28a2a85911773c37edee5  predictions/random_forest/object_stratified/repeat-07.csv
26235bfce3847f86cb1f293c254e6537b46ff563e648ec26696f4174273c56e0  predictions/random_forest/host_group_stratified/repeat-07.csv
81aa77a57f4f6315c71b25cfeafba30b3ff9ebc4c9def70b86b53ae89815fcd5  predictions/random_forest/object_stratified/repeat-08.csv
be55d06e2629ea49162f0f0e1b86b5bfa003940b5035e76acf829e634db191c5  predictions/random_forest/host_group_stratified/repeat-08.csv
2df4823689aa91caa851dd2d70bb2d47ecf1036a21a963029e852cf11c921c31  predictions/random_forest/object_stratified/repeat-09.csv
c64efc21b9882ff82f71b13a190aa0c120cdafe219de1cb374f41b4305726a42  predictions/random_forest/host_group_stratified/repeat-09.csv
```

Metric, provenance and verification hashes:

| Artifact | SHA-256 |
|---|---|
| `calibration_bins.csv` | `ec4b6e97e797e7718c054daf7fe49242dc2f5669d714ab0d4f25910f0038d9f7` |
| `checksums.sha256` | `6fbcdbb08800c9f43eab6c3bf24e088592fa4bc16c893a26824b12dd938c75ff` |
| `host_diagnostic.csv` | `5f1ca10bbd78aaf54618432c2a38fcb4e9f4526a47a209f41f8fe0b499b74eae` |
| `host_overlap.csv` | `a501004c15f858b366bd5029394b1c534a8d8ba7188ef57e2fa6a9414d60b176` |
| `input_validation.json` | `20eab7749ee707e4fe885593e946e42036886ae18c4dd5f70cb2b747645ef242` |
| `metrics.csv` | `20e51d646347b18769f31aef78487bfa57f2b300fc6531f293749b9039621511` |
| `model_checksums.sha256` | `ef1e4773ab803e61c07577c9df7122bd9e93f7f06964406d932cc8f3e6576862` |
| `objects.csv` | `b0978f99cf046fe53a5b189d0cbdf428c9a66da3a1083a2bb716a138e1ed8d90` |
| `paired_differences.csv` | `767cd89197ca66d0b868f080c71f92e9c1a4e2cc38219dc9df8cd69de61f4471` |
| `paired_summary.csv` | `eb362a9061a64cb2c20e0cd2d0bcf52f7ebf674d44b43d312e5accddb55f847c` |
| `prediction_checksums.sha256` | `b501ddfca7d666d15664ee96b2e4dbb62a1f896ce34749c8d2dc3d28342d424f` |
| `prediction_verification.json` | `d8be119bb9c6dcfa2550a4ca960805d6b0d517266d50ef8ecdf7ee41e43ddad7` |
| `provenance.json` | `73bed931860b4a610298eb2e802174a88942453405deb24ff4f9de600987aa45` |
| `repeat_summary.csv` | `e95500682f913a1fd509d7149a0502be679809b490c18fecddff4b103f7dfe44` |
| `run_manifest.json` | `404115c3320a5a1df904c67533fd6a5125c902173949ffb5cead3a1ac66a56ed` |
| `runtime.json` | `59f578b68118a6a1f255bd842d42a73f9aef8ff9a538ffdf58d7728af485a4de` |

[Model checksums](../data/stage4b/model_checksums.sha256) cover every saved estimator. The feature-cache SHA-256 and all frozen-input hashes are recorded in provenance. Scientific paths are relative; timestamps/runtime and serialized estimator files are operational artifacts rather than a cross-platform byte-determinism claim.

## Tests and scope

Complete offline suite: **134 passed**, 0 errors, 0 failures, 0 skipped; three network tests deselected. The full suite was run once. JUnit evidence: `data/tmp/stage4b-full-20260919.xml`.

`git diff --check`: passed. Stage 1, Stage 2 scientific preprocessing, Stage 3 assignments, Stage 4A outputs/configuration and the frozen MAST inventory remain unchanged. No new splits, CNN, Stage 5, TESS, app integration or automatic commit occurred. Stop here for review.
