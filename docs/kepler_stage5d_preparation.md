# Stage 5D: 128-unit fusion implementation

Implementation and non-training verification only. No Stage 5D scientific fits
have been authorized for this preparation step.

The sole architectural intervention is `Dense(32, linear)` to
`Dense(128, linear)`, retaining LeakyReLU(0.01) and the 0.2 dropout. Total
parameters are 538,113. Branch architecture, initialization rules, Adam/BCE,
learning rate, clipping, batch size, seeds, splits and QC remain unchanged.
The branches train end-to-end, rather than freezing their learned weights.

The shared `kepler_cnn.model.fit` selects the earliest minimum-validation-BCE
checkpoint, with patience 10 and a 100-epoch cap. Validation ROC-AUC at that
checkpoint is compared with the matching Stage 5B history. Maximum-AUC epoch
selection is not used. All 20 strategy/repeat/seed pairs remain in the summary.

## Commands from the repository root

Read-only input checks and initialization checks, writing only preflight evidence:

```powershell
& '.\.venv\stage5\Scripts\python.exe' -u -m scripts.stage5d_run_fusion_width --preflight
```

Full execution, only when separately authorized (20 CNN fits and 80 learned
linear-probe fits; 40 compatible raw-input probes reused):

```powershell
& '.\.venv\stage5\Scripts\python.exe' -u -m scripts.stage5d_run_fusion_width --execute
```

The same explicit execution command resumes completed phases after validating
their checksums, identities, input scope and model binding. A partial phase
without a completion receipt stops for review; it is never silently discarded
or retrained. There is no within-fit continuation from optimizer state.

Progress: `Get-Content data/stage5d/status.json`. The inherited trainer prints
epoch progress to the launching terminal. Completion receipts, models,
histories and predictions live under `data/stage5d_runtime/width128_v1/`.
Final paired results, probe results, checksums and provenance live under
`data/stage5d/`. No previous experiment directory is written.

## Evaluation and probe boundaries

Only train/validation rows are materialized from the certified feature cache.
No held-out test predictions or evaluation phase is available. Split metadata
is checked against the original assignments; canonical/cache equivalence is
reused with the pinned cache hash, without downloading or regenerating NPZs.

The Stage 5C training-only StandardScaler and fixed logistic-regression fitting
function is reused unchanged. Four learned representations are recomputed:
global branch, local branch, concatenated branches and the 128-unit hidden
representation. Raw-global and raw-local probabilities are reused only after
checking protocol/source/configuration anchors, seed, strategy, repeat,
training population and validation KOI/label order. Reuse failures stop rather
than triggering new fits. Hidden-probe changes remain secondary to CNN AUC.

Same seeds do not imply equal differently shaped heads. Branch initialization
is compared byte-for-byte to the 32-unit builder; the complete 128-unit model
must initialize identically on repeated construction with the same seed.

The runner requires the pinned Python 3.13.15 / TensorFlow 2.21.0 / Keras 3.15.1
environment and deterministic single-thread CPU policy. It rejects changed
frozen sources or inputs rather than re-sealing them. Repeated validation use
is development evidence, not independent confirmation. Runtime is unmeasured
for the wider model; the prior 32-unit experiment took about 4.7 completed-fit
hours. No tuning or additional architectures are part of this implementation.
