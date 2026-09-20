# Stage 5: fixed dual-view CNN

Stage 5 extends the certified Stage 4B experiment with one approved CNN, two
frozen strategies and ten repeats. Scientific inputs and baseline outputs are
read-only. The configuration anchors their expected SHA-256 values. All 6,637
KOIs (5,804 hosts), including 283 QC failures, remain in the primary experiment.

Only canonical global_flux (2000,1) and local_flux (200,1), float32, enter the
network. No additional normalization, imputation, augmentation or metadata is
allowed. The existing Stage 4B flux cache is reused only after hash, identity and
exact value comparison with independently validated canonical NPZ arrays.
Stage 5 does not create another array copy on disk.

Each branch has Conv1D(16,5,same,ReLU), max pooling, Conv1D(32,5,same,ReLU), max
pooling and flattening. Global pool/stride5 yields2560; local pool/stride2
yields1600. Fusion4160 → Dense32 ReLU → Dropout0.2 → Dense1 sigmoid.
Exactly138,561 parameters are required before fitting. Hidden kernels use
HeNormal, output GlorotUniform, biases zero. Initializer offsets0..4 and6 and
dropout offset5 are added to the Stage 3 seed; both strategies start from the
same initial weights, verified by hashes. No architecture search occurs.

Adam uses learning_rate0.001, beta1=0.9, beta2=0.999, epsilon1e-7,
global_clipnorm1, no weight decay. BCE, batch64, at most100 epochs, no sample or
class weights. Training order uses NumPy default_rng(Stage3 seed) permutations.
Only training arrays enter gradient updates. The fitting function accepts
training/validation arrays only and never receives a test array or full cohort.
The final short batch is retained.

EarlyStopping uses val_loss, mode=min, patience10, min_delta0, and restores the
earliest best weights, including when the epoch cap is reached. Epoch losses
are end-of-epoch inference BCE (dropout off); exact sklearn ROC-AUC is recorded
for train and validation. Validation selects an epoch but receives no gradient
updates. Test predictions are generated after selection. No train+validation
refit, threshold tuning or post-hoc calibration occurs. Epoch cap is reported
as epoch_cap_reached, not convergence. Saved optimizer slots correspond to the
stopping epoch; restored model weights correspond to the best epoch, so saved
models are inference artifacts, not within-fit resume checkpoints.

Every run is identified by model version, strategy and repeat ID. Completion is
atomic after model, predictions and history are written and serialized replay
passes. A completed run is skipped only after manifest context and all artifact
hashes pass. Partial attempts are preserved with a partial timestamp and restart
from their original seed; no within-fit optimizer continuation is attempted.
Corrupt completed runs stop for review rather than silently retraining.

The isolated environment lock is configs/kepler_stage5_environment_lock.txt.
The Stage 4B Python environment is unchanged. CPU runs are sequential, with
numerical and TensorFlow intra/inter-op threads set to1, deterministic operations
enabled, and oneDNN disabled. Same-environment synthetic training must reproduce
exactly. Scientific repeated/reloaded probabilities use atol1e-7, rtol0 and exact
predicted labels. Fresh-process replay verifies all20 saved models and
recomputes all120 metric rows. Neither prediction replay nor synthetic tests
constitute a second independent scientific training experiment.

From the repository root in PowerShell:

```powershell
.venv/stage5/Scripts/python.exe -m scripts.stage5_run_cnn --dataset 'G:\My Drive\Exominer Research Data\kepler_dr25'
.venv/stage5/Scripts/python.exe -m scripts.stage5_run_cnn --replay
.venv/stage5/Scripts/python.exe -m pytest -m 'not network' --basetemp=data/tmp/stage5-full --junitxml=data/tmp/stage5-full.xml
.venv/stage5/Scripts/python.exe -m scripts.stage5_report
```

The infrastructure proof is recorded only after the focused synthetic suite
passes. Frozen scientific hashes, the proof's model-source hash and environment
versions are checked before fitting. Non-finite inputs/loss/gradients/weights,
identity errors, grouped overlap and replay failures stop execution without
hyperparameter changes. Missing inputs are never repaired automatically.

Stage 4B metric functions are reused unchanged, including threshold>=0.5,
undefined flags, calibration bins and QC-pass evaluation of the same predictions.
Object-minus-grouped comparisons and CNN-minus-baseline comparisons are paired
by repeat. Random Forest is principal; logistic comparison is also reported.
Signed and absolute split-sensitivity contrasts are descriptive; repeated splits
are not independent. Host seen/unseen is secondary only.

No morphology attribution is included. Predictive improvement alone cannot
prove useful transit-morphology learning. Stage 5A/5.1 requires separate review.
No TESS or Streamlit integration and no automatic commit are authorized here.
