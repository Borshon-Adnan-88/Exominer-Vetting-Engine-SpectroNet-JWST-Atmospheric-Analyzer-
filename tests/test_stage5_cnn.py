"""Small infrastructure tests; no scientific arrays or scientific fits."""
import json
from pathlib import Path

import numpy as np
import pytest

# Apply production thread/device policy before TensorFlow import.
import scripts.stage5_run_cnn
tf = pytest.importorskip("tensorflow")
from kepler_baselines.inputs import StopForReview, sha256
from kepler_baselines.runner import write_json
from kepler_cnn.artifacts import completed_run, views
from kepler_cnn.model import build, configure, fit, predict, validate_views, verify_probability, weight_hash


@pytest.fixture(scope="module", autouse=True)
def cpu_policy():
    configure()


def synthetic():
    x = np.random.default_rng(901).normal(0, .1, (12, 2200)).astype(np.float32)
    return views(x), np.tile([0, 1], 6)


def test_architecture_and_initialization():
    first = build(1729); digest = weight_hash(first)
    assert first.count_params() == 138561
    assert first.input_shape == [(None, 2000, 1), (None, 200, 1)]
    assert first.get_layer("global_flatten").output.shape[-1] == 2560
    assert first.get_layer("local_flatten").output.shape[-1] == 1600
    for layer in first.layers:
        if hasattr(layer, "bias") and layer.bias is not None:
            assert np.count_nonzero(layer.bias.numpy()) == 0
    second = build(1729)
    assert weight_hash(second) == digest
    assert weight_hash(build(2718)) != digest


def test_flux_allowlist_and_exact_cast():
    x, _ = synthetic()
    with pytest.raises(StopForReview, match="forbidden"):
        validate_views({**x, "kepid": np.arange(12)})
    x["local_flux"] = x["local_flux"][:-1]
    with pytest.raises(StopForReview, match="shape"):
        validate_views(x)
    with pytest.raises(StopForReview, match="cast"):
        views(np.full((2, 2200), .1, dtype=np.float64))
    with pytest.raises(StopForReview, match="invalid"):
        views(np.full((2, 2200), np.nan, dtype=np.float32))


def test_completed_runs_require_hashes(tmp_path):
    identity = {"model_version": "test", "strategy": "object", "repeat_id": 0}
    assert completed_run(tmp_path, identity, {}) is None
    for name in ("model.keras", "predictions.csv", "history.csv"):
        (tmp_path / name).write_text("fixture")
    record = {"identity": identity, "context": {}, "sha256": {p.name: sha256(p) for p in tmp_path.iterdir()}}
    write_json(tmp_path / "manifest.json", record)
    assert completed_run(tmp_path, identity, {}) == record
    (tmp_path / "model.keras").write_text("changed")
    with pytest.raises(StopForReview, match="artifact mismatch"):
        completed_run(tmp_path, identity, {})


def test_tolerance_never_allows_label_change():
    verify_probability(np.array([.2]), np.array([.2 + 5e-8]))
    with pytest.raises(StopForReview, match="labels"):
        verify_probability(np.array([.5 - 1e-8]), np.array([.5 + 1e-8]))
    with pytest.raises(StopForReview, match="replay failed"):
        verify_probability(np.array([.2]), np.array([.21]))


def test_training_boundary_and_synthetic_determinism(tmp_path, monkeypatch):
    import inspect
    assert list(inspect.signature(fit).parameters)[:6] == ["model", "training", "training_labels", "validation", "validation_labels", "seed"]
    x, y = synthetic()
    training = {k: v[:8] for k, v in x.items()}; validation = {k: v[8:] for k, v in x.items()}
    outputs = []; weight_digests = []
    # Two two-epoch synthetic trainings are one infrastructure test, never scientific fits.
    for iteration in range(2):
        model = build(1729)
        initial = weight_hash(model)
        seen = []
        original_call = model.call
        def audited_call(inputs, training=False, **kwargs):
            if training:
                # tf.function traces only the training update; tensor values are checked in graph.
                g = inputs["global_flux"]
                matches = tf.reduce_all(tf.equal(g[:, None], tf.constant(x["global_flux"][:8])[None]), axis=[2, 3])
                tf.debugging.assert_equal(tf.reduce_all(tf.reduce_any(matches, axis=1)), True)
                seen.append("training_only")
            return original_call(inputs, training=training, **kwargs)
        model.call = audited_call
        history, outcome = fit(model, training, y[:8], validation, y[8:], 1729, epochs=2, patience=1)
        assert seen and weight_hash(model) != initial
        assert outcome["best_epoch"] == min(history, key=lambda r:r["validation_loss"])["epoch"]
        before = weight_hash(model); probability = predict(model, x)
        assert weight_hash(model) == before
        model.call = original_call
        path = tmp_path / f"synthetic-{iteration}.keras"; model.save(path)
        verify_probability(probability, predict(tf.keras.models.load_model(path, compile=False), x))
        outputs.append(probability); weight_digests.append(weight_hash(model))
    assert np.array_equal(*outputs) and weight_digests[0] == weight_digests[1]


def test_early_stopping_validation_only_restores_earliest_best():
    model = build(1729)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0, restore_best_weights=True)
    callback.set_model(model); callback.on_train_begin()
    hashes = []
    for epoch, value in enumerate([.8, .4, .4, .6]):
        model.trainable_variables[-1].assign([float(epoch)])
        hashes.append(weight_hash(model))
        callback.on_epoch_end(epoch, {"val_loss": value, "loss": 100 - epoch})
    assert model.stop_training
    callback.on_train_end()
    assert weight_hash(model) == hashes[1]
