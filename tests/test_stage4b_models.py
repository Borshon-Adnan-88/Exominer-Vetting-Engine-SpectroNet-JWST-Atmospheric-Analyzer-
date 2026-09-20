import copy
import warnings

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from kepler_baselines import models
from kepler_baselines.inputs import StopForReview, load_config


def toy():
    rng = np.random.default_rng(1729)
    features = rng.normal(size=(24, 2200))
    labels = np.tile([0, 1], 12)
    partitions = np.array(["train"] * 16 + ["validation"] * 4 + ["test"] * 4)
    features[16:20] += 100
    features[20:] -= 100
    return features, labels, partitions


def test_single_pipeline_fits_scaler_and_classifier_on_training_only(monkeypatch):
    features, labels, partitions = toy()
    scaler_calls, classifier_calls = [], []
    scaler_fit = models.StandardScaler.fit
    classifier_fit = models.LogisticRegression.fit
    def record_scaler(self, X, y=None, **kwargs):
        scaler_calls.append(np.array(X, copy=True))
        return scaler_fit(self, X, y, **kwargs)
    def record_classifier(self, X, y, **kwargs):
        classifier_calls.append((np.array(X, copy=True), np.array(y, copy=True)))
        return classifier_fit(self, X, y, **kwargs)
    monkeypatch.setattr(models.StandardScaler, "fit", record_scaler)
    monkeypatch.setattr(models.LogisticRegression, "fit", record_classifier)
    model, _ = models.fit_training_only("logistic_regression", features, labels, partitions, load_config(), 1729)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["scaler", "classifier"]
    assert len(scaler_calls) == len(classifier_calls) == 1
    np.testing.assert_array_equal(scaler_calls[0], features[:16])
    np.testing.assert_array_equal(classifier_calls[0][1], labels[:16])
    assert classifier_calls[0][0].shape == (16, 2200)
    np.testing.assert_allclose(classifier_calls[0][0].mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(model["scaler"].mean_, features[:16].mean(axis=0))
    assert model["scaler"].n_samples_seen_ == 16
    original_mean = model["scaler"].mean_.copy()
    models.probabilities(model, features[partitions == "validation"])
    models.probabilities(model, features[partitions == "test"])
    # Holdouts can change arbitrarily without a fit or changed scaler state.
    models.probabilities(model, features[16:] * 1e4)
    assert len(scaler_calls) == len(classifier_calls) == 1
    np.testing.assert_array_equal(model["scaler"].mean_, original_mean)


def test_convergence_warning_stops_instead_of_accepting_model(monkeypatch):
    features, labels, partitions = toy()
    def fail(*args, **kwargs):
        warnings.warn("iteration limit", ConvergenceWarning)
    monkeypatch.setattr(models.LogisticRegression, "fit", fail)
    with pytest.raises(StopForReview, match="failed to converge"):
        models.fit_training_only("logistic_regression", features, labels, partitions, load_config(), 1729)


def test_nonfinite_input_stops_before_fitting(monkeypatch):
    features, labels, partitions = toy(); features[-1, 0] = np.nan
    monkeypatch.setattr(models, "make_model", lambda *a: pytest.fail("model constructed with invalid inputs"))
    with pytest.raises(StopForReview, match="non-finite"):
        models.fit_training_only("logistic_regression", features, labels, partitions, load_config(), 1729)


def test_prediction_verification_rejects_even_one_bit_change():
    expected = np.array([.1, .5, .9])
    actual = expected.copy(); actual[0] = np.nextafter(actual[0], 1)
    with pytest.raises(StopForReview, match="deterministic prediction verification failed"):
        models.verify_identical_predictions(expected, actual, "test")


@pytest.mark.parametrize("family", ["logistic_regression", "random_forest"])
def test_stored_model_prediction_is_bit_identical(tmp_path, family):
    features, labels, partitions = toy()
    config = copy.deepcopy(load_config())
    config["models"]["random_forest"]["n_estimators"] = 5  # synthetic test only
    model, _ = models.fit_training_only(family, features, labels, partitions, config, 1729)
    expected = models.probabilities(model, features)
    path = tmp_path / "model.joblib"; joblib.dump(model, path, compress=3)
    np.testing.assert_array_equal(expected, models.probabilities(joblib.load(path), features))
