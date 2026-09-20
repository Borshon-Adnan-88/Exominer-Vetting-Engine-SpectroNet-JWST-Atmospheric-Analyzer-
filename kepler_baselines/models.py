"""Fixed models and a single training-only fitting boundary."""
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .inputs import require, StopForReview


def make_model(family, config, seed):
    if family == "logistic_regression":
        return Pipeline([("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(**config["models"][family]))])
    require(family == "random_forest", "unapproved model family")
    return RandomForestClassifier(**config["models"][family], random_state=int(seed))


def fit_training_only(family, features, labels, partitions, config, seed):
    require(isinstance(features, np.ndarray) and features.ndim == 2 and features.shape[1] == 2200, "invalid model feature representation")
    require(np.isfinite(features).all(), "non-finite model inputs")
    mask = np.asarray(partitions) == "train"
    require(mask.any() and set(labels[mask]) == {0, 1}, "invalid training partition")
    model = make_model(family, config, seed)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            # The ONLY fit call. Neither validation nor test participates.
            model.fit(features[mask], labels[mask])
        except ConvergenceWarning as exc:
            raise StopForReview("logistic regression failed to converge") from exc
    if family == "logistic_regression":
        require(int(model["scaler"].n_samples_seen_) == int(mask.sum()), "scaler fitted outside training partition")
        iterations = int(model["classifier"].n_iter_.max())
        require(iterations < config["models"][family]["max_iter"], "logistic regression reached iteration limit")
    else:
        iterations = None
    return model, iterations


def probabilities(model, features):
    with threadpool_limits(limits=1):
        classes = list(model.classes_)
        require(classes == [0, 1], "model class ordering changed")
        result = model.predict_proba(features)[:, classes.index(1)]
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "invalid model probabilities")
    return result


def verify_identical_predictions(expected, actual, context):
    require(np.array_equal(expected, actual), f"deterministic prediction verification failed: {context}")
