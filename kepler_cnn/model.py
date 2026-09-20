"""Fixed architecture and training boundary: this module receives no test rows."""
import hashlib
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from kepler_baselines.inputs import require


def configure():
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def weight_hash(model):
    digest = hashlib.sha256()
    for value in model.get_weights():
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build(seed):
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    inputs, branches = [], []
    for branch, size, pool, offset in (("global", 2000, 5, 0), ("local", 200, 2, 2)):
        inp = tf.keras.Input((size, 1), dtype="float32", name=branch + "_flux")
        x = inp
        for block, filters in enumerate((16, 32)):
            x = tf.keras.layers.Conv1D(filters, 5, strides=1, padding="same", activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + offset + block),
                bias_initializer="zeros", name=f"{branch}_conv{block + 1}")(x)
            x = tf.keras.layers.MaxPooling1D(pool, strides=pool, padding="valid", name=f"{branch}_pool{block + 1}")(x)
        inputs.append(inp)
        branches.append(tf.keras.layers.Flatten(name=branch + "_flatten")(x))
    x = tf.keras.layers.Concatenate(name="fusion")(branches)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 4),
        bias_initializer="zeros", name="hidden")(x)
    x = tf.keras.layers.Dropout(.2, seed=seed + 5, name="dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 6),
        bias_initializer="zeros", name="confirmed_probability")(x)
    model = tf.keras.Model(inputs, out, name="kepler_dual_view_v1")
    require(model.count_params() == 138561, "CNN parameter count mismatch")
    require(model.get_layer("global_flatten").output.shape[-1] == 2560, "global flatten mismatch")
    require(model.get_layer("local_flatten").output.shape[-1] == 1600, "local flatten mismatch")
    require(model.get_layer("fusion").output.shape[-1] == 4160, "fusion mismatch")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001, beta_1=.9, beta_2=.999,
        epsilon=1e-7, global_clipnorm=1., weight_decay=None), loss="binary_crossentropy", jit_compile=False)
    return model


def validate_views(views):
    require(set(views) == {"global_flux", "local_flux"}, "forbidden CNN input fields")
    n = len(views["global_flux"])
    for key, size in (("global_flux", 2000), ("local_flux", 200)):
        a = views[key]
        require(a.shape == (n, size, 1) and a.dtype == np.float32, "CNN shape/dtype mismatch")
        require(np.isfinite(a).all(), "non-finite CNN input")


def predict(model, views, batch_size=64):
    validate_views(views)
    result = np.concatenate([model({k: v[i:i + batch_size] for k, v in views.items()}, training=False).numpy().ravel()
                            for i in range(0, len(views["global_flux"]), batch_size)])
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "invalid CNN predictions")
    return result


def verify_probability(expected, actual):
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "non-finite replay")
    require(np.shape(expected) == np.shape(actual), "replay shape mismatch")
    require(np.allclose(expected, actual, rtol=0, atol=1e-7), "saved-model prediction replay failed")
    require(np.array_equal(np.asarray(expected) >= .5, np.asarray(actual) >= .5), "replay predicted labels changed")


def fit(model, training, training_labels, validation, validation_labels, seed, *, epochs=100, patience=10):
    """Only explicit training and validation arrays are accepted; no cohort/test object."""
    validate_views(training); validate_views(validation)
    y = np.asarray(training_labels, dtype=np.float32).reshape(-1, 1)
    vy = np.asarray(validation_labels, dtype=np.float32).reshape(-1, 1)
    require(len(y) == len(training["global_flux"]) and len(vy) == len(validation["global_flux"]), "training label alignment")
    require(set(np.unique(y)) == {0, 1} and set(np.unique(vy)) == {0, 1}, "invalid training/validation labels")
    loss_function = tf.keras.losses.BinaryCrossentropy()

    @tf.function(input_signature=[tf.TensorSpec((None, 2000, 1), tf.float32),
                                 tf.TensorSpec((None, 200, 1), tf.float32), tf.TensorSpec((None, 1), tf.float32)])
    def step(global_flux, local_flux, labels):
        with tf.GradientTape() as tape:
            probability = model({"global_flux": global_flux, "local_flux": local_flux}, training=True)
            loss = loss_function(labels, probability)
            tf.debugging.assert_all_finite(loss, "non-finite training loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "non-finite training gradient")
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for variable in model.trainable_variables:
            tf.debugging.assert_all_finite(variable, "non-finite model weights")
        return loss

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience,
        min_delta=0, restore_best_weights=True)
    callback.set_model(model); callback.on_train_begin(); model.stop_training = False
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        begin = time.perf_counter()
        order = rng.permutation(len(y))
        for start in range(0, len(y), 64):
            batch = order[start:start + 64]
            step(training["global_flux"][batch], training["local_flux"][batch], y[batch])
        before = weight_hash(model)
        tp, vp = predict(model, training), predict(model, validation)
        train_loss = float(loss_function(y, tp.reshape(-1, 1)).numpy())
        val_loss = float(loss_function(vy, vp.reshape(-1, 1)).numpy())
        require(np.isfinite([train_loss, val_loss]).all(), "non-finite epoch loss")
        require(weight_hash(model) == before, "evaluation changed model weights")
        history.append({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss,
            "training_roc_auc": float(roc_auc_score(y.ravel(), tp)), "validation_roc_auc": float(roc_auc_score(vy.ravel(), vp)),
            "epoch_seconds": time.perf_counter() - begin})
        callback.on_epoch_end(epoch, {"loss": train_loss, "val_loss": val_loss})
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}: validation loss={val_loss:.6f}", flush=True)
        if model.stop_training:
            break
    callback.on_train_end()
    best = min(history, key=lambda row: row["validation_loss"])
    restored = float(loss_function(vy, predict(model, validation).reshape(-1, 1)).numpy())
    require(abs(restored - best["validation_loss"]) <= 1e-7, "best validation checkpoint restoration failed")
    return history, {"best_epoch": best["epoch"], "stopping_epoch": len(history),
        "best_validation_loss": best["validation_loss"], "epoch_cap_reached": len(history) == epochs,
        "status": "epoch_cap_reached" if len(history) == epochs else "early_stopped",
        "optimizer_state_note": "Saved optimizer reflects stopping epoch; restored weights reflect best epoch. No within-fit resume."}
