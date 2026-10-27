"""Stage 5B architecture with only hidden width changed from 32 to 128."""
import random
import hashlib
import numpy as np
import tensorflow as tf
from kepler_baselines.inputs import require


def branch_hash(model):
    digest = hashlib.sha256()
    for name in ('global_conv1', 'global_conv2', 'local_conv1', 'local_conv2'):
        for value in model.get_layer(name).get_weights():
            digest.update(str(value.shape).encode())
            digest.update(value.dtype.str.encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def checked_initialization(seed):
    """Return a fresh wide model; compare branches, not differently shaped heads."""
    from kepler_cnn_variant.model import build as build32
    from kepler_cnn.model import weight_hash
    parent = build32(seed)
    expected_branch = branch_hash(parent)
    del parent
    first = build(seed)
    expected_wide = weight_hash(first)
    require(branch_hash(first) == expected_branch, 'Branch initialization changed')
    del first
    model = build(seed)
    require(weight_hash(model) == expected_wide, '128-unit initialization is not deterministic')
    require(branch_hash(model) == expected_branch, 'Branch initialization is not deterministic')
    return model, dict(seed=int(seed), branch_sha256=expected_branch, initial_weight_sha256=expected_wide,
                      parameters=model.count_params(), hidden_units=128)


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
    x = tf.keras.layers.Dense(128, activation="linear", kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + 4),
        bias_initializer="zeros", name="hidden")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.01, name="hidden_leaky_relu")(x)
    x = tf.keras.layers.Dropout(.2, seed=seed + 5, name="dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 6),
        bias_initializer="zeros", name="confirmed_probability")(x)
    model = tf.keras.Model(inputs, out, name="kepler_dual_view_v1")
    require(model.count_params() == 538113, "CNN parameter count mismatch")
    require(model.get_layer("global_flatten").output.shape[-1] == 2560, "global flatten mismatch")
    require(model.get_layer("local_flatten").output.shape[-1] == 1600, "local flatten mismatch")
    require(model.get_layer("fusion").output.shape[-1] == 4160, "fusion mismatch")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001, beta_1=.9, beta_2=.999,
        epsilon=1e-7, global_clipnorm=1., weight_decay=None), loss="binary_crossentropy", jit_compile=False)
    return model
