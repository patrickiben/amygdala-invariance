import numpy as np

from amyg_inv.modeling.variance_partition import unique_variance


def test_unique_variance_is_large_when_target_adds_signal():
    rng = np.random.default_rng(0)
    T, V = 400, 4
    s_shared = rng.standard_normal((T, 3))
    s_target = rng.standard_normal((T, 3))
    y = s_target @ rng.standard_normal((3, V)) + 0.1 * rng.standard_normal((T, V))
    feat = {
        "target": np.hstack([s_target, rng.standard_normal((T, 5))]),
        "other": np.hstack([s_shared, rng.standard_normal((T, 5))]),  # unrelated to y
    }
    u = unique_variance(feat, "target", ["other"], y,
                        lambdas_grid=[1.0, 10.0, 100.0], n_folds=5)
    assert u > 0.5


def test_unique_variance_near_zero_when_target_redundant():
    rng = np.random.default_rng(1)
    T, V = 400, 4
    s = rng.standard_normal((T, 3))
    y = s @ rng.standard_normal((3, V)) + 0.1 * rng.standard_normal((T, V))
    # both feature spaces carry the SAME driving latent -> target adds ~nothing
    feat = {
        "target": np.hstack([s, rng.standard_normal((T, 5))]),
        "other": np.hstack([s, rng.standard_normal((T, 5))]),
    }
    u = unique_variance(feat, "target", ["other"], y,
                        lambdas_grid=[1.0, 10.0, 100.0], n_folds=5)
    assert u < 0.15
