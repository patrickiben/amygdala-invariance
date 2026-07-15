import numpy as np

from amyg_inv.modeling.banded_ridge import fit_r2, r2_per_column


def test_recovers_linear_signal():
    rng = np.random.default_rng(0)
    T, D, V = 300, 10, 5
    X = rng.standard_normal((T, D))
    W = rng.standard_normal((D, V))
    y = X @ W + rng.standard_normal((T, V)) * 0.1
    r2, lambdas = fit_r2([X], y, lambdas_grid=[1.0, 10.0, 100.0], n_folds=5)
    assert np.nanmean(r2) > 0.9
    assert len(lambdas) == 1


def test_pure_noise_r2_near_zero():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 8))
    y = rng.standard_normal((300, 4))  # unrelated
    r2, _ = fit_r2([X], y, lambdas_grid=[100.0, 1000.0], n_folds=5)
    assert np.nanmean(r2) < 0.15


def test_r2_per_column_perfect():
    y = np.arange(20).reshape(-1, 1).astype(float)
    assert np.isclose(r2_per_column(y, y)[0], 1.0)
