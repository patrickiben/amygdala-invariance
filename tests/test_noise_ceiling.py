import numpy as np

from amyg_inv.modeling.noise_ceiling import loo_ceiling_r2


def test_ceiling_high_for_shared_signal():
    rng = np.random.default_rng(0)
    N, T, V = 8, 300, 6
    signal = rng.standard_normal((T, V))
    bold = signal[None] + 0.6 * rng.standard_normal((N, T, V))
    assert loo_ceiling_r2(bold) > 0.3


def test_ceiling_near_zero_for_independent_noise():
    rng = np.random.default_rng(1)
    bold = rng.standard_normal((8, 300, 6))  # no shared signal
    assert loo_ceiling_r2(bold) < 0.05
