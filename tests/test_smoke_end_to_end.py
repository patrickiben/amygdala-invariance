"""End-to-end: the three synthetic ground-truth worlds must produce the right verdict
through the REAL pipeline (ridge -> partition -> ceiling -> gate -> rule).
"""
import numpy as np
import pytest

from amyg_inv.config import load_prereg
from amyg_inv.data import synthetic
from amyg_inv.pipeline import Overrides, run


@pytest.mark.parametrize("scenario,expected", [
    ("H1", "H1"),
    ("H0", "H0"),
    ("inapplicable", "INAPPLICABLE"),
])
def test_scenario_verdict(scenario, expected):
    cfg = load_prereg()
    rng = np.random.default_rng(cfg.seed)
    feat, bold, tsnr = synthetic.make(scenario, cfg, rng, sizes=synthetic.FAST)
    res = run(feat, bold, tsnr, cfg, rng, overrides=Overrides(n_boot=60, n_folds=5))
    assert res.verdict.branch == expected, res.verdict.reason
