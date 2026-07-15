"""The LGN positive control: proves the harness detects signal and passes the gate where it must.

On the low-level-dominated synthetic world (the LGN-realistic regime, since a first-order
visual relay carries the low-level image), the harness must:
  * PASS the reliability gate (LGN has strong, subject-shared signal),
  * detect strong encoder prediction (high ceiling-normalized R^2),
  * and return H0 — "low-level dominates" — which for a positive control is the CORRECT
    outcome, not an artifact (the inverted semantics).
If this fails, every null elsewhere (NAc/OFC RED) is suspect: the pipeline might just be dead.
"""
import numpy as np

from amyg_inv.checks import positive_control_ok
from amyg_inv.config import load_prereg
from amyg_inv.data import synthetic
from amyg_inv.pipeline import Overrides, run


def test_lgn_present_and_flagged_positive_control():
    cfg = load_prereg("lgn")
    assert cfg.raw.get("role") == "positive_control"
    assert cfg.raw.get("feasibility") == "GREEN"
    assert cfg.subregions == ["LGN_L", "LGN_R"]


def test_lgn_detects_signal_and_passes_gate():
    cfg = load_prereg("lgn")
    rng = np.random.default_rng(cfg.seed)
    # low-level-dominated world == the LGN-realistic regime
    feat, bold, tsnr = synthetic.make("H0", cfg, rng, sizes=synthetic.FAST)
    res = run(feat, bold, tsnr, cfg, rng, overrides=Overrides(n_boot=60, n_folds=5))
    ok, reason = positive_control_ok(res, cfg)
    assert ok, reason
    # H0 (low-level dominates) is the EXPECTED, correct outcome for a low-level relay
    assert res.verdict.branch == "H0", res.verdict.reason


def test_positive_control_would_flag_a_dead_pipeline():
    # If the region cannot be scored (inapplicable world), the positive-control check must FAIL —
    # that is exactly the alarm it exists to raise.
    cfg = load_prereg("lgn")
    rng = np.random.default_rng(cfg.seed)
    feat, bold, tsnr = synthetic.make("inapplicable", cfg, rng, sizes=synthetic.FAST)
    res = run(feat, bold, tsnr, cfg, rng, overrides=Overrides(n_boot=60, n_folds=5))
    ok, _ = positive_control_ok(res, cfg)
    assert not ok
