from amyg_inv.config import load_prereg
from amyg_inv.reliability.gate import evaluate_gate


def test_gate_passes_reliable_subregion():
    cfg = load_prereg()
    g = evaluate_gate("LB", tsnr=55.0, ceiling_lo=0.3, cfg=cfg, venous_flip=False)
    assert g.passed


def test_gate_fails_low_tsnr():
    cfg = load_prereg()
    g = evaluate_gate("CM", tsnr=10.0, ceiling_lo=0.3, cfg=cfg)
    assert not g.passed
    assert "tSNR" in g.reason


def test_gate_fails_zero_ceiling():
    cfg = load_prereg()
    g = evaluate_gate("SF", tsnr=55.0, ceiling_lo=0.0, cfg=cfg)
    assert not g.passed


def test_gate_fails_on_venous_flip():
    cfg = load_prereg()
    g = evaluate_gate("AStr", tsnr=55.0, ceiling_lo=0.3, cfg=cfg, venous_flip=True)
    assert not g.passed
