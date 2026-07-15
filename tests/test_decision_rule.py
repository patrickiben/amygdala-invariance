"""Unit tests for the frozen decision rule, constructing SubregionResults directly so the
rule logic is tested independently of the fitting pipeline.
"""
from amyg_inv.config import load_prereg
from amyg_inv.decision.rule import Interval, SubregionResult, decide

CFG = load_prereg()


def _sr(name, emonet, lowlevel, random, dino, clip, ue, passed=True, above=True):
    return SubregionResult(
        name=name,
        reliability_pass=passed,
        encoder_r2={
            "emonet": emonet, "lowlevel": lowlevel, "random": random,
            "dinov2": dino, "clip": clip,
        },
        emonet_unique_vs_lowlevel=ue,
        any_encoder_above_ceiling=above,
    )


def _iv(c, w=0.03):
    return Interval(central=c, lo=c - w, hi=c + w)


def test_h1_verdict():
    # EmoNet high, dino/clip recover it, low-level/random low, EmoNet-unique well above delta
    srs = [_sr(f"S{i}", _iv(0.80), _iv(0.10), _iv(0.08), _iv(0.78), _iv(0.76), _iv(0.40))
           for i in range(4)]
    assert decide(srs, CFG).branch == "H1"


def test_h0_break_case():
    # low-level reaches EmoNet -> break-case triggers; EmoNet-unique ~ 0
    srs = [_sr(f"S{i}", _iv(0.50), _iv(0.49), _iv(0.30), _iv(0.48), _iv(0.47), _iv(0.00))
           for i in range(4)]
    assert decide(srs, CFG).branch == "H0"


def test_h0_degenerate_all_overlap():
    # all five encoders indistinguishable (mutually overlapping CIs)
    srs = [_sr(f"S{i}", _iv(0.40), _iv(0.40), _iv(0.40), _iv(0.40), _iv(0.40), _iv(0.01))
           for i in range(4)]
    assert decide(srs, CFG).branch == "H0"


def test_abstain_inconclusive():
    # EmoNet is separated from the adversaries (NOT degenerate), the break-case does not
    # trigger, but EmoNet-unique cannot exclude delta and the high-level encoders do not
    # clearly recover -> genuinely inconclusive -> ABSTAIN.
    srs = [_sr(f"S{i}",
               Interval(0.30, 0.20, 0.40),   # emonet, separated from random below
               Interval(0.10, 0.03, 0.17),   # lowlevel
               Interval(0.08, 0.00, 0.16),   # random (emonet.lo 0.20 > random.hi 0.16 -> not all-overlap)
               Interval(0.12, 0.04, 0.20),   # dinov2 (hi 0.20 < 0.8*0.30 -> does not "recover")
               Interval(0.12, 0.04, 0.20),   # clip
               Interval(0.03, 0.00, 0.06))   # emonet-unique lo 0.0 !> delta 0.02
           for i in range(4)]
    assert decide(srs, CFG).branch == "ABSTAIN"


def test_inapplicable_too_few_pass():
    srs = [_sr("S0", _iv(0.8), _iv(0.1), _iv(0.08), _iv(0.78), _iv(0.76), _iv(0.4), passed=True)]
    srs += [_sr(f"S{i}", _iv(0.8), _iv(0.1), _iv(0.08), _iv(0.78), _iv(0.76), _iv(0.4),
                passed=False) for i in range(1, 4)]
    assert decide(srs, CFG).branch == "INAPPLICABLE"


def test_inapplicable_nothing_above_ceiling():
    srs = [_sr(f"S{i}", _iv(0.01), _iv(0.01), _iv(0.0), _iv(0.0), _iv(0.0), _iv(0.0),
               passed=True, above=False) for i in range(4)]
    assert decide(srs, CFG).branch == "INAPPLICABLE"
