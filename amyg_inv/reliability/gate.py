"""The amygdala-reliability GATE (protocol section iv): run and reported BEFORE any encoder
comparison. Each amygdala confound gets a *specific* control, not a caveat.

  1. Dropout / tSNR : per-subregion temporal SNR must clear a pre-set floor at 1.5 T.
  2. Venous confound: re-run the encoder comparison WITH and WITHOUT venous nuisance
     regression; if any encoder-ranking conclusion flips sign, abstain that subregion.
     (This module exposes the flag; the pipeline supplies the two rankings.)
  3. Reliability   : LOO/ISC ceiling R^2 CI must EXCLUDE ceiling_r2_min (else signal ~ 0).

A subregion must pass all applicable checks to enter the comparison; failures are reported
as 'abstained (uninterpretable signal)', never as evidence of null encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GateResult:
    subregion: str
    tsnr: float
    tsnr_pass: bool
    ceiling_lo: float
    ceiling_pass: bool
    venous_flip: Optional[bool]      # None if venous check not supplied
    venous_pass: bool
    passed: bool
    reason: str


def evaluate_gate(subregion: str, tsnr: float, ceiling_lo: float, cfg,
                  venous_flip: Optional[bool] = None) -> GateResult:
    tsnr_floor = float(cfg.d("reliability", "tsnr_floor"))
    ceil_min = float(cfg.d("reliability", "ceiling_r2_min"))

    tsnr_pass = tsnr >= tsnr_floor
    ceiling_pass = ceiling_lo > ceil_min
    venous_pass = (venous_flip is not True)   # a sign-flip fails; None or False passes

    passed = tsnr_pass and ceiling_pass and venous_pass
    reasons = []
    if not tsnr_pass:
        reasons.append(f"tSNR {tsnr:.1f} < floor {tsnr_floor:.1f}")
    if not ceiling_pass:
        reasons.append(f"ceiling lower-CI {ceiling_lo:.3f} <= min {ceil_min:.3f} (signal ~ 0)")
    if venous_flip is True:
        reasons.append("encoder ranking flips sign under venous nuisance regression")
    reason = "pass" if passed else "; ".join(reasons)
    return GateResult(subregion, tsnr, tsnr_pass, ceiling_lo, ceiling_pass,
                      venous_flip, venous_pass, passed, reason)
