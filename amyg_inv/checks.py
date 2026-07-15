"""Positive-control check for sensory-relay regions (e.g. LGN).

A positive control does NOT test encoder-specificity — its job is to prove the harness is
alive: it must (a) PASS the reliability gate on a structure that certainly has signal, and
(b) detect strong encoder prediction. The H1/H0 verdict is not the point (indeed H0 —
"low-level dominates" — is the *expected* outcome for a first-order visual relay). If a
positive control FAILS this check, the pipeline is broken and every null elsewhere is suspect.
"""
from __future__ import annotations

from typing import Tuple

from .pipeline import RunResult


def positive_control_ok(res: RunResult, cfg, min_r2: float | None = None) -> Tuple[bool, str]:
    """Return (ok, reason). ok iff the reliability gate passes on a majority of subregions AND
    some encoder achieves a ceiling-normalized R^2 above the pre-registered floor."""
    if min_r2 is None:
        min_r2 = float(cfg.raw.get("positive_control_min_r2", 0.4))
    majority = int(cfg.d("decision", "subregion_majority"))
    passing = [s for s in res.subregions if s.reliability_pass]
    if len(passing) < majority:
        return False, (f"reliability gate FAILED on a signal-rich control: "
                       f"{len(passing)}/{len(res.subregions)} subregions passed "
                       f"(< required {majority}) — the gate is miscalibrated or the pipeline is broken")
    max_r2 = max((iv.central for s in passing for iv in s.encoder_r2.values()), default=float("-inf"))
    if max_r2 < min_r2:
        return False, (f"no strong encoder prediction where it must exist: "
                       f"max encoder R^2 {max_r2:.2f} < floor {min_r2:.2f} — pipeline likely broken")
    return True, (f"OK: gate passed ({len(passing)}/{len(res.subregions)}), "
                  f"max encoder R^2 {max_r2:.2f} >= {min_r2:.2f} (harness detects signal)")
