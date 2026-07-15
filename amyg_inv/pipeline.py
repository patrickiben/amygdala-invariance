"""Wire the pieces: features + BOLD -> per-subregion CIs -> reliability gate -> decision.

Everything a claim rests on is an interval (subject bootstrap), never a point. All encoders
in a subregion are normalized by the SAME ceiling, so encoder-vs-encoder comparisons in the
decision rule are invariant to the (approximate) ceiling scale; the ceiling only gates
reliability and sets the units of the delta threshold on EmoNet-unique variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .decision.rule import Interval, SubregionResult, Verdict, decide
from .modeling.banded_ridge import BandedRidge
from .modeling.noise_ceiling import group_mean_reliability_r2, group_signal
from .modeling.variance_partition import unique_variance
from .reliability.gate import GateResult, evaluate_gate


@dataclass
class Overrides:
    n_boot: Optional[int] = None
    n_folds: Optional[int] = None


@dataclass
class RunResult:
    verdict: Verdict
    subregions: List[SubregionResult]
    gates: Dict[str, GateResult]
    diagnostics: Dict[str, dict] = field(default_factory=dict)
    prereg_sha256: str = ""


def _ci(draws, central, level):
    if len(draws) < 2:
        return central, central
    a = (1 - level) / 2 * 100
    return float(np.percentile(draws, a)), float(np.percentile(draws, 100 - a))


def _encoder_r2(feat_e, y, lam, n_folds):
    br = BandedRidge([feat_e.shape[1]], lambdas_grid=[lam], n_folds=n_folds)
    return float(np.nanmean(br.cv_r2([feat_e], y, [lam])))


def compute_subregion(name: str, feat: Dict[str, np.ndarray], bold_sr: np.ndarray,
                      tsnr: float, cfg, rng, n_boot: int, n_folds: int):
    encoders = cfg.all_encoders
    lambdas_grid = cfg.d("fitting", "lambdas")
    ci_level = cfg.d("decision", "ci_level")
    eps = 1e-6

    y_full = group_signal(bold_sr)  # (T, V) reliable signal

    # -- select one lambda per encoder ONCE on the full data (reused in bootstrap) --
    lam: Dict[str, float] = {}
    for e in encoders:
        br = BandedRidge([feat[e].shape[1]], lambdas_grid=lambdas_grid, n_folds=n_folds)
        lam[e] = br.select_lambdas([feat[e]], y_full)[0]

    # -- central estimates --
    # Ceiling = reliability of the group-mean target (bounded [0,1]); encoder R^2 are
    # normalized by this FIXED central ceiling so the point estimate stays inside its own
    # bootstrap CI. The ceiling's own uncertainty is tracked separately for the gate.
    ceiling_c = group_mean_reliability_r2(bold_sr)
    denom = max(ceiling_c, eps)
    raw_r2 = {e: _encoder_r2(feat[e], y_full, lam[e], n_folds) for e in encoders}
    ue_raw_c = unique_variance(feat, cfg.target, [cfg.lowlevel], y_full,
                               lambdas_grid, n_folds, lambdas=lam)

    # -- subject bootstrap for CIs --
    N = bold_sr.shape[0]
    draws_norm = {e: [] for e in encoders}
    draws_ue = []
    draws_ceiling = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(idx)) < 2:
            continue
        yb = group_signal(bold_sr[idx])
        draws_ceiling.append(group_mean_reliability_r2(bold_sr[idx]))
        for e in encoders:
            draws_norm[e].append(_encoder_r2(feat[e], yb, lam[e], n_folds) / denom)
        ueb = unique_variance(feat, cfg.target, [cfg.lowlevel], yb,
                              lambdas_grid, n_folds, lambdas=lam)
        draws_ue.append(ueb / denom)

    ceiling_lo, ceiling_hi = _ci(draws_ceiling, ceiling_c, ci_level)

    def interval(central_raw, draws):
        point = central_raw / denom
        lo, hi = _ci(draws, point, ci_level)
        # report the bootstrap median as the central estimate so lo <= central <= hi always
        central = float(np.median(draws)) if len(draws) >= 2 else point
        central = min(max(central, lo), hi)
        return Interval(central=central, lo=lo, hi=hi)

    encoder_ci = {e: interval(raw_r2[e], draws_norm[e]) for e in encoders}
    ue_ci = interval(ue_raw_c, draws_ue)

    any_above = max(raw_r2.values()) > float(cfg.d("reliability", "ceiling_r2_min"))
    gate = evaluate_gate(name, tsnr, ceiling_lo, cfg, venous_flip=None)

    sr = SubregionResult(
        name=name,
        reliability_pass=gate.passed,
        encoder_r2=encoder_ci,
        emonet_unique_vs_lowlevel=ue_ci,
        any_encoder_above_ceiling=any_above,
    )
    diag = {
        "ceiling": [ceiling_lo, ceiling_c, ceiling_hi],
        "raw_r2": raw_r2,
        "emonet_unique_raw": ue_raw_c,
        "lambdas": lam,
        "tsnr": tsnr,
    }
    return sr, gate, diag


def run(feat: Dict[str, np.ndarray], bold: Dict[str, np.ndarray],
        tsnr: Dict[str, float], cfg, rng, overrides: Overrides = Overrides()) -> RunResult:
    n_boot = overrides.n_boot if overrides.n_boot is not None else int(cfg.d("fitting", "bootstrap"))
    n_folds = overrides.n_folds if overrides.n_folds is not None else int(cfg.d("fitting", "n_folds"))

    results, gates, diags = [], {}, {}
    for name in cfg.subregions:
        sr, gate, diag = compute_subregion(name, feat, bold[name], tsnr[name],
                                            cfg, rng, n_boot, n_folds)
        results.append(sr)
        gates[name] = gate
        diags[name] = diag

    verdict = decide(results, cfg)
    return RunResult(verdict=verdict, subregions=results, gates=gates,
                     diagnostics=diags, prereg_sha256=cfg.sha256)
