"""The FROZEN decision rule (protocol section vi). Intervals, never bare points.

Branches, in the pre-registered precedence:
  INAPPLICABLE : no reliability-passing subregion has ANY encoder above the noise ceiling,
                 OR fewer than `subregion_majority` subregions clear the reliability gate.
                 -> a statement about the DATASET's measurement floor, not about the claim.
  H1           : in >= subregion_majority reliability-passing subregions,
                 EmoNet-UNIQUE (vs low-level) CI excludes delta, AND >= min_recovering_highlevel
                 high-level encoders recover the same subregion variance, AND the break-case
                 does NOT trigger.
  H0           : in >= subregion_majority passing subregions, the break-case triggers OR all
                 encoders are mutually indistinguishable (degeneracy).
  ABSTAIN      : none of the above -- CIs too wide to separate H1 from H0 given delta.

The break-case (protocol section ii) is adversarial-to-H0: we only call "artifact" when an
impoverished encoder's *lower* CI bound still reaches k * EmoNet's *central* estimate, so
noise cannot satisfy it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Interval:
    central: float
    lo: float
    hi: float

    def excludes_above(self, thr: float) -> bool:
        """CI lies entirely above thr (lower bound > thr)."""
        return self.lo > thr

    def overlaps(self, other: "Interval") -> bool:
        return not (self.hi < other.lo or other.hi < self.lo)


@dataclass
class SubregionResult:
    name: str
    reliability_pass: bool
    encoder_r2: Dict[str, Interval]          # ceiling-normalized R^2 CI per encoder (alone)
    emonet_unique_vs_lowlevel: Interval      # ceiling-normalized unique variance CI
    any_encoder_above_ceiling: bool          # did any encoder beat the raw ceiling at all


@dataclass
class Verdict:
    branch: str                              # H1 | H0 | ABSTAIN | INAPPLICABLE
    reason: str
    n_passing: int
    per_subregion: Dict[str, Dict] = field(default_factory=dict)


def _break_triggers(sr: SubregionResult, cfg) -> bool:
    k = cfg.d("decision", "break_case_k")
    ref = cfg.d("decision", "break_case_reference")
    emonet = sr.encoder_r2[cfg.target]
    e_ref = emonet.central if ref == "central" else emonet.hi
    for adv in cfg.adversaries:
        if adv not in sr.encoder_r2:
            continue
        if sr.encoder_r2[adv].lo > k * e_ref:
            return True
    return False


def _highlevel_recovers(sr: SubregionResult, cfg) -> int:
    frac = cfg.d("decision", "recover_fraction")
    e_central = sr.encoder_r2[cfg.target].central
    count = 0
    for hl in cfg.highlevel:
        if hl in sr.encoder_r2 and sr.encoder_r2[hl].hi >= frac * e_central:
            count += 1
    return count


def _all_indistinguishable(sr: SubregionResult, cfg) -> bool:
    encs = [sr.encoder_r2[e] for e in cfg.all_encoders if e in sr.encoder_r2]
    for i in range(len(encs)):
        for j in range(i + 1, len(encs)):
            if not encs[i].overlaps(encs[j]):
                return False
    return True


def decide(subregions: List[SubregionResult], cfg) -> Verdict:
    majority = int(cfg.d("decision", "subregion_majority"))
    delta = float(cfg.d("decision", "delta"))
    min_recover = int(cfg.d("decision", "min_recovering_highlevel"))

    passing = [s for s in subregions if s.reliability_pass]
    n_pass = len(passing)

    per = {}
    n_h1 = n_break = n_indist = 0
    for s in passing:
        brk = _break_triggers(s, cfg)
        rec = _highlevel_recovers(s, cfg)
        indist = _all_indistinguishable(s, cfg)
        h1 = s.emonet_unique_vs_lowlevel.excludes_above(delta) and rec >= min_recover and not brk
        n_h1 += int(h1)
        n_break += int(brk)
        n_indist += int(indist)
        per[s.name] = {
            "break_case": brk,
            "highlevel_recovering": rec,
            "all_indistinguishable": indist,
            "emonet_unique_ci": [s.emonet_unique_vs_lowlevel.lo,
                                 s.emonet_unique_vs_lowlevel.central,
                                 s.emonet_unique_vs_lowlevel.hi],
            "local_h1": h1,
        }

    # --- INAPPLICABLE: dataset measurement floor ---------------------------
    region = getattr(cfg, "region", "region")
    if n_pass < majority:
        return Verdict("INAPPLICABLE",
                       f"only {n_pass} of {len(subregions)} subregions cleared the reliability "
                       f"gate (< required {majority}); no adjudicable {region} signal on this dataset",
                       n_pass, per)
    if not any(s.any_encoder_above_ceiling for s in passing):
        return Verdict("INAPPLICABLE",
                       "no encoder (including EmoNet) beat the noise ceiling in any "
                       "reliability-passing subregion; dataset measurement floor, not a verdict",
                       n_pass, per)

    # --- H1 / H0 / ABSTAIN --------------------------------------------------
    if n_h1 >= majority:
        return Verdict("H1",
                       f"EmoNet-unique variance excludes delta and >={min_recover} high-level "
                       f"encoders recover the same variance in {n_h1}/{n_pass} subregions; break-case not triggered",
                       n_pass, per)
    if n_break >= majority or n_indist >= majority:
        why = "break-case triggered" if n_break >= majority else "all encoders indistinguishable (degeneracy)"
        return Verdict("H0",
                       f"{why} in >= {majority} subregions (break={n_break}, indistinguishable={n_indist})",
                       n_pass, per)
    return Verdict("ABSTAIN",
                   f"cannot separate H1 from H0 given delta={delta}: H1 in {n_h1}, break in {n_break}, "
                   f"indistinguishable in {n_indist} of {n_pass} passing subregions (CIs too wide)",
                   n_pass, per)
