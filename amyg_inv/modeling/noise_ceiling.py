"""Noise ceiling for single-viewing (NO within-subject repeats) movie fMRI.

A repeat-based ceiling is impossible here, and we do NOT substitute a point ISC value.
We use a leave-one-subject-out predictive ceiling *in R^2 units*, consistent with how the
encoding models are scored:

    for each held-out subject s, predict its timeseries with the MEAN of the other N-1
    subjects; ceiling R^2 = variance of s explained by that mean, averaged over s and voxels.

This is the best the *reliable, subject-shared* signal can do for a new subject — an honest
upper bound on any encoder trained on the group signal. It is an approximation of the true
single-subject noise ceiling, so it is always reported as an interval (subject bootstrap),
never a point, and its uncertainty is propagated into every ceiling-normalized number.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .banded_ridge import r2_per_column


def _pearson_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    den = np.where(den < 1e-12, np.nan, den)
    return num / den


def group_mean_reliability_r2(bold: np.ndarray, max_splits: int = 20) -> float:
    """Reliability of the across-subject MEAN timeseries (the model's target), in R^2 units.

    Split subjects into complementary halves, correlate the two half-means per voxel, then
    Spearman-Brown-correct the split-half reliability up to the full N-subject mean. This is a
    proper noise ceiling for an encoding model fit to the group mean (no stimulus repeats
    needed) and is bounded in [0, 1]. Deterministic (fixed complementary splits, no RNG).
    """
    from itertools import combinations

    n = bold.shape[0]
    if n < 2:
        raise ValueError("need >=2 subjects")
    half = n // 2
    seen, splits = set(), []
    for c in combinations(range(n), half):
        cset = set(c)
        comp = tuple(sorted(set(range(n)) - cset))
        key = tuple(sorted((tuple(sorted(cset)), comp)))
        if key in seen:
            continue
        seen.add(key)
        splits.append((sorted(cset), list(comp)))
        if len(splits) >= max_splits:
            break
    rs = []
    for a, b in splits:
        ma = bold[a].mean(axis=0)
        mb = bold[b].mean(axis=0)
        rs.append(np.nanmean(_pearson_cols(ma, mb)))
    r_half = max(float(np.nanmean(rs)), 0.0)
    return 2 * r_half / (1 + r_half) if r_half > 0 else 0.0


def loo_ceiling_r2(bold: np.ndarray) -> float:
    """bold: (n_subjects, T, V) -> scalar ceiling R^2 (mean over held-out subjects and voxels)."""
    n = bold.shape[0]
    if n < 2:
        raise ValueError("need >=2 subjects for a leave-one-subject-out ceiling")
    per_subject = []
    total = bold.sum(axis=0)
    for s in range(n):
        others_mean = (total - bold[s]) / (n - 1)
        r2 = r2_per_column(bold[s], others_mean)
        per_subject.append(np.nanmean(r2))
    return float(np.nanmean(per_subject))


def group_signal(bold: np.ndarray) -> np.ndarray:
    """Across-subject mean timeseries (the reliable signal the encoders are fit to)."""
    return bold.mean(axis=0)


def bootstrap_ceiling_ci(bold: np.ndarray, n_boot: int, level: float,
                         rng: np.random.Generator) -> Tuple[float, float, float]:
    n = bold.shape[0]
    central = loo_ceiling_r2(bold)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            continue
        draws.append(loo_ceiling_r2(bold[idx]))
    lo, hi = _percentile_ci(draws, level, central)
    return central, lo, hi


def _percentile_ci(draws, level: float, fallback: float) -> Tuple[float, float]:
    if len(draws) < 2:
        return fallback, fallback
    a = (1 - level) / 2 * 100
    return float(np.percentile(draws, a)), float(np.percentile(draws, 100 - a))
