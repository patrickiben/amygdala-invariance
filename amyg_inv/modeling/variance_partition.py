"""Variance partitioning across feature spaces via nested set fits.

Unique variance of band B in a set S = R2(S) - R2(S minus B).
For the EmoNet-vs-low-level attribution we use S = {emonet, lowlevel}:
    U_emonet = R2({emonet, lowlevel}) - R2({lowlevel})
which is exactly "the amygdala variance EmoNet explains ABOVE the low-level baseline".
Shared variance = R2(S) - sum(unique_b). It can be negative (suppression); reported as-is.

All R2 here are cross-validated subregion scalars (mean over voxels), fit with fixed
per-band lambdas so the quantity is comparable inside the subject-bootstrap.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .banded_ridge import BandedRidge


def _r2_set(feat: Dict[str, np.ndarray], keys: List[str], y: np.ndarray,
            lambdas_grid: List[float], n_folds: int,
            lambdas: Dict[str, float] | None = None) -> float:
    bands = [feat[k] for k in keys]
    br = BandedRidge(band_dims=[b.shape[1] for b in bands],
                     lambdas_grid=lambdas_grid, n_folds=n_folds)
    if lambdas is None:
        lam = br.select_lambdas(bands, y)
    else:
        lam = [lambdas[k] for k in keys]
    r2 = br.cv_r2(bands, y, lam)
    return float(np.nanmean(r2))


def unique_variance(feat: Dict[str, np.ndarray], target: str, others: List[str],
                    y: np.ndarray, lambdas_grid: List[float], n_folds: int,
                    lambdas: Dict[str, float] | None = None) -> float:
    """Unique subregion variance attributable to `target` above `others`."""
    full = _r2_set(feat, [target] + others, y, lambdas_grid, n_folds, lambdas)
    reduced = _r2_set(feat, others, y, lambdas_grid, n_folds, lambdas)
    return full - reduced
