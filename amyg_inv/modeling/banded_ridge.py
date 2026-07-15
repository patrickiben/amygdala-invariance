"""Banded (Tikhonov) ridge with a separate regularizer per feature space.

This is a light, dependency-free implementation adequate for the synthetic scaffold and
small real fits. For production runs prefer `himalaya` (Dupre la Tour et al. 2022), which
does a *joint* per-band hyperparameter search; here we select each band's lambda with an
independent inner CV and then solve the joint banded system exactly via the normal
equations with a block-diagonal penalty. That simplification is documented, not hidden.

Naive single-lambda ridge and raw concatenation are intentionally NOT offered: they are
unfair to whichever band has higher intrinsic dimensionality, which is exactly the failure
the pre-registration excludes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def kfold_blocks(n: int, k: int) -> List[np.ndarray]:
    """Contiguous time-block folds (fMRI samples are autocorrelated; blocks, not shuffles)."""
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def r2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mu = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mu) ** 2, axis=0)
    ss_tot = np.where(ss_tot <= 1e-12, np.nan, ss_tot)
    return 1.0 - ss_res / ss_tot


def _standardizer(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _penalized_solve(x: np.ndarray, y: np.ndarray, band_dims: Sequence[int],
                     lambdas: Sequence[float]) -> np.ndarray:
    """(X^T X + diag(penalty))^{-1} X^T y  with a per-band lambda on the diagonal."""
    penalty = np.concatenate([np.full(d, lam) for d, lam in zip(band_dims, lambdas)])
    a = x.T @ x + np.diag(penalty)
    b = x.T @ y
    return np.linalg.solve(a, b)


@dataclass
class BandedRidge:
    band_dims: List[int]            # feature count per band, in concatenation order
    lambdas_grid: List[float]
    n_folds: int = 5

    # -- lambda selection: independent inner CV per band ----------------------
    def select_lambdas(self, bands: List[np.ndarray], y: np.ndarray) -> List[float]:
        chosen = []
        for xb in bands:
            best_lam, best_score = self.lambdas_grid[0], -np.inf
            for lam in self.lambdas_grid:
                score = self._cv_score_single(xb, y, lam)
                if score > best_score:
                    best_score, best_lam = score, lam
            chosen.append(best_lam)
        return chosen

    def _cv_score_single(self, xb: np.ndarray, y: np.ndarray, lam: float) -> float:
        n = xb.shape[0]
        folds = kfold_blocks(n, self.n_folds)
        scores = []
        for i in range(self.n_folds):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(self.n_folds) if j != i])
            mu, sd = _standardizer(xb[train])
            xtr = (xb[train] - mu) / sd
            xte = (xb[test] - mu) / sd
            ymu = y[train].mean(axis=0)
            w = _penalized_solve(xtr, y[train] - ymu, [xb.shape[1]], [lam])
            pred = xte @ w + ymu
            r2 = r2_per_column(y[test], pred)
            scores.append(np.nanmean(r2))
        return float(np.nanmean(scores))

    # -- cross-validated fit with GIVEN lambdas (used inside bootstrap) -------
    def cv_r2(self, bands: List[np.ndarray], y: np.ndarray,
              lambdas: Sequence[float]) -> np.ndarray:
        """Return per-column cross-validated R^2 using fixed per-band lambdas."""
        n = bands[0].shape[0]
        folds = kfold_blocks(n, self.n_folds)
        preds = np.empty_like(y)
        for i in range(self.n_folds):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(self.n_folds) if j != i])
            xtr_parts, xte_parts = [], []
            for xb in bands:
                mu, sd = _standardizer(xb[train])
                xtr_parts.append((xb[train] - mu) / sd)
                xte_parts.append((xb[test] - mu) / sd)
            xtr = np.concatenate(xtr_parts, axis=1)
            xte = np.concatenate(xte_parts, axis=1)
            ymu = y[train].mean(axis=0)
            w = _penalized_solve(xtr, y[train] - ymu, self.band_dims, lambdas)
            preds[test] = xte @ w + ymu
        return r2_per_column(y, preds)


def fit_r2(bands: List[np.ndarray], y: np.ndarray, lambdas_grid: List[float],
           n_folds: int = 5, lambdas: Sequence[float] | None = None):
    """Convenience: build a BandedRidge, (optionally) select lambdas, return (r2, lambdas)."""
    br = BandedRidge(band_dims=[b.shape[1] for b in bands],
                     lambdas_grid=lambdas_grid, n_folds=n_folds)
    if lambdas is None:
        lambdas = br.select_lambdas(bands, y)
    return br.cv_r2(bands, y, lambdas), list(lambdas)
