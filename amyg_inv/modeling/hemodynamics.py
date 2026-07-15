"""Frame -> TR alignment and hemodynamic-lag modeling.

Encoder features are extracted per film frame (~24 fps); BOLD is sampled per TR (~1 s at
1.5 T for ds002837). Features must be (a) aggregated to the TR grid and (b) convolved with a
hemodynamic response before regression. Getting the frame->TR offset wrong silently corrupts
EVERY encoder equally (a shared upstream nuisance, quantified by data/stimulus.py).

The synthetic pipeline generates features already on the TR grid, so these are used mainly by
the real-data path — but they are implemented and unit-testable.
"""
from __future__ import annotations

import numpy as np


def spm_hrf(tr: float, length: float = 32.0) -> np.ndarray:
    """Canonical double-gamma HRF sampled at the TR (SPM defaults)."""
    from scipy.stats import gamma  # local import keeps scipy optional at import time

    dt = tr
    t = np.arange(0, length + dt, dt)
    peak = gamma.pdf(t, 6)             # response
    undershoot = gamma.pdf(t, 16)     # undershoot
    hrf = peak - undershoot / 6.0
    hrf = hrf / np.sum(hrf)
    return hrf


def convolve_hrf(features: np.ndarray, tr: float) -> np.ndarray:
    """features: (T, D) on the TR grid -> HRF-convolved (T, D), causal, truncated to T."""
    hrf = spm_hrf(tr)
    T = features.shape[0]
    out = np.empty_like(features)
    for d in range(features.shape[1]):
        out[:, d] = np.convolve(features[:, d], hrf)[:T]
    return out


def frames_to_tr(frame_features: np.ndarray, fps: float, tr: float) -> np.ndarray:
    """Average per-frame features (F, D) into TR bins -> (T, D)."""
    n_frames = frame_features.shape[0]
    frames_per_tr = fps * tr
    n_tr = int(np.floor(n_frames / frames_per_tr))
    out = np.empty((n_tr, frame_features.shape[1]), dtype=float)
    for t in range(n_tr):
        lo = int(round(t * frames_per_tr))
        hi = int(round((t + 1) * frames_per_tr))
        out[t] = frame_features[lo:hi].mean(axis=0)
    return out
