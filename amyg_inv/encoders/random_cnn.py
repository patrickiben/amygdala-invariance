"""Random-weight CNN encoder (break-case adversary #2).

An architecture matched to EmoNet's backbone but with random (untrained) weights. Random
CNNs are known to be surprisingly strong visual-feature banks; if EmoNet's amygdala
prediction is carried by generic convolutional structure rather than by EMOTION supervision,
a random-weight net of the same shape should match it. Deterministic given the prereg seed.

The scaffold ships a dependency-free random *convolutional projection* (fixed random filters
+ pooling) so the break-case runs without torch. Swap in a torchvision AlexNet with
`weights=None` for the real run — same idea, matched to EmoNet's actual backbone.
"""
from __future__ import annotations

import numpy as np

from .base import Encoder
from ..modeling.hemodynamics import frames_to_tr


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame @ np.array([0.299, 0.587, 0.114])
    return frame if frame.ndim == 2 else frame.mean(axis=-1)


class RandomCNNEncoder(Encoder):
    name = "random"

    def __init__(self, dim: int = 32, n_filters: int = 8, ksize: int = 7, seed: int = 0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._filters = rng.standard_normal((n_filters, ksize, ksize))
        self._proj = rng.standard_normal((n_filters, dim))  # pooled-energy -> dim

    def features(self, frames: np.ndarray, fps: float, tr: float) -> np.ndarray:
        from scipy.signal import fftconvolve

        F = frames.shape[0]
        pooled = np.zeros((F, self._filters.shape[0]), dtype=float)
        for f in range(F):
            gray = _to_gray(frames[f].astype(float))
            for k in range(self._filters.shape[0]):
                resp = fftconvolve(gray, self._filters[k], mode="same")
                pooled[f, k] = np.sqrt(np.mean(np.maximum(resp, 0.0) ** 2))  # ReLU energy
        feat = pooled @ self._proj
        return frames_to_tr(feat, fps, tr)
