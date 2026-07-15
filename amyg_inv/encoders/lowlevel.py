"""Low-level control encoder (break-case adversary #1): no semantic/emotion supervision.

Gabor-energy at several orientations/scales (GIST-like) + mean luminance + frame-to-frame
motion energy. If EmoNet's amygdala advantage is really just tracking low-level image
statistics, THIS encoder should reproduce it. That is the point of including it.
"""
from __future__ import annotations

import numpy as np

from .base import Encoder
from ..modeling.hemodynamics import frames_to_tr


def _gabor_bank(n_orient: int = 4, n_scale: int = 2, ksize: int = 15):
    banks = []
    for s in range(n_scale):
        sigma = 2.0 * (s + 1)
        lam = 4.0 * (s + 1)
        for o in range(n_orient):
            theta = np.pi * o / n_orient
            ax = np.arange(-(ksize // 2), ksize // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            xr = xx * np.cos(theta) + yy * np.sin(theta)
            yr = -xx * np.sin(theta) + yy * np.cos(theta)
            g = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam)
            g -= g.mean()
            banks.append(g)
    return banks


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame @ np.array([0.299, 0.587, 0.114])
    return frame if frame.ndim == 2 else frame.mean(axis=-1)


class LowLevelEncoder(Encoder):
    name = "lowlevel"

    def __init__(self, n_orient: int = 4, n_scale: int = 2):
        self._bank = _gabor_bank(n_orient, n_scale)
        self.dim = len(self._bank) + 2  # gabor energies + luminance + motion

    def features(self, frames: np.ndarray, fps: float, tr: float) -> np.ndarray:
        from scipy.signal import fftconvolve

        F = frames.shape[0]
        feat = np.zeros((F, self.dim), dtype=float)
        prev_gray = None
        for f in range(F):
            gray = _to_gray(frames[f].astype(float))
            for i, g in enumerate(self._bank):
                resp = fftconvolve(gray, g, mode="same")
                feat[f, i] = np.sqrt(np.mean(resp ** 2))
            feat[f, len(self._bank)] = gray.mean()
            feat[f, len(self._bank) + 1] = 0.0 if prev_gray is None \
                else np.mean(np.abs(gray - prev_gray))
            prev_gray = gray
        return frames_to_tr(feat, fps, tr)
