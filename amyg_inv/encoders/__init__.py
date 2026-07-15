"""Frozen feature extractors. All encoders return (T, D) features on the TR grid.

Implemented here: `lowlevel` (Gabor/GIST + luminance + motion) and `random` (architecture-
matched random-weight CNN) — the two pre-committed break-case adversaries, which need no
trained weights. `emonet`, `dinov2`, `clip` are stubs (weights/deps required); see each file.
"""
from .base import Encoder
