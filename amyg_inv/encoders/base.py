from __future__ import annotations

import abc

import numpy as np


class Encoder(abc.ABC):
    """A frozen feature extractor: video frames -> (T, D) features on the TR grid.

    `frames` is (F, H, W, C) uint8/float in [0, 1]. Implementations must be deterministic and
    must NOT depend on the fMRI (they are the same across all subjects).
    """

    name: str
    dim: int

    @abc.abstractmethod
    def features(self, frames: np.ndarray, fps: float, tr: float) -> np.ndarray:
        ...
