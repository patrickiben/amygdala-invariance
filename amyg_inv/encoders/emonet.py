"""EmoNet feature extractor (amygdala target). IMPLEMENTED but UNTESTED (needs torch + weights).

Wiring: clone https://github.com/ecco-laboratory/emonet-pytorch (MIT), load the OSF weights from
Kragel et al. 2019 (Science Advances 5:eaaw4358), return the 4096-d penultimate (fc7-equivalent)
activation — matching Jang & Kragel's encoding-model input. Then aggregate to the TR grid
(hemodynamics.frames_to_tr) and HRF-convolve (hemodynamics.convolve_hrf) IDENTICALLY to every
other encoder.
"""
from __future__ import annotations

import numpy as np

from .base import Encoder
from ..modeling.hemodynamics import convolve_hrf, frames_to_tr


class EmoNetEncoder(Encoder):
    name = "emonet"
    dim = 4096

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch  # lazy heavy dep
            # from emonet_pytorch import EmoNet  # user supplies the package + weights
            raise NotImplementedError(
                "Provide the emonet-pytorch model + OSF weights: load the network, register a forward "
                "hook on the penultimate FC layer, and set self._model. Then delete this raise. "
                "See data/README.md. (Left as an explicit stub so real weights are a conscious step.)")

    def features(self, frames: np.ndarray, fps: float, tr: float) -> np.ndarray:
        self._load()
        import torch

        feats = []
        with torch.no_grad():
            for f in frames:                     # (H,W,3) in [0,1]
                x = torch.from_numpy(f).permute(2, 0, 1)[None].float().to(self.device)
                feats.append(self._penultimate(x).cpu().numpy().ravel())
        per_frame = np.stack(feats, axis=0)      # (F, 4096)
        return convolve_hrf(frames_to_tr(per_frame, fps, tr), tr)

    def _penultimate(self, x):  # pragma: no cover - requires real weights
        raise NotImplementedError
