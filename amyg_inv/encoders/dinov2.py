"""DINOv2 feature extractor (self-supervised ViT). IMPLEMENTED but UNTESTED (needs torch + transformers).

facebook/dinov2-large (Apache-2.0). Per-frame CLS embedding -> TR grid -> HRF-convolve, identically
to every other encoder. Differs from EmoNet in BOTH architecture (ViT vs CNN) and objective
(self-supervised vs emotion-supervised) — a paradigmatic contrast.
"""
from __future__ import annotations

import numpy as np

from .base import Encoder
from ..modeling.hemodynamics import convolve_hrf, frames_to_tr


class DINOv2Encoder(Encoder):
    name = "dinov2"
    dim = 1024

    def __init__(self, model_id: str = "facebook/dinov2-large", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = self._proc = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModel  # lazy heavy dep
            self._proc = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    def features(self, frames: np.ndarray, fps: float, tr: float) -> np.ndarray:
        import torch

        self._load()
        feats = []
        with torch.no_grad():
            for f in frames:
                inp = self._proc(images=(f * 255).astype("uint8"), return_tensors="pt").to(self.device)
                out = self._model(**inp)
                cls = out.last_hidden_state[:, 0]           # CLS token
                feats.append(cls.cpu().numpy().ravel())
        per_frame = np.stack(feats, axis=0)
        return convolve_hrf(frames_to_tr(per_frame, fps, tr), tr)
