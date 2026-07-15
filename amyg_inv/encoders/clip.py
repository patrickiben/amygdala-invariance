"""CLIP ViT image-encoder features (language-aligned objective). IMPLEMENTED but UNTESTED
(needs torch + open_clip_torch).

Per-frame image embedding -> TR grid -> HRF-convolve, identically to every other encoder. A third
distinct objective (image-text contrastive), so the encoder panel spans the objective axis Conwell
et al. 2024 show drives brain-predictivity equivalence. Also usable as the NAc/OFC backbone-matched
control head-off (same CLIP backbone, no value supervision).
"""
from __future__ import annotations

import numpy as np

from .base import Encoder
from ..modeling.hemodynamics import convolve_hrf, frames_to_tr


class CLIPEncoder(Encoder):
    name = "clip"
    dim = 768  # ViT-L/14 image embedding

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cpu"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = self._preprocess = None

    def _load(self):
        if self._model is None:
            import open_clip  # lazy heavy dep
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained)
            self._model = self._model.eval().to(self.device)

    def features(self, frames: np.ndarray, fps: float, tr: float) -> np.ndarray:
        from PIL import Image
        import torch

        self._load()
        feats = []
        with torch.no_grad():
            for f in frames:
                img = Image.fromarray((f * 255).astype("uint8"))
                x = self._preprocess(img).unsqueeze(0).to(self.device)
                emb = self._model.encode_image(x)
                feats.append(emb.cpu().numpy().ravel())
        per_frame = np.stack(feats, axis=0)
        return convolve_hrf(frames_to_tr(per_frame, fps, tr), tr)
