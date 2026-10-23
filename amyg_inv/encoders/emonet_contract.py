"""Corrected Kragel EmoNet contract, independent of torch and copyrighted stimuli."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import numpy as np

EMONET_COMMIT = "41ef0d4b8063cf35a808887b9719b3659f3d19fc"
EMONET_DIM = 4096


def emonet_pixels(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 NHWC -> float32 NCHW in raw 0–255 units; never divide by 255."""
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (227, 227, 3):
        raise ValueError("EmoNet requires a batch of uint8 RGB frames sized 227x227.")
    return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2), dtype=np.float32)


def checked_features(values, n_frames: int, dim: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (n_frames, dim) or not np.isfinite(values).all():
        raise ValueError(f"{name} features must be finite with shape {(n_frames, dim)}; got {values.shape}.")
    return values.astype(np.float32, copy=False)


def load_pinned_emonet_class(repo_path):
    """Load models.EmoNet from the reviewed upstream revision, avoiding `models` collisions."""
    repo = Path(repo_path).resolve()
    try:
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "models.py"], check=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("EMONET_REPO must be the pinned upstream git checkout; see real_data/REPRODUCE.md.") from exc
    if head != EMONET_COMMIT or clean.returncode != 0:
        raise ValueError(f"EmoNet requires unmodified models.py at upstream commit {EMONET_COMMIT}.")
    spec = importlib.util.spec_from_file_location("amygdala_pinned_emonet", repo / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EmoNet
