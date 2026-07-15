"""Film-frame extraction + frame->TR alignment.

`extract_frames` IS IMPLEMENTED (opencv) but UNTESTED here (needs the video + opencv). The
`alignment_error` math is pure and unit-tested via hemodynamics. Sourcing the exact video is the
copyright-gated make-or-break step (500 Days of Summer for ds002837; Food-Pics for ds007267).
"""
from __future__ import annotations

import numpy as np


def extract_frames(video_path: str, fps: float, max_frames: int | None = None) -> np.ndarray:
    """Decode a video to (F, H, W, 3) float in [0,1], sampling at `fps`. Requires opencv."""
    import cv2  # lazy heavy dep

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(int(round(src_fps / fps)), 1)
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(rgb)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return np.stack(frames, axis=0)


def load_images(image_paths, size: int = 224) -> np.ndarray:
    """Load a list of still images (e.g. Food-Pics) to (N, size, size, 3) in [0,1]. Requires opencv."""
    import cv2

    out = []
    for p in image_paths:
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size, size))
        out.append(im.astype(np.float32) / 255.0)
    return np.stack(out, axis=0)


def alignment_error(bold_onset_tr: float, film_onset_s: float, tr: float) -> float:
    """Frame->TR misalignment in TRs (a shared nuisance affecting all encoders equally)."""
    return abs(bold_onset_tr - film_onset_s / tr)
