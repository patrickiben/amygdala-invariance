"""Region-agnostic synthetic worlds with KNOWN ground truth, to validate the decision rule.

Encoder features are built ONCE (they do not depend on the BOLD, as frozen encoders do not),
by ROLE derived from the region config:
  * target      (cfg.target)        : carries a 'meaningful' latent strongly (emotion for the
                                       amygdala; value for the NAc).
  * highlevel   (cfg.highlevel)     : also carries the meaningful latent (they should 'recover').
  * generic     (in all_encoders,   : partially carry the meaningful latent (a competent generic
                 not target/HL/adv)   encoder like DINOv2 or a backbone-matched control).
  * lowlevel    (cfg.lowlevel)      : low-level latent only, no meaningful latent  (break-case).
  * random      (other adversaries) : mostly low-level + noise                     (break-case).

Only WHAT DRIVES THE BOLD changes across scenarios:
  H1           -> meaningful latent -> only meaningful-carrying encoders predict -> target-unique
                  large, high-level encoders recover, break-case silent.
  H0           -> low-level latent  -> every encoder predicts, the low-level encoder best of all
                  -> break-case triggers / encoders indistinguishable (degeneracy).
  inapplicable -> subject-independent noise -> ISC ~ 0, low tSNR -> reliability gate fails.

The point: run these through the REAL pipeline for ANY region config and confirm the frozen
rule returns H1 / H0 / INAPPLICABLE. Works unchanged for amygdala (4 subregions, EmoNet target)
and nucleus_accumbens (2 subregions, ImageReward target).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Sizes:
    T: int = 240
    n_subjects: int = 8
    v: int = 24          # voxels per subregion
    k_emotion: int = 3   # 'meaningful' latent dims
    k_lowlevel: int = 3


FAST = Sizes(T=180, n_subjects=6, v=16)

# per-role (meaningful_weight, lowlevel_weight, dim, noise)
_ROLE_PARAMS = {
    "target":    (1.0, 0.30, 40, 0.5),
    "highlevel": (0.9, 0.30, 40, 0.5),
    "generic":   (0.7, 0.30, 32, 0.5),
    "lowlevel":  (0.0, 1.00, 16, 0.4),
    "random":    (0.0, 0.50, 16, 0.8),
}


def _role_of(enc: str, cfg) -> str:
    if enc == cfg.target:
        return "target"
    if enc in cfg.highlevel:
        return "highlevel"
    if enc == cfg.lowlevel:
        return "lowlevel"
    if enc in cfg.adversaries:      # adversary that is not the low-level one
        return "random"
    return "generic"


def _ar1(rng, T, k, rho=0.4):
    x = rng.standard_normal((T, k))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * x[t]
    return x


def _mix(latents, n_out, rng, scale):
    return latents @ (rng.standard_normal((latents.shape[1], n_out)) * scale)


def make(scenario: str, cfg, rng: np.random.Generator, sizes: Sizes = Sizes()):
    S = sizes
    s_meaning = _ar1(rng, S.T, S.k_emotion)
    s_lowlevel = _ar1(rng, S.T, S.k_lowlevel)

    feat: Dict[str, np.ndarray] = {}
    for enc in cfg.all_encoders:
        emo_w, low_w, dim, noise = _ROLE_PARAMS[_role_of(enc, cfg)]
        f = np.zeros((S.T, dim))
        if emo_w > 0:
            f += _mix(s_meaning, dim, rng, emo_w)
        f += _mix(s_lowlevel, dim, rng, low_w)
        f += rng.standard_normal((S.T, dim)) * noise
        feat[enc] = f

    bold: Dict[str, np.ndarray] = {}
    tsnr: Dict[str, float] = {}
    for sub in cfg.subregions:
        if scenario == "inapplicable":
            bold[sub] = rng.standard_normal((S.n_subjects, S.T, S.v))   # no shared signal
            tsnr[sub] = 12.0
            continue
        driver = s_meaning if scenario == "H1" else s_lowlevel
        signal = _mix(driver, S.v, rng, scale=1.4)
        signal = signal / (signal.std(axis=0, keepdims=True) + 1e-8) * 1.3
        bold[sub] = signal[None, :, :] + rng.standard_normal((S.n_subjects, S.T, S.v))
        tsnr[sub] = 55.0

    return feat, bold, tsnr
