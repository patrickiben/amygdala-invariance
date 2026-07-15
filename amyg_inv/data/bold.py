"""BOLD adapter — masking core IS unit-tested (synthetic NIfTIs); confound-cleaning + BIDS
discovery are IMPLEMENTED but UNTESTED ON REAL DATA (need nibabel/nilearn + preprocessed inputs).

Given preprocessed BOLD (in the mask's space, e.g. fMRIPrep -> MNI152NLin2009cAsym) and per-subregion
masks, returns `({subregion: (n_subjects, T, n_voxels)}, {subregion: tSNR})` for the pipeline.

ALIGNMENT ASSUMPTION (real-data caveat): all subjects must share the stimulus timeline that the
encoder features are on. For ds002837 (one film) T is shared. For ds007267 (per-trial food images)
align by TRIAL via events.tsv `image_id`, not absolute time — resample each subject onto the shared
trial order BEFORE calling this. This adapter does the masking/tSNR; it does not do that alignment.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

def mask_timeseries(bold_4d: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    """(X,Y,Z,T) + (X,Y,Z) bool -> (T, V) voxel timeseries within the mask."""
    m = mask_3d.astype(bool)
    return bold_4d[m].T  # (V, T) -> (T, V)


def _clean(ts: np.ndarray, confounds: Optional[np.ndarray], high_pass: bool, tr: Optional[float]):
    """Confound regression + high-pass, via nilearn.signal.clean when available (real path)."""
    if confounds is None and not high_pass:
        return ts
    from nilearn.signal import clean  # lazy heavy dep
    return clean(ts, confounds=confounds, high_pass=(0.01 if high_pass else None),
                 t_r=tr, standardize=False, detrend=True)


def load_region_bold(bold_paths: List[str], mask_3d: np.ndarray,
                     confounds_paths: Optional[List[str]] = None,
                     high_pass: bool = True, tr: Optional[float] = None) -> np.ndarray:
    """Stack subjects for ONE subregion -> (n_subjects, T, V). Requires shared T (see alignment note)."""
    import nibabel as nib

    series = []
    for i, p in enumerate(bold_paths):
        arr = np.asarray(nib.load(p).dataobj, dtype=np.float32)
        ts = mask_timeseries(arr, mask_3d)
        conf = None
        if confounds_paths:
            conf = np.loadtxt(confounds_paths[i])
        series.append(_clean(ts, conf, high_pass, tr))
    T = min(s.shape[0] for s in series)
    return np.stack([s[:T] for s in series], axis=0)  # (N, T, V)


def load_bold(bold_paths: List[str], subregion_masks: Dict[str, np.ndarray],
              confounds_paths: Optional[List[str]] = None, high_pass: bool = True,
              tr: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Return ({subregion: (N,T,V)}, {subregion: tSNR}) for all subregions."""
    bold: Dict[str, np.ndarray] = {}
    tsnr: Dict[str, float] = {}
    for name, mask in subregion_masks.items():
        region = load_region_bold(bold_paths, mask, confounds_paths, high_pass, tr)
        bold[name] = region
        # summary = mean over subjects of each subject's median voxel tSNR (comparable to the
        # per-subject tsnr_audit gate; NOT the group-mean, which averages noise away and inflates it)
        if region.size:
            per_subj = [np.median(ts.mean(axis=0) / (ts.std(axis=0) + 1e-8)) for ts in region]
            tsnr[name] = float(np.mean(per_subj))
        else:
            tsnr[name] = float("nan")
    return bold, tsnr
