"""Subregion masks — IMPLEMENTED, but UNTESTED ON REAL ATLASES (needs nilearn + atlas files).

Two paths:
  * turnkey (nilearn Harvard-Oxford): whole amygdala (L/R) and accumbens (L/R). Coarser than the
    protocol's amygdala subnuclei, but zero-setup for a first pass and for the NAc.
  * file-based (multi-label NIfTI): the protocol atlases — Amunts amygdala subnuclei (LB/CM/SF/AStr,
    Julich-Brain/SPM Anatomy) and CIT168 NAc — supplied as a labelled volume + a {label_value: name}
    map. This is the accurate path; you provide the atlas file.

Returns `{subregion_name: (X, Y, Z) bool ndarray}` in the reference image's grid. The core
resampling/threshold logic is standard nilearn; verify on your atlas before trusting voxel counts.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# --- turnkey Harvard-Oxford label sets (nilearn) ---------------------------------
_HO_LABELS = {
    "amygdala": ["Left Amygdala", "Right Amygdala"],
    "nucleus_accumbens": ["Left Accumbens", "Right Accumbens"],
}


def load_harvard_oxford(region: str, reference_img, thr: str = "sub-maxprob-thr25-2mm"):
    """Turnkey L/R masks for whole amygdala or accumbens, resampled to `reference_img`."""
    from nilearn import datasets, image  # lazy: heavy optional dep

    ho = datasets.fetch_atlas_harvard_oxford(thr)
    atlas_img, labels = ho["maps"], ho["labels"]
    want = _HO_LABELS.get(region)
    if want is None:
        raise ValueError(f"no Harvard-Oxford turnkey set for region {region!r}; use load_labelled_atlas")
    resampled = image.resample_to_img(atlas_img, reference_img, interpolation="nearest")
    data = np.asarray(resampled.dataobj)
    out: Dict[str, np.ndarray] = {}
    for name in want:
        idx = labels.index(name)
        out[name.replace(" ", "_")] = (data == idx)
    return out


def load_labelled_atlas(atlas_path: str, label_map: Dict[int, str], reference_img):
    """Accurate path: a labelled NIfTI (e.g. Amunts subnuclei, CIT168 NAc) + {label_value: name}.

    label_map example (Amunts): {1: "LB", 2: "CM", 3: "SF", 4: "AStr"}.
    """
    import nibabel as nib
    from nilearn import image

    atlas_img = nib.load(atlas_path)
    resampled = image.resample_to_img(atlas_img, reference_img, interpolation="nearest")
    data = np.asarray(resampled.dataobj)
    return {name: (data == val) for val, name in label_map.items()}


def load_subregion_masks(reference_img, region: str,
                         atlas_path: Optional[str] = None,
                         label_map: Optional[Dict[int, str]] = None) -> Dict[str, np.ndarray]:
    """Dispatch: a supplied labelled atlas (accurate) else the Harvard-Oxford turnkey (coarse)."""
    if atlas_path and label_map:
        return load_labelled_atlas(atlas_path, label_map, reference_img)
    return load_harvard_oxford(region, reference_img)
