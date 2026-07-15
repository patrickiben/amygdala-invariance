"""NAc tSNR pre-flight audit (protocol `mask.qc_gate: nac_tsnr_audit`).

The FIRST gate of the NAc live test on ds007267: the multiband factor is MB6 (confirmed from the
live bold.json), aggressive enough that NAc SNR is a genuine coin-flip. This gate decides
applicability BEFORE any stimulus/encoder work. It needs only preprocessed BOLD + a NAc mask —
no film, no food images, no GPU.

  tSNR(voxel) = mean_t(signal) / SD_t(signal)

PASS iff, across a majority of audited subjects, the NAc median tSNR clears the pre-registered
floor AND a minimum fraction of NAc voxels clear it. FAIL -> report **INAPPLICABLE** (a statement
about the dataset's measurement floor, not a verdict on value coding) and stop.

The core computation is pure numpy + nibabel (light). Real use needs preprocessed BOLD in the mask's
space (e.g. fMRIPrep -> MNI152NLin2009cAsym) + the CIT168 NAc mask; if grids differ, resample the
mask with nilearn (`resample_to_img`, nearest) — see `load_and_audit`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def compute_tsnr(bold_4d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(X, Y, Z, T) -> (X, Y, Z) voxelwise temporal SNR = mean_t / SD_t."""
    if bold_4d.ndim != 4:
        raise ValueError(f"expected 4D BOLD, got shape {bold_4d.shape}")
    mean_t = bold_4d.mean(axis=-1)
    sd_t = bold_4d.std(axis=-1)
    return mean_t / (sd_t + eps)


@dataclass
class SubjectAudit:
    subject: str
    n_vox: int
    median_tsnr: float
    frac_above_floor: float
    passed: bool


@dataclass
class AuditResult:
    verdict: str                       # "PASS" | "INAPPLICABLE"
    floor: float
    min_coverage_frac: float
    majority: int
    per_subject: List[SubjectAudit] = field(default_factory=list)
    reason: str = ""


def audit_subject(subject: str, tsnr_3d: np.ndarray, mask_3d: np.ndarray,
                  floor: float, min_coverage_frac: float) -> SubjectAudit:
    m = mask_3d.astype(bool)
    vals = tsnr_3d[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return SubjectAudit(subject, 0, float("nan"), 0.0, False)
    frac = float(np.mean(vals >= floor))
    med = float(np.median(vals))
    passed = (med >= floor) and (frac >= min_coverage_frac)
    return SubjectAudit(subject, int(vals.size), med, frac, passed)


def gate(per_subject: List[SubjectAudit], floor: float, min_coverage_frac: float,
         majority: int) -> AuditResult:
    n_pass = sum(s.passed for s in per_subject)
    if n_pass >= majority:
        verdict, reason = "PASS", (f"{n_pass}/{len(per_subject)} subjects cleared NAc tSNR floor "
                                   f"{floor:g} at >= {min_coverage_frac:.0%} coverage — proceed to the NAc test")
    else:
        verdict, reason = "INAPPLICABLE", (
            f"only {n_pass}/{len(per_subject)} subjects cleared the NAc tSNR floor {floor:g} "
            f"(need {majority}); NAc not adequately measured on this dataset (MB6 signal floor) — "
            f"report INAPPLICABLE, do NOT proceed to encoders")
    return AuditResult(verdict, floor, min_coverage_frac, majority, per_subject, reason)


def audit_from_arrays(subjects, bolds_4d, masks_3d, cfg,
                      min_coverage_frac: float = 0.5) -> AuditResult:
    """In-memory audit (used by tests and by `load_and_audit`).

    `masks_3d` may be one shared mask (same grid for all) or one per subject.
    """
    floor = float(cfg.d("reliability", "tsnr_floor"))
    majority = int(cfg.d("decision", "subregion_majority"))
    if not isinstance(masks_3d, (list, tuple)):
        masks_3d = [masks_3d] * len(subjects)
    per = [audit_subject(s, compute_tsnr(b), m, floor, min_coverage_frac)
           for s, b, m in zip(subjects, bolds_4d, masks_3d)]
    return gate(per, floor, min_coverage_frac, majority)


def load_and_audit(bold_paths: List[str], mask_path: str, cfg,
                   min_coverage_frac: float = 0.5,
                   subjects: Optional[List[str]] = None) -> AuditResult:
    """Real-data entry: load preprocessed BOLD NIfTIs + a NAc mask, resample if needed, audit.

    Requires nibabel; uses nilearn to resample the mask to each BOLD grid when affines/shapes
    differ (nearest-neighbour). BOLD must already be in the mask's space (e.g. fMRIPrep MNI 2009c).
    """
    import nibabel as nib

    mask_img = nib.load(mask_path)
    subjects = subjects or [f"sub{i+1:02d}" for i in range(len(bold_paths))]
    bolds, masks = [], []
    for p in bold_paths:
        img = nib.load(p)
        m = mask_img
        if img.shape[:3] != mask_img.shape[:3] or not np.allclose(img.affine, mask_img.affine):
            from nilearn.image import resample_to_img  # optional dep, only on grid mismatch
            m = resample_to_img(mask_img, nib.Nifti1Image(np.zeros(img.shape[:3]), img.affine),
                                interpolation="nearest")
        bolds.append(np.asarray(img.dataobj, dtype=np.float32))
        masks.append(np.asarray(m.dataobj) > 0.5)
    return audit_from_arrays(subjects, bolds, masks, cfg, min_coverage_frac)
