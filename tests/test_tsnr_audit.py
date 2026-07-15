"""The NAc tSNR pre-flight gate, verified on synthetic NIfTIs.

Cannot run real fMRI here, but the audit LOGIC (tSNR computation + the PASS/INAPPLICABLE gate)
is verifiable on synthetic data with known ground-truth SNR — exactly how the rest of the harness
is validated.
"""
import numpy as np
import pytest

from amyg_inv.config import load_prereg
from amyg_inv.tsnr_audit import (audit_from_arrays, compute_tsnr, load_and_audit)

CFG = load_prereg("nucleus_accumbens_3t")   # floor=30 (reliability.tsnr_floor), majority=2


def _synthetic_bold(mean_level, noise_sd, shape=(6, 6, 6, 60), seed=0):
    rng = np.random.default_rng(seed)
    return mean_level + noise_sd * rng.standard_normal(shape)


def test_compute_tsnr_matches_definition():
    b = _synthetic_bold(mean_level=500.0, noise_sd=10.0, seed=1)
    tsnr = compute_tsnr(b)
    # tSNR ~ 500/10 = 50, within sampling error over 60 timepoints
    assert 40 < np.median(tsnr) < 62


def test_gate_passes_high_tsnr():
    mask = np.ones((6, 6, 6), dtype=bool)
    bolds = [_synthetic_bold(500.0, 10.0, seed=s) for s in range(3)]   # tSNR ~50 >> floor 30
    res = audit_from_arrays([f"s{i}" for i in range(3)], bolds, mask, CFG)
    assert res.verdict == "PASS", res.reason
    assert all(s.passed for s in res.per_subject)


def test_gate_returns_inapplicable_low_tsnr():
    mask = np.ones((6, 6, 6), dtype=bool)
    bolds = [_synthetic_bold(500.0, 40.0, seed=s) for s in range(3)]   # tSNR ~12.5 < floor 30
    res = audit_from_arrays([f"s{i}" for i in range(3)], bolds, mask, CFG)
    assert res.verdict == "INAPPLICABLE", res.reason


def test_load_and_audit_roundtrip(tmp_path):
    nib = pytest.importorskip("nibabel")
    affine = np.eye(4)
    mask = np.zeros((6, 6, 6), dtype=np.int16)
    mask[2:4, 2:4, 2:4] = 1                       # a small "NAc" ROI
    mask_path = tmp_path / "nac_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))

    bold_paths = []
    for s in range(2):
        b = _synthetic_bold(500.0, 10.0, seed=10 + s).astype(np.float32)  # tSNR ~50
        p = tmp_path / f"sub-{s+1:02d}_bold.nii.gz"
        nib.save(nib.Nifti1Image(b, affine), str(p))
        bold_paths.append(str(p))

    res = load_and_audit(bold_paths, str(mask_path), CFG)
    assert res.verdict == "PASS", res.reason
    assert res.per_subject[0].n_vox == 8          # the 2x2x2 ROI
