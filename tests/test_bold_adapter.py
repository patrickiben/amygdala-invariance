"""The BOLD-adapter masking core, verified on synthetic NIfTIs (the piece that needs no torch/film).

Confound-cleaning + BIDS discovery + encoder feature extraction are implemented but untested on real
data (they need nibabel/nilearn/torch + real inputs); the mask->timeseries->stack->tSNR spine is
verifiable here and is what the pipeline consumes.
"""
import numpy as np

from amyg_inv.data.bold import load_bold, load_region_bold, mask_timeseries


def test_mask_timeseries_shape_and_values():
    bold = np.arange(2 * 2 * 2 * 5, dtype=float).reshape(2, 2, 2, 5)
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    ts = mask_timeseries(bold, mask)
    assert ts.shape == (5, 2)                      # (T, V=2 masked voxels)
    assert np.allclose(ts[:, 0], bold[0, 0, 0, :])  # first voxel's timeseries preserved


def test_load_region_and_load_bold_roundtrip(tmp_path):
    import pytest
    nib = pytest.importorskip("nibabel")
    affine = np.eye(4)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True                     # 8-voxel ROI
    paths = []
    rng = np.random.default_rng(0)
    for s in range(3):
        b = (500 + 10 * rng.standard_normal((4, 4, 4, 40))).astype(np.float32)
        p = tmp_path / f"sub-{s+1}_bold.nii.gz"
        nib.save(nib.Nifti1Image(b, affine), str(p))
        paths.append(str(p))

    region = load_region_bold(paths, mask, high_pass=False)   # no cleaning -> no nilearn needed
    assert region.shape == (3, 40, 8)                          # (N, T, V)

    bold, tsnr = load_bold(paths, {"NAc_L": mask}, high_pass=False)
    assert bold["NAc_L"].shape == (3, 40, 8)
    assert 40 < tsnr["NAc_L"] < 62                            # ~500/10 = 50
