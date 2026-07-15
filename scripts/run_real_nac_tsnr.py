#!/usr/bin/env python
"""Real NAc tSNR pre-flight audit on ds007267 — the COARSE-but-runnable pipeline.

This is the script that produced REAL-RESULT_nac-tsnr-audit_ds007267.md (INAPPLICABLE, 0/3).
It runs WITHOUT fMRIPrep/Docker: it downloads raw BOLD from OpenNeuro S3, fetches the Harvard-Oxford
Accumbens mask (nilearn), affine-registers MNI->native EPI (antspyx), warps the mask, and computes
detrended NAc tSNR + the pre-registered gate.

COARSE-PIPELINE CAVEATS (see the result doc): raw BOLD (optional --motion-correct), affine-only
cross-contrast registration, Harvard-Oxford (not CIT168) mask. It UNDERESTIMATES tSNR (no motion
correction by default). This is a pre-flight indicator; the archival result needs fMRIPrep + CIT168
+ all subjects + the pre-registered floor.

Requires a Python with antspyx wheels (3.9-3.12; NOT 3.14):
    python3.9 -m venv .venv39 && . .venv39/bin/activate
    pip install antspyx nilearn nibabel numpy
    python scripts/run_real_nac_tsnr.py --subjects 001 002 003 [--motion-correct] [--floor 30]

Reuses the harness gate logic in amyg_inv.tsnr_audit.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request

import numpy as np

S3 = "https://s3.amazonaws.com/openneuro.org/ds007267"


def download(sid: str, ses: str, run: str, dest_dir: str) -> str:
    fn = f"sub-{sid}_{ses}_task-food_{run}_bold.nii.gz"
    out = os.path.join(dest_dir, f"ds007267_{fn}")
    if not os.path.exists(out):
        url = f"{S3}/sub-{sid}/{ses}/func/{fn}"
        print(f"  downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, out)
    return out


def detrended_tsnr(X: np.ndarray) -> np.ndarray:
    mean_t = X.mean(-1)
    T = X.shape[-1]
    t = np.arange(T, dtype=np.float32); t -= t.mean(); tt = float((t * t).sum())
    Xd = X - ((X * t).sum(-1) / tt)[..., None] * t
    return mean_t / (Xd.std(-1) + 1e-8)


def nac_tsnr_values(bold_path: str, tmp: str, moving, mask_mni, motion_correct: bool) -> np.ndarray:
    import ants
    import nibabel as nib

    aff = nib.load(bold_path).affine
    if motion_correct:
        mc = ants.motion_correction(ants.image_read(bold_path), type_of_transform="Rigid")
        X = mc["motion_corrected"].numpy().astype(np.float32)
    else:
        X = np.asarray(nib.load(bold_path).dataobj, dtype=np.float32)
    mean_t = X.mean(-1)
    nib.save(nib.Nifti1Image(mean_t, aff), f"{tmp}/mean.nii.gz")
    fixed = ants.image_read(f"{tmp}/mean.nii.gz")
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="AffineFast")
    m = ants.apply_transforms(fixed=fixed, moving=mask_mni, transformlist=reg["fwdtransforms"],
                              interpolator="nearestNeighbor").numpy() > 0.5
    v = detrended_tsnr(X)[m]
    return v[np.isfinite(v)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["001", "002", "003"])
    ap.add_argument("--session", default="ses-01")
    ap.add_argument("--run", default="run-01")
    ap.add_argument("--floor", type=float, default=None, help="tSNR floor (default: prereg reliability.tsnr_floor)")
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--motion-correct", action="store_true")
    ap.add_argument("--workdir", default=tempfile.gettempdir())
    args = ap.parse_args()

    import ants
    import nibabel as nib
    from nilearn import datasets

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from amyg_inv.config import load_prereg
    from amyg_inv.tsnr_audit import audit_subject, gate

    cfg = load_prereg("nucleus_accumbens_3t")
    floor = args.floor if args.floor is not None else float(cfg.d("reliability", "tsnr_floor"))
    majority = int(cfg.d("decision", "subregion_majority"))
    tmp = tempfile.mkdtemp()

    mni = datasets.load_mni152_template(resolution=2); nib.save(mni, f"{tmp}/mni.nii.gz")
    moving = ants.image_read(f"{tmp}/mni.nii.gz")
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    adata = np.asarray(ho["maps"].dataobj); labels = list(ho["labels"])
    li = [labels.index("Left Accumbens"), labels.index("Right Accumbens")]
    nib.save(nib.Nifti1Image(np.isin(adata, li).astype(np.uint8), ho["maps"].affine), f"{tmp}/nacc.nii.gz")
    mask_mni = ants.image_read(f"{tmp}/nacc.nii.gz")

    print(f"NAc tSNR audit — ds007267 | floor={floor:g} coverage={args.coverage:.0%} need {majority}/{len(args.subjects)}"
          f" | motion_correct={args.motion_correct}\n")
    per = []
    for sid in args.subjects:
        path = download(sid, args.session, args.run, args.workdir)
        vals = nac_tsnr_values(path, tmp, moving, mask_mni, args.motion_correct)
        # 3D tSNR + mask surrogate for the shared gate helper: pass tSNR as a 1-voxel-row volume
        tsnr3d = vals.reshape(-1, 1, 1)
        mask3d = np.ones_like(tsnr3d, dtype=bool)
        sa = audit_subject(f"sub-{sid}", tsnr3d, mask3d, floor, args.coverage)
        per.append(sa)
        print(f"  sub-{sid}: NAc median tSNR {sa.median_tsnr:5.1f} | frac>=floor {sa.frac_above_floor:4.0%}"
              f" | n_vox {sa.n_vox:3d} | {'PASS' if sa.passed else 'FAIL'}")
    res = gate(per, floor, args.coverage, majority)
    print(f"\nVERDICT: {res.verdict} — {res.reason}")
    return 0 if res.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
