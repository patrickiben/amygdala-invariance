"""CLI: `python -m amyg_inv.cli smoke [region]` | `python -m amyg_inv.cli run [region]`.

  smoke : run the three synthetic ground-truth worlds through the real pipeline for the given
          region (or ALL regions if none named) and print the verdict for each. This is the
          runnable proof that the SAME decision engine works region-agnostically.
  run   : the real-data path -- raises until the adapters in amyg_inv/data are wired.

Regions are defined in preregistration.yaml (`amygdala`, `nucleus_accumbens`).
"""
from __future__ import annotations

import sys

import numpy as np

from .config import list_regions, load_prereg
from .data import synthetic
from .pipeline import Overrides, run


SMOKE = {"H1": "H1", "H0": "H0", "inapplicable": "INAPPLICABLE"}


def smoke(region: str | None = None) -> int:
    regions = [region] if region else list_regions()
    all_ok = True
    for reg in regions:
        cfg = load_prereg(reg)
        print(f"\n=== region: {reg}  (target={cfg.target}, subregions={cfg.subregions}, "
              f"prereg {cfg.sha256[:12]}…) ===")
        for scenario, expected in SMOKE.items():
            rng = np.random.default_rng(cfg.seed)
            feat, bold, tsnr = synthetic.make(scenario, cfg, rng, sizes=synthetic.FAST)
            res = run(feat, bold, tsnr, cfg, rng, overrides=Overrides(n_boot=120, n_folds=5))
            got = res.verdict.branch
            ok = got == expected
            all_ok = all_ok and ok
            print(f"[{'OK ' if ok else 'XX '}] {scenario:<12} expected={expected:<12} got={got}")
    print("\nSMOKE PASSED" if all_ok else "\nSMOKE FAILED")
    return 0 if all_ok else 1


def audit(argv) -> int:
    """`audit <region> --mask <nac_mask.nii.gz> --bold <b1.nii.gz> <b2.nii.gz> ...`

    Runs the NAc tSNR pre-flight gate on real preprocessed BOLD. PASS -> proceed to the live test;
    INAPPLICABLE -> stop (dataset measurement floor). Needs nibabel (+ nilearn if grids differ).
    """
    from .tsnr_audit import load_and_audit
    if not argv or "--mask" not in argv or "--bold" not in argv:
        print("usage: audit <region> --mask <mask.nii.gz> --bold <bold1.nii.gz> [bold2 ...]")
        return 2
    region = argv[0]
    mask = argv[argv.index("--mask") + 1]
    bold = argv[argv.index("--bold") + 1:]
    cfg = load_prereg(region)
    res = load_and_audit(bold, mask, cfg)
    print(f"NAc tSNR audit — region={region}, floor={res.floor:g}, "
          f"need {res.majority}/{len(res.per_subject)} subjects")
    for s in res.per_subject:
        print(f"  {s.subject}: median tSNR {s.median_tsnr:.1f}, "
              f"frac>=floor {s.frac_above_floor:.2f}, n_vox {s.n_vox}, pass={s.passed}")
    print(f"VERDICT: {res.verdict} — {res.reason}")
    return 0 if res.verdict == "PASS" else 1


def real_run(region: str | None = None) -> int:
    from .data.bold import load_bold  # noqa: F401
    reg = region or "the chosen region"
    raise SystemExit(
        f"Real-data path not available for {reg}: the BOLD / atlas / stimulus / encoder adapters "
        "in amyg_inv/data and amyg_inv/encoders are stubs. Wire them per data/README.md, then "
        "build feat/bold/tsnr from real inputs and call run(). For nucleus_accumbens, note the RED "
        "feasibility verdict: INAPPLICABLE is the expected outcome (see nucleus-accumbens-gap_feasibility.md)."
    )


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "smoke"
    region = argv[1] if len(argv) > 1 else None
    if cmd == "smoke":
        return smoke(region)
    if cmd == "audit":
        return audit(argv[1:])
    if cmd == "run":
        return real_run(region)
    print(f"unknown command: {cmd!r} (use 'smoke', 'audit', or 'run', optionally with a region)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
