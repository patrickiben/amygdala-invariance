# Real-data amygdala encoder-invariance analysis (ds002837, n=20)

These are the **actual analysis scripts and result logs** behind the measured amygdala verdict written
up in `../../REAL-RESULT_amygdala-encoding_ds002837.md` and visualized in
`../../amygdala-false-floor_verdict.html`. They were developed in an ephemeral scratchpad and preserved
here so the finding is reproducible. **Verdict: the amygdala's EmoNet-predictable signal is real but
NOT emotion-specific — a "false floor" (a generic object encoder recovers it equally).**

## Data provenance (not redistributed here)
- **BOLD:** OpenNeuro **ds002837** (Naturalistic Neuroimaging Database, Aliko et al. 2020),
  `derivatives/sub-{1..20}/func/*task-500daysofsummer*no_blur_no_censor.nii.gz` (MNI152, denoised,
  runs concatenated w/ scanning-pause timing correction, ~6 mm smoothing, TR=1 s, ~5470 vols).
  Fetched from `s3.amazonaws.com/openneuro.org/ds002837/…`. ~1.7 GB × 20.
- **Encoders:** EmoNet (Kragel et al. 2019) weights from OSF (`osf.io/amdju`), `conv_6` 4096-d;
  torchvision ViT-B/16 + ResNet50 (ImageNet); a low-level model (luminance/Gabor/motion); random-
  projection CNN-of-noise (dimensionality-matched null).
- **Stimulus:** the film was supplied by the user for on-device feature extraction only. Frames and
  extracted features are **deliberately not included here** (provenance) — only the derived, film-
  agnostic result numbers. The presentation was **PAL (25 fps)** vs the rip's 23.976 fps → features are
  resampled at **scale 1.042** to align to the BOLD timeline (validated: low-level→V1 and
  EmoNet→pooled-fusiform both peak at 1.042).
- **Annotations:** NNDb face annotation (`task-500daysofsummer_face-annotation`, already on the
  presented/BOLD timeline) → the 72%-face-presence baseline regressor.

## Pipeline (all scripts)
Per subject: extract region voxels (Harvard-Oxford masks), z-score; PAL-align + HRF-convolve + z-score +
PCA-300 the features; **per-subject PLS regression** (n_components=10, 5-fold contiguous-block CV);
R² = mean-over-voxels of held-out `1 − SS_res/SS_tot`. Group inference = one-sample t-test of the
per-subject values across 20 subjects (df=19). Variance partitions are differences of held-out R² on
identical targets (SS_tot cancels — valid despite negative absolute R²).

| script | what it does | key result |
|---|---|---|
| `deep_diag.py` | pooled-fusiform sanity: PCA-dim × scale sweep | deep features work when pooled; PCA-150 over-fit per-subject |
| `amyg_verdict.py` | pooled amygdala verdict + ISC | amygdala ISC 0.07 vs fusiform 0.45 → pooling washes amygdala |
| `pls20.py` | **main** 20-subject per-subject PLS + group t-tests | EmoNet-unique-over-ImageNet L +0.0014/R +0.0016 (p<.001) |
| `pls20_ctrl.py` | random-encoder + region-specificity controls | random null in amygdala; artifact in V1/fusiform |
| `shifted_null.py` | temporal null (misaligned EmoNet) | amygdala effect time-locked (misaligned→0) |
| `combined_controls.py` | face-in-baseline + 2nd-object (ResNet-vs-EmoNet over ViT) | **decisive**: ResNet=EmoNet in amygdala (p=.38/.17); EmoNet>ResNet in fusiform (p=.003) |
| `prep_hardening.py` | OpenCV rich-face regressor + encoder probe | faces in 23% of frames (frontal) |
| `mega_harden.py` | eroded(thr50) mask + hippocampus bleed control + purged/gap CV + rich-face + empirical null/TOST | verdict robust; hippocampus shows same generic pattern |

## To reproduce
1. `pip install` the pinned stack (Python 3.9): torch 2.2.2, torchvision 0.17.2, **numpy<2** (1.26.4),
   nilearn, nibabel, scikit-learn, scipy, opencv-python==4.10.0.84, antspyx (only if re-registering).
2. Download ds002837 derivatives for sub-1..20; get EmoNet weights from OSF; extract features from the
   presented film at 6 frames/TR, average per TR, cache as `feat_avg_cache.npz` (E/VT/RN) +
   `feat_cache.npz` (low-level LL, random RC).
3. Set `SP=<cache dir>` and run the scripts in the table order. Results append to `results/*.txt`.

Result logs in `results/` are the exact console output from the run of 2026-07-08.

## The last control — EXECUTED ✅ (verdict now airtight)
Ran a 2nd independent affect encoder (**Toisoul face-EmoNet**, `extract_toisoul.py` → `second_affect_swap.py`,
results in `results/second_affect_swap_result.txt`): it *also* fails to beat generic object vision in the amygdala
(≤ ResNet; R sig. worse), adds ~nothing over ImageNet, and the two affect encoders disagree in amygdala (r=−.3/−.5)
but agree in fusiform (r=+.94). ≥2-encoder invariance test satisfied → fusiform passes, amygdala false floor confirmed.
The how-to below stays valid for replication (HSEmotion / CLIP paths, `run_second_affect.sh`).

## How it was done / how to replicate
**Staged & runnable** (see `NEXT-second-affect-encoder-protocol.md` for the full rationale):
```bash
SP=<cache dir> FILM="/path/to/500 Days of Summer.mp4" bash scripts/run_second_affect.sh
```
- `scripts/extract_affect2.py` — extracts a 2nd affect encoder at 6 frames/TR (`ENCODER=hsemotion` default, or `clip`) → `feat_affect2.npz`.
- `scripts/second_affect_swap.py` — the 3-way swap (A2 vs ResNet vs EmoNet over ViT) + redundancy + across-subject agreement, with a self-interpreting verdict, for eroded amygdala L/R + fusiform control.

Predicted outcome: the amygdala false-floor verdict becomes airtight (a 2nd independent affect encoder also
fails to beat generic object vision in the amygdala); the escape-hatch outcome (a *face*-affect encoder beating
object vision where EmoNet's *scene*-affect didn't) would be a publishable surprise. Needs `pip install timm hsemotion`
(or `open_clip_torch`) — one-time weight download; the machine is networked, only a transient DNS flake blocked it in-session.
