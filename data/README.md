# Wiring the real-data path

The pipeline runs today on synthetic data (`make smoke`). The real-data adapters below are now
**IMPLEMENTED but UNTESTED ON REAL DATA** — they carry `UNTESTED` banners, lazy-import their heavy
deps, and only become trustworthy once you run and debug them against real inputs on a compute box.
Two things still require *your* action: EmoNet's weights (a conscious stub), and the copyright-gated
stimuli. Start with the **NAc tSNR pre-flight audit** — it's laptop-runnable and needs no stimuli:

```bash
python -m amyg_inv.cli audit nucleus_accumbens_3t \
    --mask CIT168_NAc_MNI2009c.nii.gz \
    --bold sub-001_..._space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz sub-002_...nii.gz sub-003_...nii.gz
# PASS -> proceed to the full test;  INAPPLICABLE -> stop (MB6 signal floor).
```

Adapter status: `data/atlas.py` (Harvard-Oxford turnkey + labelled-atlas loader) · `data/bold.py`
(masking core unit-tested; confound-cleaning lazy) · `data/stimulus.py` (opencv frame/image loaders)
· `encoders/dinov2.py`, `encoders/clip.py` (real HF/open_clip loaders) — all implemented; `encoders/emonet.py`
left as an explicit stub so loading real OSF weights is a conscious step. Details below.

## 1. fMRI — OpenNeuro `ds002837` (Naturalistic Neuroimaging Database)
- Download `ds002837` (CC0). The **500 Days of Summer** run has **N=20** participants,
  **1.5 T** Siemens Avanto. Prefer fMRIPrep derivatives; otherwise preprocess consistently.
- Confound-clean identically across subjects (motion, aCompCor, high-pass); resample to a
  common space.
- Implement `amyg_inv/data/bold.py::load_bold` → `{subregion: (n_subjects, T, n_voxels)}` and
  a per-subregion tSNR dict.
- Refs: Aliko et al. 2020, *Scientific Data* 7:347 — https://www.nature.com/articles/s41597-020-00680-2 ·
  dataset https://openneuro.org/datasets/ds002837

## 2. Amygdala subregion atlas — Amunts et al. (2005)
- Fetch the Julich-Brain / SPM Anatomy cytoarchitectonic amygdala maps (LB, CM, SF, AStr) —
  the atlas Jang & Kragel used — threshold, resample to the BOLD reference.
- Implement `amyg_inv/data/atlas.py::load_subregion_masks` → `{subregion: voxel_indices}`.
- Report each subregion's voxel count + mean probability (small masks are a real limitation).

## 3. Film frames + alignment — the make-or-break step
- Source the exact **500 Days of Summer** cut used by NNDb (NOT redistributed in the dataset).
- Implement `amyg_inv/data/stimulus.py::extract_frames`; validate frame→TR alignment against
  the NNDb timing and keep `alignment_error` under a pre-set tolerance. Misalignment corrupts
  **every** encoder equally, so it is audited as a shared upstream nuisance.

## 4. Encoders — EmoNet (target) + DINOv2 + CLIP (low-level & random already implemented)
- **EmoNet** (`amyg_inv/data`… actually `amyg_inv/encoders/emonet.py`): clone
  https://github.com/ecco-laboratory/emonet-pytorch (MIT), load the OSF weights from
  Kragel et al. 2019 (*Science Advances* 5:eaaw4358), return the **4096-d penultimate** features.
- **DINOv2**: `facebook/dinov2-large` (Apache-2.0). **CLIP**: an OpenCLIP ViT.
- All encoders: per-frame features → `hemodynamics.frames_to_tr` → `hemodynamics.convolve_hrf`,
  applied identically. The low-level and random-CNN adversaries are already implemented and
  need no weights.

## Licensing note
`ecco-laboratory/AMOD` (the authors' analysis code) had **no LICENSE file** at the time of the
feasibility check — treat it as reference-only and re-implement, or clarify reuse rights with
the authors before forking. See `../amygdala-rank1_feasibility-and-protocol.md` §1.
