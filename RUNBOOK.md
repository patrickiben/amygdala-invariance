# Runbook — executing the two AMBER live tests

This is the recipe to actually *run* the two live tests on real fMRI. It is **not** runnable in
the scaffold's dev sandbox (no GPU, no ~20 GB of data, no stimulus pixels) — it targets a real
compute box. Nothing here fabricates a result; you run it and report what comes back.
`INAPPLICABLE` and `H0` are first-class, publishable outcomes — do **not** tune toward a positive.

## Verified live (2026-07-05, via OpenNeuro S3)
- **ds002837** (NNDb, *500 Days of Summer*): exists, CC0. — amygdala test.
- **ds007267** (Sugawara et al., food valuation): exists, CC0, **N=31** (sub-001…sub-031),
  BIDS 1.8.0. — NAc test.

## Prerequisites
- Linux + NVIDIA GPU (≥16 GB), ~200 GB free disk.
- Uncomment the real-data block in `requirements.txt`: `nibabel nilearn himalaya torch
  torchvision open_clip_torch transformers opencv-python`, plus `datalad`/`awscli` for downloads
  and `fmriprep` (Docker/Singularity) if derivatives aren't shipped.
- Two stimulus sets **you must supply** (neither is in the datasets):
  the exact *500 Days of Summer* cut NNDb used (copyright), and the Food-Pics images
  (register at food-pics.sbg.ac.at; join to trials via `events.tsv` `image_id`).

---

## Test A — `amygdala` on ds002837  (the live positive test)

1. **Data.** `datalad clone https://github.com/OpenNeuroDatasets/ds002837.git` (or `aws s3 sync
   s3://openneuro.org/ds002837 …`), the N=20 *500DoS* subjects. Prefer fMRIPrep derivatives in
   `MNI152NLin2009cAsym`; else run fMRIPrep.
2. **Mask** → implement `amyg_inv/data/atlas.py`: Amunts LB/CM/SF/AStr subnuclei (SPM Anatomy /
   Julich-Brain), resample to BOLD, QC voxel counts.
3. **Stimulus** → `amyg_inv/data/stimulus.py`: extract frames at the NNDb fps; validate
   frame→TR alignment (`alignment_error` under tolerance). **Make-or-break step.**
4. **Encoders** → `amyg_inv/encoders/{emonet,dinov2,clip}.py`: per-frame features → `frames_to_tr`
   → `convolve_hrf`, *identically* across EmoNet (4096-d penultimate; emonet-pytorch + OSF
   weights), DINOv2-large, CLIP ViT. `lowlevel` + `random` are already implemented.
5. **BOLD** → `amyg_inv/data/bold.py`: return `{subregion: (n_subj, T, n_vox)}` + tSNR.
6. **Run.** `python -m amyg_inv.cli run amygdala`.
7. **Outcome.** Reliability gate decides applicability (ds002837 is 1.5 T, but the amygdala is far
   larger/more forgiving than the NAc). Then `H1` = valence coding is encoder-invariant (survives
   DINOv2/CLIP) → strengthens Jang & Kragel; `H0` = EmoNet-specific → the encoder-artifact result.
   Both publishable.

---

## Test B — `nucleus_accumbens_3t` on ds007267  (GATE FIRST)

### 0. PRE-FLIGHT: the NAc tSNR audit (`mask.qc_gate: nac_tsnr_audit`) — do this BEFORE anything else
The multiband factor is **absent from the public sidecar** (only TR=0.8 s / 60 slices given), so
NAc SNR is unverified and this gate is the whole ballgame. Needs only BOLD + a NAc mask — **no
stimuli, no encoders, no GPU.**
- Download ≥3 subjects' func (or fMRIPrep derivatives in MNI 2009c).
- Place the CIT168 NAc mask (Pauli 2018, MNI 2009c) — or Harvard-Oxford "Accumbens" from
  `nilearn.datasets.fetch_atlas_harvard_oxford` as a turnkey stand-in.
- Compute per-voxel tSNR = mean(BOLD)/SD(BOLD) over time within the NAc mask; check against the
  pre-registered floor. **FAIL → report `INAPPLICABLE` and stop** (dataset measurement floor, not a
  verdict on value coding). **PASS → proceed.**

### 1–7 (only if the gate passes)
1. **Data.** ds007267, `task-food`, 3 sessions/subject (sub-016 ses-03 has 2 runs). Preprocess to
   MNI152NLin2009cAsym.
2. **Mask.** CIT168_NAc, MNI 2009c.
3. **Stimulus.** Food-Pics 568-image subset; join to trials on `events.tsv` `image_id`.
4. **Encoders.** Value heads: ImageReward + a food-appeal head + HPSv2 (≥2 must agree for `H1`).
   `imagenet_backbone_matched` control (same backbone, no value supervision) — value must beat it.
   `lowlevel` (incl. the `events.tsv` RGB + nutrition columns) + `random` adversaries. Align
   per-image features to each trial's 4 s `valuation` window.
5. **Value anchor.** Use `rating_value` (per-trial 1–4) as the subject's own within-subject value,
   head-to-head against the population encoder (mitigates "population appeal ≠ this subject's value").
6. **Control (mandatory).** Visual-category: within-food value gradients + low-level covariate
   regression (all-food design → no neutral baseline).
7. **Run.** `python -m amyg_inv.cli run nucleus_accumbens_3t`.
8. **Outcome.** `H1` = value coding encoder-invariant + rating-aligned; `H0` = value ≈
   food-category / low-level appearance; `INAPPLICABLE` = tSNR gate failed. All real.

---

## Compute estimate
fMRIPrep ~8–24 h/subject (skip if derivatives shipped) · encoder feature extraction minutes–hours
on GPU · banded ridge + subject bootstrap minutes/region on CPU. The **tSNR pre-flight (Test B
step 0)** is ~an hour and can be done on a laptop — run it first; it may end the NAc test cheaply.
