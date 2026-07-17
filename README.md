# No emotion-specific image-computable code in the human left amygdala — analysis code

This repository is the permanent code deposit for the manuscript **"No emotion-specific
image-computable code in the human left amygdala"** (Patrick Iben, Independent Researcher,
Saint Petersburg FL; ORCID [0009-0005-2443-7973](https://orcid.org/0009-0005-2443-7973)).
It contains the analysis code, the result logs that are the evidence for every headline
number, the four publication figures, and the accompanying manuscript itself (final DOCX plus
Markdown source, under [`manuscript/`](manuscript/)). It does **not** contain any brain data,
stimuli, or model weights (see **Provenance / Data** below).

## What the code does

The manuscript applies a **representation-invariance ("false-floor") test** to the claim
that the human amygdala carries an emotion-specific, image-computable code. The logic: a
genuine emotion-*specific* neural code should be recovered *better* by an emotion-supervised
encoder than by a matched generic object/vision encoder, and that advantage should be
**invariant across >=2 paradigmatically different emotion encoders**. If instead a generic
encoder recovers the same signal equally well — and independent affect encoders disagree —
the apparent "emotion code" is a **false floor**: real, reliable, stimulus-locked signal that
is nonetheless generic visual/auditory structure, not graded emotion.

The test is run across **four independent modalities**, each with a sensory-cortex region as a
live positive control:

| Arm | Data | Method | Amygdala result | Positive control |
|---|---|---|---|---|
| Movie fMRI | ds002837, n=20 | per-subject PLS encoding, dR2 | EmoNet-unique-over-ImageNet ~ 0; ResNet matches EmoNet (n.s.) | fusiform: EmoNet > object (sig.) |
| Face morphs | OSF 26RHZ fMRI | RSA, drho | affect encoders not above ResNet | fusiform / V1: affect encoders above object |
| Single neurons | OSF 26RHZ firing rates | RSA + **TOST** bounded null | MFC-sized affect-specificity **ruled out** (equivalence) | MFC |
| Audio | film audio, n=20 | wav2vec2 affect encoding, dR2 over vision | audio-affect not unique in amygdala | STG / Heschl |

Two independent affect encoders (EmoNet scene-affect, Toisoul face-EmoNet) **converge in
fusiform (r ~ +0.94) but diverge in the amygdala (r ~ -0.3/-0.5)** — the >=2-encoder invariance
test the false-floor argument requires. A positive account of what the amygdala *does* encode
(a population bias toward stimulus intensity/salience, not graded emotion) is included.

## Provenance / Data — READ THIS

**This repository contains ONLY analysis code, derived result logs (`*.txt`), and the four
figures. It contains no input data of any kind** — no fMRI volumes, no single-neuron firing
rates, no film frames, no extracted features, and no model weights. All inputs are **third-party
and publicly available from their original sources** and are **not redistributed here**:

- **Movie fMRI:** OpenNeuro **ds002837** (Naturalistic Neuroimaging Database, Aliko et al. 2020), CC0.
- **Face-morph fMRI + single-neuron firing rates + morph stimuli:** OSF project **26RHZ** (Sun et al. 2023).
- **Encoders / weights:** EmoNet (Kragel et al. 2019, OSF `amdju`); Toisoul face-EmoNet
  (github.com/face-analysis/emonet); torchvision ViT-B/16 + ResNet50 (ImageNet); audeering
  wav2vec2 dimensional-emotion model (HuggingFace).

**The film stimulus (*500 Days of Summer*) is copyrighted and is NEVER included, in any form.**
It was author-supplied to the extraction machine for on-device feature/audio extraction only;
frames and extracted features are, by design, not stored in or distributed with this repository.
Because bit-exact reproduction depends on that copyrighted stimulus and on the third-party public
data, the **reproducibility target is the verdicts** (signs, significance, equivalence bounds),
not 4th-decimal values (BLAS/GPU/MPS differences also perturb low-order digits).

## Repository layout

```
real_data/                     the manuscript analyses (the four arms above)
  README.md, REPRODUCE.md      arm-level provenance + step-by-step reproduction
  requirements.txt             pinned real-analysis stack, Python 3.9-3.12 (numpy<2)
  scripts/                     movie-fMRI arm scripts (pls20.py, pls20_ctrl.py, ...)
  results/*.txt                movie-arm result logs  <-- EVIDENCE, included
  sun2023/scripts/, results/   face-morph fMRI + single-neuron arm (+ result logs)
  audio/                       audio arm scripts + result log
amyg_inv/                      shared package (banded ridge, variance partition, noise
                               ceiling, reliability gate, decision rule) imported by the
                               real_data scripts (located by walking parents) — REQUIRED
figures/                       Fig1-Fig4 (.png/.pdf) + make_figures.py
validation/, real_data/validation/  path/citation/number/robustness gate scripts
preregistration.yaml           frozen region-agnostic decision rule
requirements.txt               top-level lightweight synthetic/test stack (runs on 3.9-3.14)
README.md, RUNBOOK.md          this file; scaffold/live-test runbook
```

Note: the top-level `results/` directory holds a regenerable **synthetic-scaffold** run and is
excluded from this deposit; the manuscript evidence lives in `real_data/**/results/`, which is
**included**. (In `.gitignore`, ignore only `/results/` with a leading slash so nested
`real_data/results/` and `real_data/sun2023/results/` are not swept out.)

## How to reproduce

All scripts are path-parameterized (no machine-specific paths) and read their data roots from
environment variables. See `real_data/REPRODUCE.md` for the authoritative, most detailed version.

**1. Environment**
```bash
# torch 2.2.2 supports Python 3.9-3.12 ONLY; do NOT use 3.13/3.14 for the real-data stack.
python3.9 -m venv .venv && . .venv/bin/activate
pip install -r real_data/requirements.txt          # numpy stays <2 (mandatory)
export AMYG_DATA="$HOME/amyg_data"                  # cache root (features, BOLD, weights, logs)
export SP="$AMYG_DATA"                              # movie-arm scripts read SP
export SUN2023_DIR="$AMYG_DATA/sun2023"             # Sun 2023 fMRI + single-neuron + stimuli
export AUDIO_DIR="$AMYG_DATA/audio"                 # extracted film audio + features
export FILM="/path/to/your/legally-obtained/500DaysOfSummer.mp4"   # copyright-gated, author-supplied
mkdir -p "$AMYG_DATA" "$SUN2023_DIR" "$AUDIO_DIR"
```
(The top-level `requirements.txt` is only the lightweight synthetic/test stack and is the sole
part that runs on Python 3.13/3.14; the real analysis uses `real_data/requirements.txt`.)

**2. Fetch the public data + weights (once)** — none of this ships with the repo:
- **ds002837** derivatives, all 20 subjects, from OpenNeuro S3 (task `500daysofsummer`,
  `*no_blur_no_censor.nii.gz`) -> save each as `$AMYG_DATA/deriv_sub-<k>_bold.nii.gz`
  (`pls20.py` globs `$SP/deriv_sub-*_bold.nii.gz`).
- **OSF 26RHZ** (Sun 2023) -> `$SUN2023_DIR`: `01 Stimuli/`, `04 fMRI Data and Code/`,
  `05 Single-neuron/FiringRate_Amygdala.mat` + `FiringRate_MFC.mat`.
- **EmoNet** weights (OSF `amdju`) -> `$AMYG_DATA/emonet/`; **Toisoul face-EmoNet** clone
  (`git clone github.com/face-analysis/emonet`) -> `$AMYG_DATA/toisoul_emonet/`; torchvision
  ViT-B/16 + ResNet50 and audeering wav2vec2 auto-download on first use.

**Movie feature caches.** Every movie-arm script `np.load()`s two caches: `$SP/feat_avg_cache.npz` (keys `E`=EmoNet fc7 4096-d, `VT`=ViT-B/16 768-d, `RN`=ResNet-50 2048-d, all TR-averaged) and `$SP/feat_cache.npz` (keys `LL`=low-level Gabor/luminance/motion, `RC`=random-weight AlexNet). Build them with `python real_data/scripts/extract_movie_features.py` (reads `FILM`+`SP`; frames the film at 6 fps, runs each encoder, TR-averages, writes both `.npz`). **Note:** this extractor is a *reference re-implementation* -- the original was not preserved, so it reproduces the pipeline and verdicts but not necessarily the committed 4th-decimal values; see the provenance header in that script and validate its `pls20.py` output against `real_data/results/pls20_result.txt`. The sun2023 morph feature cache is built by `extract_morph_features.py`; the audio cache by `extract_audio_features.py`.

**3. Run each arm.** Each script writes its log into the DATA CACHE root it reads
(`$SP` / `$SUN2023_DIR` / `$AUDIO_DIR`), **not** into the repo's `results/` folders — you copy
the regenerated logs back and compare (see step 4). Scripts marked `>` only `print()`, so the
redirection shown is required to capture their log.
```bash
# Movie fMRI arm (all read SP, write into $SP)
python real_data/scripts/extract_movie_features.py   # build feat_avg_cache.npz + feat_cache.npz (reference re-impl; validate vs committed log)
python real_data/scripts/pls20.py               # -> $SP/pls20_result.txt   (headline EmoNet-unique effect)
python real_data/scripts/pls20_ctrl.py          # -> $SP/pls20_ctrl_result.txt
python real_data/scripts/shifted_null.py        # -> $SP/shifted_null_result.txt
python real_data/scripts/combined_controls.py   # -> $SP/combined_result.txt
python real_data/scripts/mega_harden.py         # -> $SP/mega_result.txt
# 2nd-affect control: runs extract_toisoul.py|extract_affect2.py -> $SP/feat_affect2.npz,
# then second_affect_swap.py which PRINTS -> redirect to capture the log:
bash real_data/scripts/run_second_affect.sh > "$SP/second_affect_swap_result.txt"

# Face-morph fMRI + single-neuron arm (read SUN2023_DIR)
python real_data/sun2023/scripts/coverage_gate.py
python real_data/sun2023/scripts/extract_morph_features.py     # builds its own morph feature cache
python real_data/sun2023/scripts/sun_rsa.py     # -> $SUN2023_DIR/sun_rsa_result.txt
python real_data/sun2023/scripts/sun_univar.py  # -> $SUN2023_DIR/sun_univar_result.txt
python real_data/sun2023/scripts/amyg_neuron_rsa.py > "$SUN2023_DIR/amyg_neuron_rsa_result.txt"
python real_data/sun2023/scripts/compute_tost.py    > "$SUN2023_DIR/amyg_neuron_tost.txt"
python real_data/sun2023/scripts/amyg_what.py       > "$SUN2023_DIR/amyg_what_result.txt"

# Audio arm
python real_data/audio/extract_audio_features.py   # -> $AUDIO_DIR/audio_features.npz
python real_data/audio/audio_amyg.py               # -> $AMYG_DATA/audio_amyg_result.txt

# Figures (from the numbers above)
python figures/make_figures.py
```

**4. Collect & compare.** The freshly regenerated logs land in `$AMYG_DATA` / `$SUN2023_DIR` /
`$AUDIO_DIR`, NOT in the repo. Copy them next to the committed evidence logs in
`real_data/results/`, `real_data/sun2023/results/`, and `real_data/audio/`, and compare
**verdicts** (signs, significance, TOST equivalence bounds), not bit-identical 4th-decimal
values (BLAS/MPS differences are expected).

**5. Static gates (no data needed, run anytime).** `bash validation/presubmit.sh` runs the path
guard, numbers-gate, robustness re-derivation, citation resolver, and AI-artifact scrub; CI
(`.github/workflows/validate.yml`) runs the path guard + citation resolver on a cold runner.
`make test` / `make smoke` exercise the synthetic pipeline.

## Result log -> manuscript claim map

Each headline number in the manuscript maps to a committed result log (the exact console output
of the reference run). Verdicts, not 4th-decimals, are the reproducibility target.

> **v1.1.0 correction — read first.** The table immediately below maps the ORIGINAL v1.0.0 run. Under
> corrected canonical EmoNet preprocessing (v1.1.0) **all four arms were re-run**; the corrected logs
> live under `real_data/scripts/reanalysis/results/` and **supersede the same-named original logs**
> (each original now carries a `SUPERSEDED` header). The manuscript reports the **corrected** numbers.
> See `CHANGELOG.md`. Canonical v1.1.0 headline logs:
>
> | Corrected log (`real_data/scripts/reanalysis/results/`) | Headline |
> |---|---|
> | `amyg_neuron_rsa_result.txt` | Single-neuron affect−object −0.086 (boot-p=.033); left −0.109 (p=.003); MFC +0.100 |
> | `single_neuron_tost_result.txt` | Bounded null: margin +0.100, unit-boot 95% upper −0.0065, power ~72% |
> | `pls20_result.txt` | Movie EmoNet-unique over ImageNet +0.0022/+0.0017 |
> | `second_affect_swap_result.txt` | Movie EmoNet−ResNet +0.0009/+0.0006; fusiform +0.0105; convergence −0.28/−0.00 (amyg) vs +0.79 (fusiform) |
> | `sun_rsa_rerun_result.txt` | Face-morph fusiform kragel−ResNet +0.093 (p=.002); amygdala null |
> | `audio_rerun_result.txt` | Audio STG over-vision +0.0189 (p=.001); amygdala negative |
> | `banded_robust_result.txt` / `noise_ceiling_result.txt` / `univariate_byfdr_result.txt` | Banded-ridge over-regularization; Fig-3 reliabilities; univariate BY-FDR (both q=.06) |

| Result log (ORIGINAL v1.0.0 run — see correction note above) | Supports (claim) | Figure |
|---|---|---|
| `real_data/results/pls20_result.txt` | Amygdala BOLD is EmoNet-predictable, but EmoNet-unique-over-ImageNet is ~0 (+0.0014/+0.0016); fusiform positive control is large | Fig 1 |
| `real_data/results/pls20_ctrl_result.txt` | Random-encoder null holds in amygdala; region-specificity (artifact appears in V1/fusiform, not amygdala) | Fig 1 |
| `real_data/results/shifted_null_result.txt` | The small amygdala effect is temporally specific (misaligned EmoNet -> 0) | — |
| `real_data/results/combined_result.txt` | **Decisive:** ResNet matches EmoNet in amygdala (p=.38/.17) while EmoNet > ResNet in fusiform; not a face-in-baseline confound | Fig 1 |
| `real_data/results/mega_result.txt` | Robustness: eroded-mask, hippocampus-bleed control, purged/gapped CV, empirical random null + TOST artifact bound | — |
| `real_data/results/second_affect_swap_result.txt` | Two affect encoders diverge in amygdala (r -0.26/-0.51), converge in fusiform (r +0.94) -> airtight false floor | Fig 3 |
| `real_data/sun2023/results/sun_rsa_result.txt` | Face-morph RSA: affect encoders not above ResNet in amygdala; large in fusiform/V1 | Fig 1 |
| `real_data/sun2023/results/sun_univar_result.txt` | Univariate face-morph control | — |
| `real_data/sun2023/results/sun_profileRDM_result.txt` | Profile-RDM cross-check | — |
| `real_data/sun2023/results/amyg_neuron_rsa_result.txt` | Single-neuron RSA: amygdala affect - object ~ 0 vs MFC | Fig 1 |
| `real_data/sun2023/results/amyg_neuron_tost.txt` | **TOST bounded null:** MFC-sized amygdala affect-specificity ruled out (equivalence, ~82% power) | Fig 2 |
| `real_data/sun2023/results/amyg_what_result.txt` | Positive account: amygdala population bias toward stimulus intensity/salience, not graded emotion | Fig 4 |
| `real_data/audio/audio_amyg_result.txt` | Audio-affect not unique in amygdala over vision; STG/Heschl positive control | Fig 1 |

## Cite this

If you use this code, please cite both the manuscript and this archived deposit:

> Iben, P. (2026). *No emotion-specific image-computable code in the human left amygdala.*
> Manuscript in preparation.

> Iben, P. (2026). *No emotion-specific image-computable code in the human left amygdala —
> analysis code* [Software]. Zenodo. https://doi.org/10.5281/zenodo.21367398 (concept DOI — always resolves to the latest version; each release also has its own version DOI on Zenodo).

See `CITATION.cff` for machine-readable metadata.

## License

Code is released under the **MIT License** (see `LICENSE`). The license covers **this
repository's code only**. It does **not** grant any rights to the third-party datasets, model
weights, or the copyrighted film stimulus, which remain under their own respective terms.
