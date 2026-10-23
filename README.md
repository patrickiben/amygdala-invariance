# No emotion-specific image-computable code in the human left amygdala — analysis code

This repository is the permanent code deposit for the manuscript **"No emotion-specific
image-computable code in the human left amygdala"** (Patrick Iben, Independent Researcher,
Saint Petersburg FL; ORCID [0009-0005-2443-7973](https://orcid.org/0009-0005-2443-7973)).
It contains the analysis code, the result logs that are the evidence for every headline
number, the four publication figures, and the accompanying manuscript itself (final DOCX plus
Markdown source, under [`manuscript/`](manuscript/)). It does **not** contain any brain data,
stimuli, or model weights (see **Provenance / Data** below).

## Educational companion

The [plain-language companion](education/COMPANION.md) explains the corrected manuscript’s
methods, findings and limits, with worked examples and exercises with answers. See its
[source and review notes](education/README.md) for the exact version and PDF hash.

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
| Movie fMRI | ds002837, n=20 | per-subject PLS encoding, dR2 | Small significant EmoNet−ResNet increment: L +0.0009 (p=.002), R +0.0006 (p=.003); not recovered by a second affect encoder | fusiform: +0.0105 |
| Face morphs | OSF 26RHZ fMRI | RSA, drho | affect encoders not above ResNet | fusiform / V1: affect encoders above object |
| Single neurons | OSF 26RHZ firing rates | RSA + **TOST** bounded null | MFC-sized affect-specificity **ruled out** (equivalence) | MFC |
| Audio | film audio, n=20 | wav2vec2 affect encoding, dR2 over vision | audio-affect not unique in amygdala | STG / Heschl |

Under corrected preprocessing, two independent affect encoders (EmoNet scene-affect, Toisoul
face-EmoNet) **converge reliably in fusiform (r=+0.79, 95% CI [+0.394,+0.965])**. Amygdala
agreement is L −0.28 / R −0.00, with CIs including zero. The small movie EmoNet increment also
survives DINOv2 (+0.0021/+0.0015), so the limitation is **construct invariance across affect
encoders**, not absence of a single-encoder increment or proof of reliable negative convergence.
A positive account of what the amygdala *does* encode
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
requirements.txt               top-level lightweight synthetic/test stack (tested on Python 3.12)
README.md, RUNBOOK.md          this file; scaffold/live-test runbook
```

Note: the top-level `results/` directory holds a regenerable **synthetic-scaffold** run and is
excluded from this deposit; the manuscript evidence lives in `real_data/**/results/`, which is
**included**. (In `.gitignore`, ignore only `/results/` with a leading slash so nested
`real_data/results/` and `real_data/sun2023/results/` are not swept out.)

## How to reproduce

Start with the lightweight, data-free checks (Python 3.12 was tested locally):

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m amyg_inv.cli smoke
python validation/numbers_gate.py
python validation/robustness_gate.py
```

These test synthetic behavior, input/model contracts and traceability to committed logs. They do not
regenerate real-data results. CI runs these checks plus the source-path and citation gates.

For the four real-data arms, use **[real_data/REPRODUCE.md](real_data/REPRODUCE.md)**. That runbook
specifies the compatible torch/numpy stack, the pinned Kragel EmoNet revision, normalized data paths,
Toisoul second-affect control and corrected log targets. No input data or weights ship here.
The reconstructed movie extractor now enforces raw 0–255 RGB and 4096-dimensional `conv_6` features;
full-data agreement with archived results remains unverified because the original extractor was lost.
The additional movie DINOv2/CLIP strong-feature cache also has no preserved extraction script.
Do not interpret successful synthetic/static checks as a complete real-data reproduction.

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
| `real_data/results/second_affect_swap_result.txt` | Historical v1.0 convergence values; superseded by the corrected two-encoder/reliability logs above | Historical Fig 3 |
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
> Preprint. https://doi.org/10.5281/zenodo.21422754.

> Iben, P. (2026). *No emotion-specific image-computable code in the human left amygdala —
> analysis code* [Software]. Zenodo. https://doi.org/10.5281/zenodo.21367398 (concept DOI — always resolves to the latest version; each release also has its own version DOI on Zenodo).

Current archived versions are [manuscript v3](https://doi.org/10.5281/zenodo.22314186) and
[software v1.3.1](https://doi.org/10.5281/zenodo.21420761). The concept DOIs above resolve to the latest
published versions; the GitHub reproducibility fixes listed as Unreleased are not a new Zenodo release.

See `CITATION.cff` for machine-readable metadata.

[Manuscript → code → demo map](https://github.com/patrickiben/patrickiben/blob/main/RESEARCH_MAP.md)
connects this study with its manuscript and the other research projects.

## License

Code is released under the **MIT License** (see `LICENSE`). The license covers **this
repository's code only**. It does **not** grant any rights to the third-party datasets, model
weights, or the copyrighted film stimulus, which remain under their own respective terms.
