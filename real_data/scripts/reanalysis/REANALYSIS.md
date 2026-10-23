# Corrected reanalyses

The current archived evidence is in `results/`; see [../../REPRODUCE.md](../../REPRODUCE.md) for the
normalized inputs, run order and output locations. This directory also retains scripts from the
historical pre-correction workflow. Historical “expected null” comments are not current acceptance
criteria: the corrected movie EmoNet−ResNet increment is small **and significant**.

| Script | Current interpretation / reproduction status |
|---|---|
| `noise_ceiling.py` | Fusiform convergence +0.79 is reliable; amygdala CIs include 0. Run with the corrected movie caches. |
| `banded_robust.py` | Preserved banded-ridge run collapses the fusiform control too; diagnose over-regularization rather than claim confirmation. |
| `strong_baseline_rsa.py` | Single-neuron contrast remains negative with stronger generic/affect encoders; needs raw morph/neuron inputs. |
| `spec_curve.py` | All 136 archived pooled+left specifications have nonpositive contrasts; right-hemisphere results vary. |
| `single_neuron_tost.py` | Re-derives a non-superiority summary from committed statistics, not a fresh raw-data bootstrap. |
| `movie_strongbaseline.py` | Needs an unshipped `feat_strong.npz`; its historical extraction script was not preserved. Archived result exists, complete regeneration remains a gap. |
| `emonet_preproc_sensitivity.py`, `pls20_banded.py` | Historical exploratory scripts; they are not the corrected primary extraction/run path. |

The corrected primary Kragel extractor uses the pinned upstream model, raw 0–255 RGB and 4096-dimensional
`conv_6` features. Do not select preprocessing to maximize a control or tune toward a desired verdict.
Fresh logs should be compared against copies of corrected archived targets, never overwrite the evidence.
