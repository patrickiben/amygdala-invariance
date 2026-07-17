# Changelog

All notable changes to this analysis-code deposit. Versions correspond to Zenodo releases under
the concept DOI [10.5281/zenodo.21367398](https://doi.org/10.5281/zenodo.21367398).

## v1.2.0 — 2026-07-17

Adds the accompanying manuscript to the deposit (no code or result changes).

- New `manuscript/` folder: `MANUSCRIPT_amygdala-false-floor_v2.docx` (final, figures embedded) and its Markdown source.
- `.zenodo.json` / `CITATION.cff` descriptions updated — the deposit now contains code, result logs, figures, **and** the accompanying manuscript (was "only analysis code").
- The published v1.1.x Zenodo records are immutable, so the manuscript is delivered as this new version under the concept DOI.

## v1.1.1 — 2026-07-17

Deposit-hygiene patch. **No scientific results changed** — post-audit cleanup of the v1.1.0 archive.

- Metadata: `.zenodo.json` / `CITATION.cff` / `README.md` titles scoped to the human *left* amygdala; the stale "submitted to eLife" note removed.
- `validation/robustness_gate.py` now re-derives the **corrected** single-neuron TOST (margin +0.100, unit-boot 95% upper −0.0065, ~72% power) and the corrected positive-control p-values; it previously certified superseded v1.0.0 statistics and conflicted with `numbers_gate.py`.
- `validation/numbers_gate.py` docstring corrected to describe what it actually verifies (number-in-log co-occurrence, not manuscript coverage).
- Three pre-correction result logs sharing filenames with the corrected logs now carry a `SUPERSEDED` header; the README reproducibility table gained a v1.1.0 correction banner and a canonical corrected-log map.
- Scrubbed incidental machine-path / username lines from two committed logs; softened the Methods path claim to "executable code".
- Abstract: "significantly worse" scoped to object encoders only (no affect-vs-low-level significance test exists). Figure 1 caption ("at or near zero"; face-morph +0.010 n.s.; error bars labeled p-derived) and a `.11`→`.10` rounding fix.

## v1.1.0 — 2026-07-16

Corrected re-analysis of the same study; the overarching verdict (no emotion-encoder-specific,
image-computable code) is unchanged and, for the single-neuron arm, strengthened.

- **Corrected EmoNet preprocessing** to the canonical raw 0-255 RGB input of the reference
  implementation (`ecco-laboratory/emonet-pytorch`), replacing a prior per-frame `/255` (movie) and
  ImageNet-normalized (face-morph) input. Encoder feature caches regenerated.
- **Re-ran all four arms** under the corrected preprocessing (naturalistic movie fMRI, face-morph
  fMRI RSA, single-neuron RSA, audio); per-analysis result logs added under
  `real_data/scripts/reanalysis/results/`.
- **Movie arm updated (a prior claim was reversed):** the earlier "EmoNet does not beat a matched
  object encoder in the amygdala" (p=.38/.17) was a preprocessing artifact. Under corrected
  preprocessing EmoNet shows a tiny, near-floor significant increment (+0.0009) that a second affect
  encoder does not reproduce and that the two affect encoders do not reliably converge on. Prior
  v1.0.0 movie-arm numbers should not be cited.
- **Single-neuron arm strengthened** and made the anchor; the correction-robust result is the left
  amygdala (per-session faceEmoNet−ResNet −0.089, p<.001).
- **Title scoped** to the human *left* amygdala.
- Added reanalysis scripts: `banded_robust.py` (closed-form banded ridge), `single_neuron_tost.py`
  (corrected +0.100 equivalence margin), `univariate_byfdr.py` (BY-FDR recompute).
- Regenerated Figures 1–4; rewrote `validation/numbers_gate.py` to the corrected headline numbers
  (29/29 trace; `validation/presubmit.sh` passes).

## v1.0.0 — 2026-07-14

Initial deposit (uncorrected EmoNet preprocessing; superseded by v1.1.0).
