# Re-analyses for specialist-venue submission

These three scripts implement the re-analyses a specialist referee (Imaging Neuroscience / the
encoding-model community) will expect, identified in the pre-submission referee pass. Each reuses
the committed pipeline verbatim (same feature caches, Harvard-Oxford masks, `prep()` = PAL 1.042 ->
HRF -> z -> PCA-300 -> z, 5-fold contiguous-block CV, variance-partition as difference of held-out
R2) and changes exactly one thing. **Run each, then confirm the reported verdict holds; report the
result in the paper or a supplement.** They are provided pre-run; the committed `results/` logs are
the original PLS run.

Set the environment first (see `../../REPRODUCE.md`): `export SP=$AMYG_DATA`, with the movie feature
caches (`feat_avg_cache.npz`, `feat_cache.npz`, `feat_affect2.npz`), the BOLD derivatives, and (for
#3) the film frames + EmoNet weights present.

| # | Script | Referee concern | Pass condition |
|---|--------|-----------------|----------------|
| 1 | `pls20_banded.py` | Executed path was per-participant **PLS**, not the **preregistered banded ridge** | Amygdala EmoNet-unique-over-ImageNet stays non-significant; fusiform positive control stays significant. Requires `pip install himalaya>=0.4`. |
| 2 | `noise_ceiling.py` | Fig 3 two-encoder "divergence" not separable from a noisier amygdala | Amygdala split-half reliabilities are clearly positive AND the convergence CI excludes the fusiform value -> divergence is real. If amygdala reliabilities ~0, reframe Fig 3 as supporting. |
| 3 | `emonet_preproc_sensitivity.py` | EmoNet preprocessing differs across arms (movie: raw; faces: ImageNet-norm) | Amygdala verdict + fusiform control hold under all three preprocessings (raw / imagenet / caffe). Adopt the variant that maximizes the fusiform control in **both** arms. |

## What to feed back into the manuscript

- **#1 banded ridge:** if the verdict holds (expected, since the finding already survives the
  random-encoder, temporal-null, purged-CV, and second-affect-encoder controls), either add a
  one-paragraph "preregistered banded-ridge confirmation" to Methods 4.3 / a supplement, or promote
  banded ridge to the primary estimator and update the Results numbers. Either resolves the
  prereg-vs-executed deviation disclosed in Methods 4.10.
- **#2 reliability:** add the per-region split-half reliabilities next to Figure 3 (Methods 4.8),
  or, if amygdala reliability is low, retitle Fig 3 as supporting evidence and lean on the
  criterion-A object-encoder-match result (which does not depend on cross-encoder reliability).
- **#3 preprocessing:** report the sensitivity as a one-line robustness statement in Methods 4.2,
  and re-run the face-morph `kragelEmoNet` extraction (`extract_morph_features.py` -> `sun_rsa.py`)
  under the chosen preprocessing so both arms are consistent.

## Author confirmations that also unblock the write-up
- Which second-affect encoder produced the committed `feat_affect2.npz` (Toisoul face-EmoNet per the
  README; confirm vs HSEmotion/CLIP). This locks the r=-.26/-.51/+.94 convergence numbers.
- Multiband factor and remaining acquisition parameters from the ds002837 / OSF BIDS sidecars (the
  manuscript now defers to the dataset descriptors rather than asserting a value).
