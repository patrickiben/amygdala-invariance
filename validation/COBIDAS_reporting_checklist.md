# G6 · Reporting-standard checklist (OHBM COBIDAS, + single-unit addenda)

Neuroimaging reporting standard (Nichols et al. 2017, *Nat Neurosci* — COBIDAS) matched to this paper's fMRI arms,
with a single-unit addendum for the neuron arm. ✓ reported | ⚠ partial/see note | n/a not applicable.
Locations reference `MANUSCRIPT_amygdala-false-floor_v2.md`.

## 1. Experimental design
- ✓ Design type (naturalistic movie; event-related face-morph; passive+task) — §2.1, §2.2, §4.1
- ✓ Stimuli, timing, task — §4.1 (film 25 fps PAL; 7-level morph, parametric fear/ambiguity)
- ✓ Number of participants / units, inclusion — §2.1 n=20; §2.2 n=19; §2.3 442 units/22 sessions; §4.1
- ✓ Independence of datasets stated — §5 (movie is the only fully-independent arm; morph fMRI + single-neuron share the stimulus)

## 2. Acquisition
- ✓ Scanner field strength, TR, sequence — §4.1 (ds002837 1.5T TR=1s; Sun 3T TR=2s), deferred to dataset descriptors for full parameters
- ✓ Provenance of derivatives (public repositories) — §4.1, §4.10 (OpenNeuro ds002837 no_blur derivative; OSF 26rhz SPM12 first-levels)

## 3. Preprocessing
- ✓ Software + versions — §4.10 (nilearn, SPM12 upstream, pinned stack)
- ✓ Spatial normalization / smoothing state — §4.1 (movie = unsmoothed no_blur; morph = SPM sw* normalized+smoothed)
- ✓ **Stimulus-model preprocessing fully specified** — §4.2 (canonical emonet-pytorch raw 0-255 RGB @227px; the correction from a prior 1/255 is stated explicitly, with the reversed movie result flagged — a COBIDAS-plus transparency item)
- ✓ Confounds / nuisance (realignment params, HRF, drift) — §4.1, §4.3

## 4. Statistical modeling & inference
- ✓ Model per arm (PLS variance partition; RSA Spearman on 1−Pearson RDMs; per-unit OLS) — §4.3, §4.4, §4.6
- ✓ Estimator + CV scheme — §4.3 (PLS n_comp=10, PCA-300, 5-fold contiguous-block); banded-ridge deviation justified (over-regularizes) — §4.3, §4.10
- ✓ Inference framework + multiple-comparison correction — §4.9 (BY-FDR over the confirmatory positive-control family; confirmatory amygdala contrasts interpreted uncorrected, stated explicitly; exploratory family flagged)
- ✓ Primary vs secondary inference declared — §4.5 (per-session primary; unit-bootstrap descriptive)
- ✓ Effect sizes + uncertainty intervals (not just p) — throughout (RSA Δρ, ΔR², bootstrap 95% CIs, TOST bound)
- ✓ Equivalence / bounded-null method — §4.5 (one-sided non-superiority; margin = MFC-sized affect effect)
- ✓ ROI definition + atlas + threshold — §4.3 (Harvard-Oxford maxprob-thr25; thr50 sensitivity in controls)

## 5. Results reporting
- ✓ All ROIs/contrasts reported incl. nulls — §2.1–§2.5, Figs 1–4
- ✓ Positive controls reported for every arm — fusiform / MFC / STG
- ✓ Robustness / adversarial battery — §2.1 (eroded mask, purged CV, face-presence baseline, random-encoder null); **strong-baseline representation-invariance red-team** (G4, `strong_baseline_result.txt`)
- ✓ Reliability of the key figure disclosed — §2.2 / Fig 3 (raw bootstrap convergence CI; disattenuated estimate withheld as unstable)

## 6. Data, code, and reproducibility sharing
- ✓ Code publicly deposited — GitHub `patrickiben/amygdala-invariance`; Zenodo concept DOI 10.5281/zenodo.21367398 (latest v1.2.0 = 21403906, includes the manuscript)
- ✓ Result logs = the archived evidence — §4.10; `real_data/scripts/reanalysis/results/`
- ✓ Raw data not redistributed (third-party/public), pointers given — §4.10, Data & Code Availability
- ✓ Deterministic pre-submission gate — `validation/presubmit.sh`, `EVIDENCE_LEDGER.md`

## 7. Single-unit addendum (neuron arm)
- ✓ Recording provenance + counts (442 amygdala, 268 MFC; 22 sessions; epilepsy patients) — §2.3, §4.1
- ✓ Spike measure + baseline correction — §4.5, §4.6 (countAll − countBaseline; fear-happy trials)
- ✓ Pseudo-population construction + nesting caveat — §4.5 (units-by-7-level; per-session test respects nesting)
- ✓ Ethics / consent (secondary analysis of public de-identified data) — Ethics section

## Human sign-off
Author confirms the above reflects the manuscript as submitted: __________________  (date __________)
