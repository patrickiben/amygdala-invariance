# Reproducing the corrected amygdala analyses

Current manuscript targets are the corrected logs in `real_data/scripts/reanalysis/results/`.
Original logs elsewhere are historical where a corrected log exists. The movie EmoNet−ResNet
increment is small **and significant**; the construct-invariance conclusion concerns failure to
recover that increment across independent affect encoders. See the top-level README and changelog.

The original movie extractor was lost. The preserved reconstruction now enforces the documented
corrected model/input contract, but has not been validated in a complete film/brain-data rerun.
This runbook distinguishes executable reconstruction steps from archived evidence and missing inputs.
No data, stimulus frames, feature arrays or weights ship with the code.

## 1. Environment and model

The real-data stack uses torch 2.2.2 / torchvision 0.17.2 and numpy<2 on Python 3.9–3.12.
The lightweight top-level requirements are a separate synthetic/test stack. Use a fresh cache directory
for a new extraction so old feature arrays cannot be mistaken for corrected ones. From the repo root:

```bash
python3.12 -m venv .venv-real
. .venv-real/bin/activate
python -m pip install -r real_data/requirements.txt
export AMYG_DATA="$HOME/amyg_data"
export SP="$AMYG_DATA"
export SUN2023_DIR="$AMYG_DATA/sun2023"
export AUDIO_DIR="$AMYG_DATA/audio"
export FILM="/path/to/your/legally-obtained/500DaysOfSummer.mp4"
export EMONET_REPO="$AMYG_DATA/emonet_pytorch"
export EMONET_WEIGHTS="$AMYG_DATA/emonet/emonet.pth"
mkdir -p "$AMYG_DATA/emonet" "$SUN2023_DIR" "$AUDIO_DIR"
git clone https://github.com/ecco-laboratory/emonet-pytorch.git "$EMONET_REPO"
git -C "$EMONET_REPO" checkout 41ef0d4b8063cf35a808887b9719b3659f3d19fc
curl -L https://osf.io/amdju/download -o "$EMONET_WEIGHTS"
git clone https://github.com/face-analysis/emonet.git "$AMYG_DATA/toisoul_emonet"
```

Kragel EmoNet is loaded from upstream `models.py` at the pinned commit with a strict state-dict load.
Its input is float32 RGB **0–255**, 227×227; its representation is the 4096-dimensional `conv_6`
activation. Toisoul face-EmoNet is a different model: 256×256 input divided by 255 and 256-dimensional
`avg_pool_2`. Do not interchange their preprocessing. The movie reconstruction records the Kragel
weights' SHA-256 and upstream revision in `$SP/feature_extraction_provenance.json`. Other provider
weights are still obtained by their original scripts; their historical revisions were not archived.
Install `ffmpeg` separately for film frame/audio extraction. The film is never redistributed.

## 2. Download and normalize input paths

Obtain the public data from their providers and arrange them to match the **code paths** below.
The OSF download directory labels are not the names used by the scripts; extract/move the appropriate
contents into this normalized layout. Do not merely keep the downloaded zip files in the cache root.

| Required location | Source/content |
|---|---|
| `$AMYG_DATA/deriv_sub-<k>_bold.nii.gz`, k=1…20 | OpenNeuro ds002837 derivatives: `derivatives/sub-*/func/*task-500daysofsummer*no_blur_no_censor.nii.gz` |
| `$SUN2023_DIR/stimuli/Fear-Happy/*.tif` | OSF 26RHZ `01 Stimuli` contents; face-morph images |
| `$SUN2023_DIR/fmri/sub1/face_emotion_para3/beta_0001.nii` and the corresponding other subjects/files | OSF 26RHZ `04 fMRI Data and Code` subject archives, extracted into `fmri/` |
| `$SUN2023_DIR/neuron/FiringRate_Amygdala.mat` and `FiringRate_MFC.mat` | OSF 26RHZ `05 Single-neuron` firing-rate files |
| `$AMYG_DATA/toisoul_emonet/pretrained/emonet_8.pth` | Original Toisoul model repository weights |
| `$AUDIO_DIR/film_16k.wav` | Mono 16 kHz PCM audio from the author-supplied film |

Public source links: [OpenNeuro ds002837](https://openneuro.org/datasets/ds002837),
[OSF 26RHZ](https://osf.io/26rhz/), [Kragel weights](https://osf.io/amdju/),
[Toisoul model](https://github.com/face-analysis/emonet).

## 3. Feature extraction and core analyses

Run from the repository root with the environment above active. These commands can take hours and
require substantial disk/RAM. They have not been executed as part of the lightweight code tests.
Movie/face/audio extraction scripts access pretrained models and atlas providers on first use.

```bash
# Movie: builds feat_avg_cache.npz (E/VT/RN) and feat_cache.npz (LL/RC), pre-HRF/pre-PAL.
python real_data/scripts/extract_movie_features.py
python real_data/scripts/pls20.py
# The preserved second-affect control uses Toisoul, not HSEmotion/CLIP.
ENCODER=toisoul bash real_data/scripts/run_second_affect.sh > "$SP/second_affect_console.txt"

# Face morph fMRI and single neurons: both use the corrected morph feature cache.
python real_data/sun2023/scripts/coverage_gate.py
python real_data/sun2023/scripts/extract_morph_features.py
python real_data/sun2023/scripts/sun_rsa.py
python real_data/sun2023/scripts/amyg_neuron_rsa.py > "$SUN2023_DIR/amyg_neuron_rsa_result.txt"
python real_data/sun2023/scripts/amyg_what.py > "$SUN2023_DIR/amyg_what_result.txt"

# Audio: audio_amyg.py reads $AMYG_DATA/audio/audio_features.npz.
# Keep AUDIO_DIR at the default $AMYG_DATA/audio for these two scripts to agree.
ffmpeg -i "$FILM" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$AUDIO_DIR/film_16k.wav"
python real_data/audio/extract_audio_features.py
python real_data/audio/audio_amyg.py
```

The movie extractor averages six frames/TR, followed by the reader's PAL stretch (1.042), HRF
convolution and PCA. The random-network seed is a reconstruction choice; the original weights were
not preserved. **Do not promise matching verdicts until a full-data comparison succeeds.**

Most core output logs are written to the data cache: `$SP/pls20_result.txt`,
`$SP/second_affect_swap_result.txt`, `$SUN2023_DIR/sun_rsa_result.txt`, and
`$AMYG_DATA/audio_amyg_result.txt`. Compare copies with the corrected archive, without overwriting it:

| Regenerated log | Archived target / current reported result |
|---|---|
| `pls20_result.txt` | `scripts/reanalysis/results/pls20_result.txt`: unique-over-ImageNet L +0.0022 / R +0.0017 |
| `second_affect_swap_result.txt` | `scripts/reanalysis/results/second_affect_swap_result.txt`: EmoNet−ResNet L +0.0009 (p=.002) / R +0.0006 (p=.003); fusiform +0.0105 |
| `sun_rsa_result.txt` | `scripts/reanalysis/results/sun_rsa_rerun_result.txt`: fusiform Kragel−ResNet +0.093 (p=.002); amygdala +0.010 (p=.716) |
| `amyg_neuron_rsa_result.txt` | `scripts/reanalysis/results/amyg_neuron_rsa_result.txt`: pooled −0.086, left −0.109; MFC +0.100 |
| `audio_amyg_result.txt` | `scripts/reanalysis/results/audio_rerun_result.txt`: STG over-vision +0.0189; left amygdala −0.0028 |

## 4. Additional analyses and explicit limitations

- `scripts/reanalysis/noise_ceiling.py` reads the movie caches/BOLD and writes `$SP/noise_ceiling_result.txt`.
  Corrected target: fusiform convergence +0.790, CI [+0.394,+0.965]; left −0.278,
  CI [−0.737,+0.100]. Amygdala CIs include zero; negative point values alone do not establish divergence.
- `scripts/reanalysis/banded_robust.py` writes `$SP/banded_robust_result.txt`. In the preserved run,
  the positive control collapsed too, so this is an over-regularization diagnostic, not confirmation.
- `scripts/reanalysis/strong_baseline_rsa.py` and `spec_curve.py` use the normalized single-neuron/morph
  inputs. They now write logs/figures under `$SUN2023_DIR` (override with `AMYG_OUTPUT`) and place the
  additional morph-feature cache under `$SUN2023_DIR/morph_strong_cache.npz`; preserved evidence is untouched.
  Compare with the same-named logs under `scripts/reanalysis/results/`. The anchored pooled+left contrast
  was <=0 in all 136 archived specifications; the right hemisphere did not show that invariance.
- `scripts/reanalysis/single_neuron_tost.py` is a **summary-statistic re-derivation**, not a new bootstrap
  from raw neurons. It writes to `$SP` (or `AMYG_OUTPUT`) and checks the archived bound/margin/power:
  unit-bootstrap upper −0.0065, session upper +0.0063, margin +0.100, descriptive power about 72%.
- `scripts/reanalysis/movie_strongbaseline.py` requires `$SP/feat_strong.npz` with `DINO` and `CLIP`
  arrays. **The historical generator for this movie cache was not preserved and no complete extraction
  recipe is supplied.** Its corrected archived log supports EmoNet−DINOv2 L +0.0021 / R +0.0015, but that
  additional analysis is not yet independently reproducible from this deposit alone.
- `figures/make_figures.py` renders **transcribed corrected summary values**, not freshly fitted data.
  Review regenerated logs before updating figure constants or manuscript numbers.

BLAS/device/weights differences can change low-order values. Compare signs, confidence bounds and
inference, but record discrepancies rather than tuning the pipeline to recover a desired verdict.
The original v1.0 logs are not acceptance targets for corrected code.

## 5. Lightweight verification

In the top-level synthetic/test environment, without film/brain data:

```bash
python -m pytest tests/ -q
python -m amyg_inv.cli smoke
python validation/numbers_gate.py
python validation/robustness_gate.py
bash validation/presubmit.sh
```

The presubmission script additionally resolves citations over the network and scans the bundled
manuscript. CI runs the synthetic/contract tests, smoke cases, path guard, log-trace and summary gates,
and citation resolver. None of these checks replaces a complete real-data reproduction.
