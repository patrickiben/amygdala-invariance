# Reproducing the amygdala false-floor analyses (G1 clean-clone)

All scripts are path-parameterized (no machine-specific paths; verified by `validation/check_paths.sh`).
They read their cache/data roots from environment variables, so a fresh clone runs after you set them and
fetch the public data.

## 1. Environment
```bash
# torch 2.2.2 supports Python 3.9-3.12 ONLY (numpy<2). Do NOT use 3.13/3.14 for the
# real-data stack; only the top-level lightweight requirements.txt runs on 3.13/3.14.
python3.9 -m venv .venv && . .venv/bin/activate
pip install -r real_data/requirements.txt          # numpy stays <2
export AMYG_DATA="$HOME/amyg_data"                  # cache root (features, BOLD, weights, logs)
export SUN2023_DIR="$AMYG_DATA/sun2023"             # Sun 2023 fMRI + single-neuron + stimuli
export AUDIO_DIR="$AMYG_DATA/audio"                 # extracted film audio + features
export SP="$AMYG_DATA"                              # movie-arm scripts read SP
export FILM="/path/to/your/legally-obtained/500DaysOfSummer.mp4"  # author-supplied; needed by
                                                    # extract_movie_features.py, extract_toisoul.py,
                                                    # extract_affect2.py, extract_audio_features.py; NEVER redistributed
mkdir -p "$AMYG_DATA" "$SUN2023_DIR" "$AUDIO_DIR"
```

## 2. Public data + weights (fetch once)
- **ds002837 (NNDb, 1.5T)** derivatives sub-1..20: `s3://openneuro.org/ds002837/derivatives/sub-*/func/*task-500daysofsummer*no_blur_no_censor.nii.gz` → `$AMYG_DATA/deriv_sub-<k>_bold.nii.gz`.
- **OSF 26RHZ** (Sun 2023): `01 Stimuli/`, `04 fMRI Data and Code/sub1-10.zip`+`sub11-19.zip`, `05 Single-neuron/FiringRate_Amygdala.mat`+`FiringRate_MFC.mat` → under `$SUN2023_DIR`.
- **EmoNet** weights (OSF amdju) → `$AMYG_DATA/emonet/`; **Toisoul face-EmoNet** `git clone github.com/face-analysis/emonet` → `$AMYG_DATA/toisoul_emonet/`.
- The **film** (audiovisual, for feature/audio extraction) is author-supplied; do not redistribute.

## 3. Feature caches (build first)
The movie scripts `np.load` two caches you must build once: `$SP/feat_avg_cache.npz` (keys `E`=EmoNet fc7
4096-d, `VT`=ViT-B/16 768-d, `RN`=ResNet-50 2048-d) and `$SP/feat_cache.npz` (keys `LL`=low-level
Gabor/luminance/motion, `RC`=random-weight AlexNet), each `[n_TR, dim]`, TR-averaged (6 frames/TR), **pre-HRF
and pre-PAL** (`pls20.prep()` applies the 1.042 stretch + HRF + PCA). Build them with
`python real_data/scripts/extract_movie_features.py`. **This extractor is a reference re-implementation** (the
original was not preserved): it reproduces the pipeline and verdicts, not necessarily the committed 4th-decimals
(see its provenance header). The faces morph cache is built by `extract_morph_features.py`; the audio cache by
`extract_audio_features.py`.

## 4. Run order (IMPORTANT: outputs land in the DATA CACHE, not the repo)
Each script writes its log into the cache root it reads (`$SP` / `$SUN2023_DIR` / `$AUDIO_DIR`), **not** into
`real_data/**/results/`. Copy the regenerated logs back next to the committed evidence to compare. Scripts marked
`>` only `print()`, so the redirection shown is required to capture their log.
```bash
# Movie arm (read/write $SP)
python real_data/scripts/extract_movie_features.py         # build the two caches (reference re-impl)
python real_data/scripts/pls20.py                          # -> $SP/pls20_result.txt (headline)
python real_data/scripts/pls20_ctrl.py                     # -> $SP/pls20_ctrl_result.txt
python real_data/scripts/shifted_null.py                   # -> $SP/shifted_null_result.txt
python real_data/scripts/combined_controls.py              # -> $SP/combined_result.txt
python real_data/scripts/mega_harden.py                    # -> $SP/mega_result.txt
FILM="$FILM" bash real_data/scripts/run_second_affect.sh > "$SP/second_affect_swap_result.txt"

# Faces + single-neuron arm (read $SUN2023_DIR); the neuron scripts only print():
python real_data/sun2023/scripts/coverage_gate.py
python real_data/sun2023/scripts/extract_morph_features.py
python real_data/sun2023/scripts/sun_rsa.py                # -> $SUN2023_DIR/sun_rsa_result.txt
python real_data/sun2023/scripts/sun_univar.py             # -> $SUN2023_DIR/sun_univar_result.txt
python real_data/sun2023/scripts/amyg_neuron_rsa.py  > "$SUN2023_DIR/amyg_neuron_rsa_result.txt"
python real_data/sun2023/scripts/compute_tost.py     > "$SUN2023_DIR/amyg_neuron_tost.txt"
python real_data/sun2023/scripts/amyg_what.py        > "$SUN2023_DIR/amyg_what_result.txt"

# Audio arm
python real_data/audio/extract_audio_features.py           # -> $AUDIO_DIR/audio_features.npz
python real_data/audio/audio_amyg.py                       # -> $AMYG_DATA/audio_amyg_result.txt
```
Note: `sun_profileRDM_result.txt` (committed) is a profile-RDM cross-check emitted by a variant of `sun_rsa.py`;
if you do not reproduce it, it does not bear on any headline claim.

## 5. Verify numbers
Every headline number maps to a `results/*.txt` file per the claim-to-artifact map in
`../../PROPOSED_validation-ledger.md` (G2) and the table in the top-level `README.md`. Note: BLAS/GPU/MPS
differences (and, for the movie arm, the reference-extractor caveat above) mean bit-identical reproduction is
not guaranteed; verdicts (signs, significance, equivalence bounds) are the reproducibility target, not the
4th-decimal value.
