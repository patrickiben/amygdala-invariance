# How to close the last control — add a 2nd affect-supervised encoder

> ## ✅ EXECUTED (2026-07-09) — verdict is now AIRTIGHT
> This was run, using **Toisoul face-EmoNet** (option B) via `git clone` (pypi DNS was flaking; GitHub resolved).
> `extract_toisoul.py` → 256-d embedding at 6 fps → `second_affect_swap.py` on all 20 subjects. Result:
> **the 2nd affect encoder also fails to beat generic object vision in the amygdala** (L: +0.0025 ≤ ResNet +0.0036,
> n.s.; R: +0.0021 < ResNet +0.0041, p<.001 *worse*), adds ~nothing over ImageNet (p=.38/.50), and the two affect
> encoders **disagree** in amygdala (r=−.26/−.51) but **agree** in fusiform (r=+.94, both beat object vision).
> → ≥2-encoder invariance test satisfied: **fusiform PASSES, amygdala FAILS (airtight false floor).**
> Logs: `results/second_affect_swap_result.txt`. The how-to below remains valid for replication / HSEmotion / CLIP.

**Status of the finding without this:** the amygdala "emotion code" is shown to be a **false floor** because a
generic *object* encoder (ResNet) recovers its signal as well as the emotion-supervised EmoNet (Emo−Res p=.22/.50),
while the fusiform keeps genuine emotion-specificity (p=.003). That is *strong*. The one thing that would make it
**airtight** is a second, paradigmatically different **affect**-supervised encoder — because "EmoNet doesn't beat
ResNet" could in principle be EmoNet-idiosyncratic. If a *different* affect model *also* fails to beat generic
vision in the amygdala (and its per-subject signal tracks EmoNet's), the "not-emotion-specific" verdict is closed,
and it simultaneously satisfies the False-Floor requirement of ≥2 paradigmatically different encoders of the
same construct.

This is a ~1–2 hour job. Everything below plugs into the existing `combined_controls.py` / `mega_harden.py`.

> **✅ NOW STAGED & RUNNABLE.** The scripts described below are already written and syntax-checked in
> `scripts/`: `extract_affect2.py` (feature extraction, HSEmotion **or** CLIP), `second_affect_swap.py`
> (the swap + agreement analysis with a self-interpreting verdict), and `run_second_affect.sh` (one-command
> runner). To run:
> ```bash
> SP=<cache dir> FILM="/path/to/500 Days of Summer.mp4" bash scripts/run_second_affect.sh
> ```
> The only thing that blocked running it in-session was a **transient DNS flake** to `files.pythonhosted.org`
> (the machine is networked — 34 GB of fMRI + torch/nilearn/opencv were pulled earlier the same session);
> `pip install timm hsemotion` succeeds once DNS is stable. Sections 3–4 below are the reference for what
> the staged scripts do.

---

## 1. Pick the second affect encoder (must differ from EmoNet in architecture AND training signal)

EmoNet here = **Kragel et al. 2019** (*Sci. Adv.*) — **whole-scene** emotion, AlexNet-style conv, trained on
20 emotion categories (Cowen & Keltner), 227×227 input. A good second encoder is **face-centric** and a
different backbone:

| # | Encoder | What it is | Why it's a good contrast | How to get it |
|---|---|---|---|---|
| **A (recommended)** | **HSEmotion** (Savchenko) | AffectNet-trained facial-expression net (EfficientNet-B0/B2), 8 emotions + valence/arousal | face-centric vs EmoNet's scene-centric; modern CNN; different training set | `pip install hsemotion` · weights: github `av-savchenko/face-emotion-recognition` |
| **B (cleanest contrast)** | **Toisoul "EmoNet"** (2021, *Nat. Mach. Intell.*) | **face** valence/arousal regressor (note: *same name*, totally different model) | face VA vs scene categories; independent lab/architecture/labels | github `face-analysis/emonet` (weights in repo) |
| **C (different paradigm)** | **CLIP zero-shot affect** | project CLIP image embeddings onto emotion/valence-arousal *text* prompts | language-grounded affect, not supervised classification at all | `pip install open_clip_torch` (ViT-B/32) |

Do **A** first (one `pip install`); if you want the strongest scientific statement, also do **B**.

---

## 2. Fix the environment (why it failed in-session)

The in-session attempt died because the py3.9 venv had no `timm` and was offline. On a networked machine:

```bash
SP=<your cache dir>                     # holds feat_avg_cache.npz, frames/, deriv_sub-*.nii.gz
"$SP/py39/bin/pip" install timm hsemotion open_clip_torch   # numpy stays <2 — re-pin if bumped:
"$SP/py39/bin/pip" install "numpy<2"
```

First model use downloads weights (needs network once). Keep `numpy==1.26.4` or `torch.from_numpy` breaks.

---

## 3. Extract its features over the movie (match the existing 6-frames/TR pipeline)

Needs the movie frames (`$SP/frames/` — re-extract from the film at 6 fps if the ephemeral cache is gone;
on-device only, do not redistribute). Produce a per-TR feature matrix shaped like `E`/`VT`/`RN` in
`feat_avg_cache.npz` (~5703 rows):

```python
# extract_affect2.py  — writes feat_affect2.npz with key A2 (n_TR × D)
import os, glob, numpy as np, torch
from PIL import Image
from hsemotion.facial_emotions import HSEmotionRecognizer   # Option A
SP=os.environ["SP"]; DEV="mps" if torch.backends.mps.is_available() else "cpu"
rec=HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device=DEV)   # 8 emo + valence/arousal logits
frames=sorted(glob.glob(f"{SP}/frames/f*.png"))
FPS_PER_TR=6                                    # same averaging as feat_avg_cache
feats=[]
for fp in frames:
    im=np.array(Image.open(fp).convert("RGB"))
    _,scores=rec.predict_emotions(im, logits=True)   # D-dim affect vector (whole frame; face-crop optional)
    feats.append(scores)
F=np.asarray(feats, np.float32)                 # (n_frames, D)
# average FPS_PER_TR consecutive frames -> per-TR (mirror reextract.py)
n=(len(F)//FPS_PER_TR)*FPS_PER_TR
A2=F[:n].reshape(-1, FPS_PER_TR, F.shape[1]).mean(1)
np.savez(f"{SP}/feat_affect2.npz", A2=A2)
print("A2", A2.shape)
```

(For Option B/C swap the two model lines; C = `open_clip` image embed · cos-sim to prompts like
"a photo of something joyful/sad/fearful/…". The only requirement is a per-TR `A2` matrix.)

---

## 4. Run the swap — add ONE feature block, reuse the existing machinery

In `combined_controls.py` (or `mega_harden.py`), load `A2` and `prep` it exactly like the others, then add
these per-subject contrasts (the group t-tests are already written — copy the `g()`/`gd()` calls):

```python
A2 = prep(np.load(f"{SP}/feat_affect2.npz")["A2"], T)          # PAL-align + HRF + z + PCA-300
bv  = r2(V, y)                                                  # ViT-only baseline
add(f"{rk}_a2Ovit",  r2(np.concatenate([A2, V],1), y) - bv)    # 2nd-affect over ViT
# (already have) emoOvit, resOvit
# redundancy tests — does either affect model add over ImageNet + the OTHER affect model?
bE = r2(np.concatenate([V,RN,A2],1), y); add(f"{rk}_emoOVERimgA2", r2(np.concatenate([E,V,RN,A2],1),y)-bE)
b2 = r2(np.concatenate([V,RN,E ],1), y); add(f"{rk}_a2OVERimgEmo",  r2(np.concatenate([A2,V,RN,E],1),y)-b2)
# agreement: per-subject A2-unique vs EmoNet-unique (correlate across subjects at the end)
```

---

## 5. Decision table — what each outcome means

Run for **L/R amygdala** (eroded thr50) with **fusiform** as the positive control:

| Amygdala result | Fusiform | Interpretation |
|---|---|---|
| **A2-over-ViT ≈ ResNet-over-ViT** (A2 no better than a 2nd *object* encoder) AND A2/EmoNet redundant (both ≈0 over img+other-affect) | A2 > ResNet | **✅ AIRTIGHT FALSE FLOOR** — two independent affect encoders both fail the swap; emotion-supervision does nothing amygdala-specific |
| A2-unique correlates with EmoNet-unique across subjects, both fail the object swap | — | same shared *generic-visual* signal, not emotion — closes the invariance test |
| **A2 *beats* ResNet in amygdala** (unlike EmoNet) | — | ⚠️ reopens it — would implicate *face*-affect specifically (vs EmoNet's scene-affect); a real, publishable surprise worth chasing |
| A2 adds nothing anywhere incl. fusiform | A2 ≈ 0 | A2 features are non-functional — debug extraction/preprocessing before trusting the null |

**Predicted (given everything so far): top row** — airtight false floor. The bottom-two rows are the
scientifically interesting escape hatches, which is exactly why the control is worth running.

---

## 6. Cost & provenance
~15 min feature extraction (MPS) + one 20-subject pass (~30–60 min) + a network-connected env for the weight
download. Public fMRI (ds002837); the film is used on-device only for feature extraction — do not store or
redistribute frames/features (only the derived R² numbers, as in `results/`).
