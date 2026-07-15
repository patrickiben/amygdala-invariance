#!/usr/bin/env python
"""Extract a SECOND affect-supervised encoder's features over the movie, matched to the
existing feat_avg_cache pipeline (per-TR rows, averaged over FPS_PER_TR frames).

Writes  $SP/feat_affect2.npz  with key  A2  (n_TR x D).

Config (env):
  SP        cache dir (must hold frames/ or FILM must be set); output written here     [required]
  ENCODER   hsemotion | clip                                                            [hsemotion]
  FILM      path to the film (only needed to (re)extract 6-fps frames)                  [optional]
  FPS       frames per second to sample and then average per TR                          [6]
  DEVICE    cpu | mps                                                                    [cpu]

Rationale: EmoNet (Kragel 2019) = whole-scene emotion. This adds a paradigmatically
different affect encoder — HSEmotion (AffectNet facial-affect, EfficientNet) or CLIP
zero-shot affect projection — so the encoder-invariance test has >=2 affect encoders.
Whole frames are fed (same input EmoNet/ViT/ResNet see) for a fair swap.
"""
import os, sys, glob, time, subprocess, numpy as np
from PIL import Image

SP = os.environ["SP"]; ENC = os.environ.get("ENCODER", "hsemotion").lower()
FPS = int(os.environ.get("FPS", "6")); DEV = os.environ.get("DEVICE", "cpu")
FILM = os.environ.get("FILM", "")
t0 = time.time()
def log(*a): print(*a, flush=True)

# ---------- frames at FPS (mirror the original 6-fps extraction) ----------
fdir = f"{SP}/frames6" if FPS != 1 else f"{SP}/frames"
if FPS != 1 and not os.path.isdir(fdir):
    assert FILM and os.path.exists(FILM), "need FILM to extract >1 fps frames (set FILM=...)"
    os.makedirs(fdir, exist_ok=True)
    log(f"ffmpeg extracting {FPS} fps -> {fdir} ...")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",FILM,"-vf",f"fps={FPS},scale=256:-1",f"{fdir}/f%06d.png"], check=True)
frames = sorted(glob.glob(f"{fdir}/f*.png"))
assert frames, f"no frames in {fdir}"
log(f"{len(frames)} frames @ {FPS} fps from {fdir}")

# ---------- encoder ----------
import torch
if ENC == "hsemotion":
    from hsemotion.facial_emotions import HSEmotionRecognizer
    rec = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device=DEV)
    has_feat = hasattr(rec, "extract_features")
    log(f"HSEmotion loaded (extract_features={'yes' if has_feat else 'no; using logits'})")
    def encode(pil):
        img = np.asarray(pil.convert("RGB"))
        if has_feat:
            return np.asarray(rec.extract_features(img)).ravel().astype(np.float32)   # ~1280-d embedding
        _, sc = rec.predict_emotions(img, logits=True)
        return np.asarray(sc).ravel().astype(np.float32)                              # ~10-d affect logits
elif ENC == "clip":
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval().to(DEV); tok = open_clip.get_tokenizer("ViT-B-32")
    PROMPTS = ["a joyful scene","a sad scene","an angry scene","a fearful scene","a disgusting scene",
               "a surprising scene","a calm scene","an exciting scene","a tender romantic scene",
               "an anxious tense scene","a neutral scene","an awe-inspiring scene"]
    with torch.no_grad():
        T = model.encode_text(tok(PROMPTS).to(DEV)); T = (T/T.norm(dim=-1,keepdim=True)).float()
    def encode(pil):
        with torch.no_grad():
            x = pre(pil.convert("RGB")).unsqueeze(0).to(DEV)
            v = model.encode_image(x); v = (v/v.norm(dim=-1,keepdim=True)).float()
            return (v @ T.T).cpu().numpy().ravel().astype(np.float32)                 # 12-d affect-prompt similarities
else:
    raise SystemExit(f"unknown ENCODER={ENC}")

# ---------- run + per-TR average ----------
F = []
for i, fp in enumerate(frames):
    F.append(encode(Image.open(fp)))
    if i % 2000 == 0: log(f"  {i}/{len(frames)} ({time.time()-t0:.0f}s)")
F = np.asarray(F, np.float32)
n = (len(F)//FPS)*FPS
A2 = F[:n].reshape(-1, FPS, F.shape[1]).mean(1) if FPS != 1 else F
np.savez(f"{SP}/feat_affect2.npz", A2=A2)
log(f"DONE: feat_affect2.npz  A2 {A2.shape}  encoder={ENC}  ({time.time()-t0:.0f}s)")
