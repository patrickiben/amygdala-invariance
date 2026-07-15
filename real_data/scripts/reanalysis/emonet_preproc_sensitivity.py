#!/usr/bin/env python
"""REANALYSIS 3 of 3 — EmoNet preprocessing sensitivity (movie arm).

WHY: EmoNet features were extracted with different input preprocessing across arms (movie arm:
raw /255, no ImageNet norm; face-morph arm: ImageNet-normalized at 227px). EmoNet (Kragel 2019,
AlexNet-based) has its own canonical preprocessing, and using the wrong one could distort the
extracted features and hence the fusiform positive control and amygdala null. This script re-extracts
the movie-arm EmoNet features under THREE candidate preprocessings and re-runs the key movie
contrasts, so you can confirm the verdict is robust to the choice (and pick the correct one).

PREPROCESSING VARIANTS (all on 227x227 RGB frames):
  raw      : frame/255.0                                  (as used in the committed movie run)
  imagenet : ImageNet mean/std normalization              (as used in the committed face-morph run)
  caffe    : BGR, subtract ImageNet Caffe mean [104,117,123], no scaling  (AlexNet/Caffe native)

For each variant it rebuilds E, TR-averages, and computes (reusing pls20.py's exact pipeline) the
fusiform positive control and the L/R amygdala EmoNet-unique-over-ImageNet contrasts. The verdict is
robust if, under ALL THREE, the amygdala EmoNet-unique-over-ImageNet stays non-significant while the
fusiform positive control stays significant. Also cross-check which variant maximizes the fusiform
positive control (that is the empirically correct EmoNet preprocessing) and adopt it in BOTH arms.

REQUIRES: torch, torchvision, cv2; the film frames ($SP/frames6 from extract_movie_features.py or
extract_toisoul.py), the OSF EmoNet weights, and the ecco-laboratory/emonet-pytorch clone (see
extract_movie_features.py for the exact loading). ViT/ResNet/lowlevel come from the existing caches.
RUN:      SP=$AMYG_DATA EMONET_REPO=... EMONET_WEIGHTS=... python emonet_preproc_sensitivity.py
COMPANION (do the same in the face-morph arm): apply these three variants to the kragelEmoNet
extraction in extract_morph_features.py, then re-run sun_rsa.py, and confirm the fusiform RSA
positive control (kragelEmoNet-ResNet, p=.005) and the amygdala null are unchanged.
============================================================================================
"""
import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0, next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / "amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
import torch, cv2

SP = os.environ["SP"]; SCALE = 1.042; NCOMP = 10; PCADIM = 300; NFOLD = 5; FPS = 6; BATCH = 64
OUT = open(f"{SP}/emonet_preproc_sensitivity_result.txt", "w")
def log(*a): s = " ".join(map(str, a)); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
t0 = time.time()
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# --- load EmoNet (see extract_movie_features.py provenance note) ---
emonet_repo = os.environ.get("EMONET_REPO", f"{SP}/emonet_pytorch")
emonet_wts = os.environ.get("EMONET_WEIGHTS", f"{SP}/emonet/emonet.pth")
sys.path.insert(0, emonet_repo)
from emonet_pytorch import EmoNet  # CONFIRM import path per the ecco-laboratory repo
emonet = EmoNet(); emonet.load_state_dict(torch.load(emonet_wts, map_location="cpu"), strict=False)
emonet.eval().to(DEV)
ebuf = {}
emonet.classifier[-2].register_forward_hook(lambda m, i, o: ebuf.__setitem__("f", o.flatten(1).detach().cpu().numpy()))

frames = sorted(glob.glob(f"{SP}/frames6/f*.png"))
assert frames, "need $SP/frames6 (build with extract_movie_features.py or extract_toisoul.py first)"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])
CAFFE_MEAN_BGR = np.array([104.0, 117.0, 123.0])

def to_tensor(chunk, variant):
    ims = []
    for fp in chunk:
        im = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (227, 227)).astype(np.float32)
        if variant == "raw":
            im = im / 255.0
        elif variant == "imagenet":
            im = ((im / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        elif variant == "caffe":
            im = im[:, :, ::-1] - CAFFE_MEAN_BGR   # RGB->BGR, subtract Caffe mean, no /255
        ims.append(im)
    return torch.from_numpy(np.stack(ims)).float().permute(0, 3, 1, 2).to(DEV)

def extract_E(variant):
    F = []
    for i in range(0, len(frames), BATCH):
        with torch.no_grad(): emonet(to_tensor(frames[i:i + BATCH], variant))
        F.append(ebuf["f"].copy())
    F = np.vstack(F).astype(np.float32); n = (len(F) // FPS) * FPS
    return F[:n].reshape(-1, FPS, F.shape[1]).mean(1).astype(np.float32)

# --- pls20 infra (verbatim) ---
subs = sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p: int(p.split('sub-')[1].split('_')[0]))
ref = nib.load(subs[0])
hos = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl, name):
    idx = list(atl["labels"]).index(name)
    m = image.resample_to_img(image.new_img_like(atl["maps"], (np.asarray(atl["maps"].dataobj) == idx).astype("float32")), ref, interpolation="nearest", copy_header=True)
    return np.asarray(m.dataobj) > 0.5
masks = {"L": mv(hos, "Left Amygdala"), "R": mv(hos, "Right Amygdala"), "FUS": mv(hoc, "Temporal Occipital Fusiform Cortex")}
old = np.load(f"{SP}/feat_cache.npz"); av = np.load(f"{SP}/feat_avg_cache.npz")
BASE = {"vit": av["VT"], "resnet": av["RN"], "lowlevel": old["LL"]}
def pca(X, k):
    X = X - X.mean(0)
    if X.shape[1] <= k: return X
    _, _, Vt = np.linalg.svd(X, full_matrices=False); return X @ Vt[:k].T
def prep(F, T):
    Nm = F.shape[0]; mt = np.clip(SCALE * np.arange(T), 0, Nm - 1)
    R = np.stack([np.interp(mt, np.arange(Nm), F[:, c]) for c in range(F.shape[1])], 1).astype(np.float32)
    R = convolve_hrf(R, 1.0); R = (R - R.mean(0)) / (R.std(0) + 1e-6); R = pca(R, PCADIM)
    return ((R - R.mean(0)) / (R.std(0) + 1e-6)).astype(np.float32)
def blocks(n, k): e = np.linspace(0, n, k + 1).astype(int); return [np.arange(e[i], e[i + 1]) for i in range(k)]
def r2set(feat, keys, y):
    X = np.concatenate([feat[k] for k in keys], 1); n = X.shape[0]; fo = blocks(n, NFOLD); pred = np.zeros_like(y); nc = min(NCOMP, X.shape[1])
    for i in range(NFOLD):
        te = fo[i]; tr = np.concatenate([fo[j] for j in range(NFOLD) if j != i])
        pls = PLSRegression(n_components=nc, scale=False); pls.fit(X[tr], y[tr]); pred[te] = pls.predict(X[te])
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.nanmean(1 - ss / np.where(st < 1e-9, np.nan, st)))

for variant in ("raw", "imagenet", "caffe"):
    Evar = extract_E(variant); F0 = dict(BASE, emonet=Evar)
    res = {k: [] for k in ["FUS_emo", "L_uei", "R_uei"]}
    for p in subs:
        B = np.asarray(nib.load(p).dataobj, dtype=np.float32); T = B.shape[-1]
        reg = {k: (lambda ts: ((ts - ts.mean(0)) / (ts.std(0) + 1e-6)).astype(np.float32))(B[M].T) for k, M in masks.items()}; del B
        feat = {n: prep(F, T) for n, F in F0.items()}
        res["FUS_emo"].append(r2set(feat, ["emonet"], reg["FUS"]))
        for rk in ("L", "R"):
            res[f"{rk}_uei"].append(r2set(feat, ["emonet", "vit", "resnet"], reg[rk]) - r2set(feat, ["vit", "resnet"], reg[rk]))
    def grp(k): a = np.array(res[k]); t, pp = stats.ttest_1samp(a, 0); return a.mean(), t, pp
    log(f"\n=== EmoNet preprocessing = {variant} ({time.time()-t0:.0f}s) ===")
    for k, lab in [("FUS_emo", "Fusiform<-EmoNet (positive control)"), ("L_uei", "L-amyg EmoNet-unique OVER ImageNet"), ("R_uei", "R-amyg EmoNet-unique OVER ImageNet")]:
        m, t, pp = grp(k); log(f"  {lab:40s} mean {m:+.4f} t={t:+.2f} p={pp:.3f}")
log("\nROBUST if amygdala uei stays non-significant under all three variants and the fusiform control")
log("stays significant. The variant with the largest fusiform positive control is the correct EmoNet")
log("preprocessing; adopt it in BOTH the movie and face-morph arms and confirm nothing else moves.")
log(f"DONE ({time.time()-t0:.0f}s)"); OUT.close()
