#!/usr/bin/env python
"""REANALYSIS 1 of 3 — the PREREGISTERED banded-ridge encoding path.

WHY: the committed movie-arm result (pls20.py) used per-participant PLS (n_components=10,
PCA-300), whereas the preregistration specified banded ridge (himalaya). A specialist referee
will require the preregistered path. This script re-runs the identical contrasts under banded
ridge so you can confirm the verdict (EmoNet adds no unique variance over object encoders in the
amygdala; fusiform positive control holds) survives the estimator change.

FAITHFUL TO pls20.py: same caches (feat_avg_cache.npz {E,VT,RN}, feat_cache.npz {LL,RC}), same
Harvard-Oxford masks (subcortical thr25 amygdala, cortical thr25 fusiform), same prep() pipeline
(PAL 1.042 -> HRF -> z -> PCA-300 -> z), same 5-fold contiguous-block CV, same variance-partition
contrasts as difference of held-out R2. The ONLY change is the estimator: PLSRegression -> banded
ridge with a separate, cross-validated regularization per encoder band.

REQUIRES: pip install himalaya>=0.4  (uncomment it in real_data/requirements.txt).
RUN:      SP=$AMYG_DATA python pls20_banded.py   (needs the movie feature caches + BOLD derivatives)
VALIDATE: compare the group table below to real_data/results/pls20_result.txt. The verdict is
          confirmed if EmoNet-unique-OVER-ImageNet in L/R amygdala remains non-significant while
          the fusiform positive control stays significant. Report both PLS and banded-ridge in a
          supplement, or replace the primary numbers with banded ridge if you adopt it as primary.
============================================================================================
"""
import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from scipy import stats
sys.path.insert(0, next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / "amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf

# --- banded ridge (preregistered production estimator) ---
from himalaya.backend import set_backend
from himalaya.ridge import BandedRidgeCV
set_backend("numpy")  # CPU; use "torch_cuda" if a GPU is available

SP = os.environ["SP"]; SCALE = 1.042; PCADIM = 300; NFOLD = 5
ALPHAS = np.logspace(-5, 10, 20)   # per-band regularization grid searched by inner CV
N_ITER = 100                        # random-search iterations over band-alpha combinations
INNER_CV = 5                        # inner CV for alpha selection within each training fold
OUT = open(f"{SP}/pls20_banded_result.txt", "w")
def log(*a): s = " ".join(map(str, a)); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
t0 = time.time()

subs = sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p: int(p.split('sub-')[1].split('_')[0]))
log(f"PREREGISTERED banded ridge (himalaya; alphas={ALPHAS.min():.0e}..{ALPHAS.max():.0e}, n_iter={N_ITER}, "
    f"PCA={PCADIM}, {NFOLD}-fold block CV, PAL {SCALE}); n={len(subs)}")
ref = nib.load(subs[0])
hos = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
hoc = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl, name):
    idx = list(atl["labels"]).index(name)
    m = image.resample_to_img(image.new_img_like(atl["maps"], (np.asarray(atl["maps"].dataobj) == idx).astype("float32")),
                              ref, interpolation="nearest", copy_header=True)
    return np.asarray(m.dataobj) > 0.5
masks = {"L": mv(hos, "Left Amygdala"), "R": mv(hos, "Right Amygdala"), "FUS": mv(hoc, "Temporal Occipital Fusiform Cortex")}
av = np.load(f"{SP}/feat_avg_cache.npz"); old = np.load(f"{SP}/feat_cache.npz")
F0 = {"emonet": av["E"], "vit": av["VT"], "resnet": av["RN"], "lowlevel": old["LL"]}

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
    """Held-out R2 (mean over voxels) of a banded-ridge model whose bands are the encoders in `keys`.
    Each band = one encoder's PCA-300 feature block; himalaya selects a separate alpha per band."""
    Xs = [feat[k] for k in keys]
    groups = np.concatenate([np.full(x.shape[1], gi, dtype=int) for gi, x in enumerate(Xs)])
    X = np.concatenate(Xs, 1).astype(np.float32)
    n = X.shape[0]; fo = blocks(n, NFOLD); pred = np.zeros_like(y)
    for i in range(NFOLD):
        te = fo[i]; tr = np.concatenate([fo[j] for j in range(NFOLD) if j != i])
        model = BandedRidgeCV(groups=groups, cv=INNER_CV, solver="random_search",
                              solver_params=dict(alphas=ALPHAS, n_iter=N_ITER, progress_bar=False))
        model.fit(X[tr], y[tr]); pred[te] = np.asarray(model.predict(X[te]))
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.nanmean(1 - ss / np.where(st < 1e-9, np.nan, st)))

res = {k: [] for k in ["FUS_emo", "L_emo", "R_emo", "L_uel", "R_uel", "L_uei", "R_uei"]}
for p in subs:
    sid = p.split('sub-')[1].split('_')[0]; B = np.asarray(nib.load(p).dataobj, dtype=np.float32); T = B.shape[-1]
    reg = {k: (lambda ts: ((ts - ts.mean(0)) / (ts.std(0) + 1e-6)).astype(np.float32))(B[M].T) for k, M in masks.items()}; del B
    feat = {n: prep(F, T) for n, F in F0.items()}
    res["FUS_emo"].append(r2set(feat, ["emonet"], reg["FUS"]))
    for rk in ("L", "R"):
        y = reg[rk]
        res[f"{rk}_emo"].append(r2set(feat, ["emonet"], y))
        res[f"{rk}_uel"].append(r2set(feat, ["emonet", "lowlevel"], y) - r2set(feat, ["lowlevel"], y))
        res[f"{rk}_uei"].append(r2set(feat, ["emonet", "vit", "resnet"], y) - r2set(feat, ["vit", "resnet"], y))
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): FUS<-emo {res['FUS_emo'][-1]:+.4f} | "
        f"L uei {res['L_uei'][-1]:+.4f} | R uei {res['R_uei'][-1]:+.4f}")

def grp(k):
    a = np.array(res[k]); t, pp = stats.ttest_1samp(a, 0); return a.mean(), a.std(ddof=1) / np.sqrt(len(a)), t, pp
log(f"\n=== GROUP (banded ridge; n={len(subs)}, one-sample t vs 0) ===")
for k, lab in [("FUS_emo", "POS-CTRL Fusiform<-EmoNet"), ("L_emo", "L-amyg<-EmoNet"), ("R_emo", "R-amyg<-EmoNet"),
               ("L_uel", "L-amyg EmoNet-unique vs low-level"), ("R_uel", "R-amyg EmoNet-unique vs low-level"),
               ("L_uei", "L-amyg EmoNet-unique OVER ImageNet"), ("R_uei", "R-amyg EmoNet-unique OVER ImageNet")]:
    m, se, t, pp = grp(k); log(f"  {lab:38s} mean {m:+.4f} +/- {se:.4f}  t={t:+.2f} p={pp:.3f}")
log("\nVERDICT CHECK: confirmed if L/R 'EmoNet-unique OVER ImageNet' stay non-significant while "
    "'Fusiform<-EmoNet' stays significant (matches the PLS run in ../results/pls20_result.txt).")
log(f"DONE ({time.time()-t0:.0f}s)"); OUT.close()
