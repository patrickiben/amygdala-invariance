#!/usr/bin/env python
"""Robust, self-contained banded ridge (closed-form, per-band alpha via inner CV; regularized
X'X solve -> no SVD-convergence crashes). Computes the DECISIVE contrast per region:
  emoOvit = R2([emonet,vit]) - R2([vit]);  resOvit = R2([resnet,vit]) - R2([vit]);  EmoRes = emoOvit-resOvit
Amygdala verdict: EmoRes ~ 0 (EmoNet no better than a matched object encoder). Fusiform positive
control: EmoRes > 0. Writes $SP/banded_robust_result.txt. SUBSET env limits subjects."""
import os, glob, itertools, numpy as np, nibabel as nib, time
from nilearn import datasets, image
from sklearn.decomposition import PCA
from scipy import stats
import sys; sys.path.insert(0, next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / "amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf

SP = os.environ["SP"]; SCALE = 1.042; PCADIM = 300; NFOLD = 5; INNER = 2
ALPHAS = np.array([1e1, 1e3, 1e5, 1e7])          # per-band grid
SUBSET = int(os.environ.get("SUBSET", "0"))
OUT = open(f"{SP}/banded_robust_result.txt", "w")
def log(*a): s = " ".join(map(str, a)); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
t0 = time.time()
subs = sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p: int(p.split('sub-')[1].split('_')[0]))
if SUBSET: subs = subs[:SUBSET]
log(f"robust banded ridge (closed-form; alphas={list(ALPHAS)}, inner={INNER}, PCA={PCADIM}); n={len(subs)}")
ref = nib.load(subs[0])
hos = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl, name):
    idx = list(atl["labels"]).index(name)
    m = image.resample_to_img(image.new_img_like(atl["maps"], (np.asarray(atl["maps"].dataobj) == idx).astype("float32")), ref, interpolation="nearest", copy_header=True, force_resample=True)
    return np.asarray(m.dataobj) > 0.5
masks = {"L": mv(hos, "Left Amygdala"), "R": mv(hos, "Right Amygdala"), "FUS": mv(hoc, "Temporal Occipital Fusiform Cortex")}
av = np.load(f"{SP}/feat_avg_cache.npz"); F0 = {"emonet": av["E"], "vit": av["VT"], "resnet": av["RN"]}
def prep(F, T):
    Nm = F.shape[0]; mt = np.clip(SCALE * np.arange(T), 0, Nm - 1)
    R = np.stack([np.interp(mt, np.arange(Nm), F[:, c]) for c in range(F.shape[1])], 1).astype(np.float64)
    R = convolve_hrf(R, 1.0); R = (R - R.mean(0)) / (R.std(0) + 1e-6)
    if R.shape[1] > PCADIM: R = PCA(n_components=PCADIM, svd_solver="randomized", random_state=0).fit_transform(R)
    return ((R - R.mean(0)) / (R.std(0) + 1e-6)).astype(np.float64)
def blocks(n, k): e = np.linspace(0, n, k + 1).astype(int); return [np.arange(e[i], e[i + 1]) for i in range(k)]

def ridge_fit_predict(Xtr, ytr, Xte, pen):
    A = Xtr.T @ Xtr + np.diag(pen); A[np.diag_indices_from(A)] += 1e-6   # ensure PD
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w
def r2_banded(feat, keys, y):
    """held-out R2 (mean over voxels) of banded ridge over the given feature bands, per-band alpha by inner CV."""
    Xs = [feat[k] for k in keys]; groups = np.concatenate([np.full(x.shape[1], gi) for gi, x in enumerate(Xs)])
    X = np.concatenate(Xs, 1); n = X.shape[0]; fo = blocks(n, NFOLD); pred = np.zeros_like(y)
    combos = list(itertools.product(ALPHAS, repeat=len(keys)))
    for i in range(NFOLD):
        te = fo[i]; tr = np.concatenate([fo[j] for j in range(NFOLD) if j != i]); Xtr, ytr = X[tr], y[tr]
        ifo = blocks(len(tr), INNER); best, bestsse = None, np.inf
        for combo in combos:
            pen = np.array([combo[g] for g in groups]); sse = 0.0
            for j in range(INNER):
                ite = ifo[j]; itr = np.concatenate([ifo[m] for m in range(INNER) if m != j])
                p = ridge_fit_predict(Xtr[itr], ytr[itr], Xtr[ite], pen); sse += ((ytr[ite] - p) ** 2).sum()
            if sse < bestsse: bestsse, best = sse, pen
        pred[te] = ridge_fit_predict(Xtr, ytr, X[te], best)
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.nanmean(1 - ss / np.where(st < 1e-9, np.nan, st)))

res = {f"{r}_{c}": [] for r in ("L", "R", "FUS") for c in ("emoOvit", "resOvit", "EmoRes")}
for p in subs:
    st = time.time(); sid = p.split('sub-')[1].split('_')[0]; B = np.asarray(nib.load(p).dataobj, dtype=np.float32); T = B.shape[-1]
    reg = {k: ((lambda ts: ((ts - ts.mean(0)) / (ts.std(0) + 1e-6)).astype(np.float64))(B[M].T)) for k, M in masks.items()}; del B
    feat = {n: prep(F, T) for n, F in F0.items()}
    for rk in ("L", "R", "FUS"):
        y = reg[rk]; bv = r2_banded(feat, ["vit"], y)
        emo = r2_banded(feat, ["emonet", "vit"], y) - bv; rez = r2_banded(feat, ["resnet", "vit"], y) - bv
        res[f"{rk}_emoOvit"].append(emo); res[f"{rk}_resOvit"].append(rez); res[f"{rk}_EmoRes"].append(emo - rez)
    log(f"  sub-{sid:>2} ({time.time()-st:.0f}s/subj): L EmoRes {res['L_EmoRes'][-1]:+.4f} R EmoRes {res['R_EmoRes'][-1]:+.4f} FUS EmoRes {res['FUS_EmoRes'][-1]:+.4f}")
if len(subs) >= 3:
    def g(k): a = np.array(res[k]); t, pp = stats.ttest_1samp(a, 0); return a.mean(), a.std(ddof=1) / np.sqrt(len(a)), t, pp
    log(f"\n=== GROUP (robust banded ridge; n={len(subs)}) ===")
    for rk, lab in [("L", "L-amygdala"), ("R", "R-amygdala"), ("FUS", "Fusiform (pos. control)")]:
        me, se, t, pp = g(f"{rk}_EmoRes"); log(f"  {lab:24s} EmoNet-minus-ResNet {me:+.4f} +/- {se:.4f}  t={t:+.2f} p={pp:.3f}")
log(f"DONE ({time.time()-t0:.0f}s)"); OUT.close()
