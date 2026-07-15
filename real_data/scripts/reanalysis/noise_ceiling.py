#!/usr/bin/env python
"""REANALYSIS 2 of 3 — reliability / noise-ceiling for the two-encoder convergence (Figure 3).

WHY: Figure 3 reports the across-participant correlation between the two affect encoders' unique
variance (EmoNet-over-ImageNet vs face-affect-over-ImageNet): r=+0.94 in fusiform, r=-0.26/-0.51
in the amygdala. A referee will object that the amygdala sits at the noise floor, so a low or
negative correlation there could reflect UNRELIABLE per-participant estimates rather than genuine
encoder divergence. This script quantifies that: it estimates the split-half RELIABILITY of each
encoder's per-participant unique-variance in each region, disattenuates the convergence correlation
by that reliability, and puts a bootstrap CI on the observed correlation.

INTERPRETATION:
  - If amygdala reliabilities are near zero -> the -0.26/-0.51 correlations are uninterpretable
    (the referee is right); reframe Fig 3 as supporting, or restrict the convergence claim.
  - If amygdala reliabilities are clearly positive AND the observed correlation is significantly
    below the fusiform value (CI excludes it) -> the divergence is real, not a noise artifact;
    report the reliabilities alongside Fig 3.

FAITHFUL TO second_affect_swap.py: same caches, thr50 amygdala + thr25 fusiform masks, prep()
pipeline, PLS r2. Reliability = across-participant corr of unique-variance estimated on the first
vs second temporal half of the film (two independent estimates per participant).

RUN:      SP=$AMYG_DATA python noise_ceiling.py   (needs feat_avg_cache.npz, feat_cache/affect2, BOLD)
VALIDATE: sanity-check that fusiform reliability is high (it should be, given r=+0.94).
============================================================================================
"""
import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
sys.path.insert(0, next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / "amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf

SP = os.environ["SP"]; SCALE = 1.042; NCOMP = 10; PCADIM = 300; NFOLD = 5; NBOOT = 5000
OUT = open(f"{SP}/noise_ceiling_result.txt", "w")
def log(*a): s = " ".join(map(str, a)); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
t0 = time.time()
subs = sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p: int(p.split('sub-')[1].split('_')[0]))
A2raw = np.load(f"{SP}/feat_affect2.npz")["A2"]; av = np.load(f"{SP}/feat_avg_cache.npz")
ref = nib.load(subs[0])
hos5 = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
hoc = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl, name):
    idx = list(atl["labels"]).index(name)
    m = image.resample_to_img(image.new_img_like(atl["maps"], (np.asarray(atl["maps"].dataobj) == idx).astype("float32")),
                              ref, interpolation="nearest", copy_header=True)
    return np.asarray(m.dataobj) > 0.5
masks = {"L": mv(hos5, "Left Amygdala"), "R": mv(hos5, "Right Amygdala"), "FUS": mv(hoc, "Temporal Occipital Fusiform Cortex")}
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
def r2(X, y):
    n = X.shape[0]; fo = blocks(n, NFOLD); pred = np.zeros_like(y); nc = min(NCOMP, X.shape[1])
    for i in range(NFOLD):
        te = fo[i]; tr = np.concatenate([fo[j] for j in range(NFOLD) if j != i])
        pls = PLSRegression(n_components=nc, scale=False); pls.fit(X[tr], y[tr]); pred[te] = pls.predict(X[te])
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.nanmean(1 - ss / np.where(st < 1e-9, np.nan, st)))
def unique_over_img(E, VR, y):  # X-unique over ImageNet, on a given index subset already applied
    return r2(np.concatenate([E, VR], 1), y) - r2(VR, y)

# per-participant estimates: full film, first half, second half -> for each region, emo & a2 unique-over-img
est = {rk: {q: {"emo": [], "a2": []} for q in ("full", "h1", "h2")} for rk in masks}
for p in subs:
    sid = p.split('sub-')[1].split('_')[0]; B = np.asarray(nib.load(p).dataobj, dtype=np.float32); T = B.shape[-1]
    reg = {k: (lambda ts: ((ts - ts.mean(0)) / (ts.std(0) + 1e-6)).astype(np.float32))(B[M].T) for k, M in masks.items()}; del B
    E = prep(av["E"], T); V = prep(av["VT"], T); RN = prep(av["RN"], T); A2 = prep(A2raw, T); VR = np.concatenate([V, RN], 1)
    half = T // 2
    idxs = {"full": np.arange(T), "h1": np.arange(half), "h2": np.arange(half, T)}
    for rk in masks:
        for q, ix in idxs.items():
            y = reg[rk][ix]
            est[rk][q]["emo"].append(unique_over_img(E[ix], VR[ix], y))
            est[rk][q]["a2"].append(unique_over_img(A2[ix], VR[ix], y))
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s) done")

def corr(a, b): a = np.asarray(a); b = np.asarray(b); return float(np.corrcoef(a, b)[0, 1])
rng = np.random.RandomState(0)
log(f"\n=== RELIABILITY + DISATTENUATED CONVERGENCE (n={len(subs)} participants) ===")
log("  region  rel_emo  rel_a2   obs_conv   disattenuated   boot95%CI(obs)")
for rk in ("FUS", "L", "R"):
    d = est[rk]
    rel_emo = corr(d["h1"]["emo"], d["h2"]["emo"])   # split-half reliability of EmoNet unique-var estimate
    rel_a2 = corr(d["h1"]["a2"], d["h2"]["a2"])
    obs = corr(d["full"]["emo"], d["full"]["a2"])     # the Figure-3 convergence correlation
    denom = np.sqrt(max(rel_emo, 0) * max(rel_a2, 0))
    dis = obs / denom if denom > 1e-6 else float("nan")
    ef = np.array(d["full"]["emo"]); af = np.array(d["full"]["a2"]); m = len(ef)
    bs = [corr(ef[ii], af[ii]) for ii in (rng.randint(0, m, m) for _ in range(NBOOT))]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    lab = {"FUS": "Fusiform", "L": "L-amyg", "R": "R-amyg"}[rk]
    log(f"  {lab:8s} {rel_emo:+.3f}  {rel_a2:+.3f}   {obs:+.3f}      {dis:+.3f}         [{lo:+.3f}, {hi:+.3f}]")
log("\nREAD: high fusiform reliability + r~+0.94 validates the metric. In amygdala, if reliabilities are")
log("near 0 the negative correlation is a noise artifact (reframe Fig 3); if clearly positive and the CI")
log("excludes the fusiform value, the encoder divergence is real. Report rel_emo/rel_a2 with Fig 3.")
log(f"DONE ({time.time()-t0:.0f}s)"); OUT.close()
