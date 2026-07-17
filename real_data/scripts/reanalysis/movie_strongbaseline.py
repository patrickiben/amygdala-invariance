#!/usr/bin/env python
"""Movie-arm STRONG-BASELINE red-team. The paper's movie finding is that a generic object encoder (ResNet)
recovers the amygdala's tiny EmoNet signal (EmoNet-minus-ResNet ~ +0.0009). Strong-baseline test: does the
STRONGEST generic encoder (DINOv2, self-distilled) recover it as well or better? Per-subject PLS variance
partitions: {enc}OverViT = R2([enc,vit]) - R2([vit]); EmoRes = emo-res (paper's); EmoDino = emo-dino (strong).
If EmoDino <= 0 (or n.s.), the tiny amygdala EmoNet signal is captured by the strongest generic encoder too ->
not affect-specific, robust to a stronger baseline. Writes $SP/movie_strongbaseline_result.txt."""
import os, glob, numpy as np, nibabel as nib, time
from nilearn import datasets, image
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
import sys; sys.path.insert(0, next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / "amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ.get("SP", os.path.expanduser("~/amyg_rerun/amyg_data")); SCALE=1.042; PCADIM=300; NFOLD=5; NCOMP=10
OUT=open(f"{SP}/movie_strongbaseline_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
log(f"movie strong-baseline (PLS n_comp={NCOMP}, PCA={PCADIM}, PAL {SCALE}); n={len(subs)}")
ref=nib.load(subs[0])
hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name)
    m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True,force_resample=True)
    return np.asarray(m.dataobj)>0.5
masks={"L":mv(hos,"Left Amygdala"),"R":mv(hos,"Right Amygdala"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex")}
av=np.load(f"{SP}/feat_avg_cache.npz"); st=np.load(f"{SP}/feat_strong.npz")
F0={"emonet":av["E"],"vit":av["VT"],"resnet":av["RN"],"dinov2":st["DINO"],"clip":st["CLIP"]}
def prep(F,T):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float64)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6)
    if R.shape[1]>PCADIM: R=PCA(n_components=PCADIM,svd_solver="randomized",random_state=0).fit_transform(R)
    return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float64)
def blocks(n,k): e=np.linspace(0,n,k+1).astype(int); return [np.arange(e[i],e[i+1]) for i in range(k)]
def r2(feat,keys,y):
    X=np.concatenate([feat[k] for k in keys],1); n=X.shape[0]; fo=blocks(n,NFOLD); pred=np.zeros_like(y)
    for i in range(NFOLD):
        te=fo[i]; tr=np.concatenate([fo[j] for j in range(NFOLD) if j!=i])
        p=PLSRegression(n_components=min(NCOMP,X.shape[1]),scale=False); p.fit(X[tr],y[tr]); pred[te]=p.predict(X[te])
    ss=((y-pred)**2).sum(0); su=((y-y.mean(0))**2).sum(0); return float(np.nanmean(1-ss/np.where(su<1e-9,np.nan,su)))
res={f"{r}_{c}":[] for r in ("L","R","FUS") for c in ("EmoRes","EmoDino","emoOvit","dinoOvit","resOvit")}
for p in subs:
    ss=time.time(); sid=p.split('sub-')[1].split('_')[0]; B=np.asarray(nib.load(p).dataobj,dtype=np.float32); T=B.shape[-1]
    reg={k:((lambda ts:((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float64))(B[M].T)) for k,M in masks.items()}; del B
    feat={n:prep(F,T) for n,F in F0.items()}
    for rk in ("L","R","FUS"):
        y=reg[rk]; bv=r2(feat,["vit"],y)
        emo=r2(feat,["emonet","vit"],y)-bv; dino=r2(feat,["dinov2","vit"],y)-bv; rez=r2(feat,["resnet","vit"],y)-bv
        res[f"{rk}_emoOvit"].append(emo); res[f"{rk}_dinoOvit"].append(dino); res[f"{rk}_resOvit"].append(rez)
        res[f"{rk}_EmoRes"].append(emo-rez); res[f"{rk}_EmoDino"].append(emo-dino)
    log(f"  sub-{sid:>2} ({time.time()-ss:.0f}s): L EmoRes {res['L_EmoRes'][-1]:+.4f} EmoDino {res['L_EmoDino'][-1]:+.4f} | FUS EmoRes {res['FUS_EmoRes'][-1]:+.4f} EmoDino {res['FUS_EmoDino'][-1]:+.4f}")
def g(k): a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),pp
log(f"\n=== GROUP (n={len(subs)}) — does EmoNet beat the STRONGEST generic (DINOv2)? ===")
for rk,lab in [("L","L-amygdala"),("R","R-amygdala"),("FUS","Fusiform (pos.ctrl)")]:
    for c,cl in [("EmoRes","EmoNet-minus-ResNet (paper)"),("EmoDino","EmoNet-minus-DINOv2 (strong)"),("dinoOvit","DINOv2-unique-over-ViT")]:
        me,se,pp=g(f"{rk}_{c}"); log(f"  {lab:20s} {cl:32s} {me:+.4f} +/- {se:.4f}  p={pp:.3f}")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
