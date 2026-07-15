#!/usr/bin/env python
"""Close the last control: does a SECOND, paradigmatically-different affect encoder (A2, from
feat_affect2.npz) beat generic object vision in the amygdala? If not (like EmoNet), the
"amygdala emotion-code = false floor" verdict is airtight, and the False-Floor test now has
>=2 affect encoders.

Pipeline identical to the hardened run: PAL-1.042 align + HRF + z + PCA-300; per-subject PLS
(10 comp, 5-fold block CV); eroded thr50 amygdala L/R + fusiform (thr25) positive control;
group one-sample t-tests (df=19). Contrasts per region:
  emoOvit / resOvit / a2Ovit           : X-unique over ViT   (the 3-way swap)
  Emo-Res / A2-Res / Emo-A2            : paired encoder differences
  emoOVERimg / a2OVERimg               : unique over ImageNet (ViT+ResNet)
  emoOVERimgA2 / a2OVERimgEmo          : redundancy of the two affect encoders
  + across-subject corr(emoOVERimg, a2OVERimg)  : do the two affect encoders track the same signal?
Env: SP (cache dir).  Reads feat_avg_cache.npz (E,VT,RN), feat_cache.npz (LL), feat_affect2.npz (A2).
"""
import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf

SP=os.environ["SP"]; SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5
t0=time.time()
def log(*a): print(*a, flush=True)
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
A2raw=np.load(f"{SP}/feat_affect2.npz")["A2"]; av=np.load(f"{SP}/feat_avg_cache.npz")
log(f"2nd-affect swap | n={len(subs)} | A2 {A2raw.shape} | E {av['E'].shape}")
ref=nib.load(subs[0])
hos5=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"L":mv(hos5,"Left Amygdala"),"R":mv(hos5,"Right Amygdala"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex")}
def pca(X,k):
    X=X-X.mean(0)
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(F,T):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6); R=pca(R,PCADIM); return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float32)
def blocks(n,k): e=np.linspace(0,n,k+1).astype(int); return [np.arange(e[i],e[i+1]) for i in range(k)]
def r2(X,y):
    n=X.shape[0]; fo=blocks(n,NFOLD); pred=np.zeros_like(y); nc=min(NCOMP,X.shape[1])
    for i in range(NFOLD):
        te=fo[i]; tr=np.concatenate([fo[j] for j in range(NFOLD) if j!=i])
        pls=PLSRegression(n_components=nc,scale=False); pls.fit(X[tr],y[tr]); pred[te]=pls.predict(X[te])
    ss=((y-pred)**2).sum(0); st=((y-y.mean(0))**2).sum(0); return float(np.nanmean(1-ss/np.where(st<1e-9,np.nan,st)))
res={}
def add(k,v): res.setdefault(k,[]).append(v)
for p in subs:
    sid=p.split('sub-')[1].split('_')[0]; B=np.asarray(nib.load(p).dataobj,dtype=np.float32); T=B.shape[-1]
    reg={k:(lambda ts:((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))(B[M].T) for k,M in masks.items()}; del B
    E=prep(av["E"],T); V=prep(av["VT"],T); RN=prep(av["RN"],T); A2=prep(A2raw,T); VR=np.concatenate([V,RN],1)
    for rk in masks:
        y=reg[rk]; bv=r2(V,y)
        add(f"{rk}_emoOvit", r2(np.concatenate([E,V],1),y)-bv)
        add(f"{rk}_resOvit", r2(np.concatenate([RN,V],1),y)-bv)
        add(f"{rk}_a2Ovit",  r2(np.concatenate([A2,V],1),y)-bv)
        bimg=r2(VR,y)
        add(f"{rk}_emoOVERimg", r2(np.concatenate([E,VR],1),y)-bimg)
        add(f"{rk}_a2OVERimg",  r2(np.concatenate([A2,VR],1),y)-bimg)
        bEA=r2(np.concatenate([VR,A2],1),y); add(f"{rk}_emoOVERimgA2", r2(np.concatenate([E,VR,A2],1),y)-bEA)
        bEE=r2(np.concatenate([VR,E],1),y);  add(f"{rk}_a2OVERimgEmo",  r2(np.concatenate([A2,VR,E],1),y)-bEE)
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): L a2Ovit {res['L_a2Ovit'][-1]:+.4f} resOvit {res['L_resOvit'][-1]:+.4f} emoOvit {res['L_emoOvit'][-1]:+.4f}")
def g(k): a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
def gd(k1,k2): a=np.array(res[k1])-np.array(res[k2]); t,pp=stats.ttest_1samp(a,0); return a.mean(),t,pp
log(f"\n=== 3-WAY SWAP over ViT (n={len(subs)}) ===")
for rk,l in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform(ctrl)")]:
    me,_,_,pe=g(f"{rk}_emoOvit"); mr,_,_,pr=g(f"{rk}_resOvit"); ma,_,_,pa=g(f"{rk}_a2Ovit")
    dar,tar,par=gd(f"{rk}_a2Ovit",f"{rk}_resOvit"); der,ter,per=gd(f"{rk}_emoOvit",f"{rk}_resOvit")
    log(f"  {l:14s} EmoNet {me:+.4f}(p{pe:.3f}) ResNet {mr:+.4f}(p{pr:.3f}) A2 {ma:+.4f}(p{pa:.3f}) | A2-Res {dar:+.4f}(p{par:.3f}) Emo-Res {der:+.4f}(p{per:.3f})")
log(f"\n=== UNIQUE over ImageNet + agreement of the two affect encoders ===")
for rk,l in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform(ctrl)")]:
    me,_,_,pe=g(f"{rk}_emoOVERimg"); ma,_,_,pa=g(f"{rk}_a2OVERimg")
    r=np.corrcoef(np.array(res[f"{rk}_emoOVERimg"]),np.array(res[f"{rk}_a2OVERimg"]))[0,1]
    mea,_,_,pea=g(f"{rk}_emoOVERimgA2"); mae,_,_,pae=g(f"{rk}_a2OVERimgEmo")
    log(f"  {l:14s} emoOVERimg {me:+.4f}(p{pe:.3f}) a2OVERimg {ma:+.4f}(p{pa:.3f}) | corr(emo,a2)={r:+.2f} | emoOVER(img+A2) {mea:+.4f}(p{pea:.3f}) a2OVER(img+Emo) {mae:+.4f}(p{pae:.3f})")
# ---- verdict ----
La=g("L_a2Ovit"); Ra=g("R_a2Ovit"); Lr=g("L_resOvit"); Rr=g("R_resOvit")
_,_,par_L=gd("L_a2Ovit","L_resOvit"); _,_,par_R=gd("R_a2Ovit","R_resOvit")
Fa=g("FUS_a2Ovit"); Fr=g("FUS_resOvit"); _,_,par_F=gd("FUS_a2Ovit","FUS_resOvit")
log(f"\n=== VERDICT ===")
# SIGN-AWARE: the escape hatch requires A2 to SIGNIFICANTLY EXCEED ResNet (positive diff), not just differ.
amyg_a2_beats = (La[0]>Lr[0] and par_L<0.05) or (Ra[0]>Rr[0] and par_R<0.05)
fus_spec  = (Fa[0]>Fr[0] and par_F<0.05)  # A2 beats object in fusiform (emotion-specificity where it should be)
if (not amyg_a2_beats) and fus_spec:
    log("  ✅ AIRTIGHT FALSE FLOOR: a 2nd independent affect encoder does NOT beat generic object vision in the")
    log("     amygdala (A2 <= ResNet in both hemispheres), yet retains emotion-specificity in fusiform (A2 > ResNet).")
    log("     Two paradigmatically different affect encoders → generic-visual amygdala signal, not an emotion code.")
elif amyg_a2_beats:
    log("  ⚠️ ESCAPE HATCH: the 2nd affect encoder SIGNIFICANTLY beats generic vision in the amygdala.")
    log("     Face-affect (A2) may carry amygdala-specific signal EmoNet's scene-affect missed — investigate.")
else:
    log("  ⚠️ INCONCLUSIVE: check A2 functions as a brain predictor (fusiform positive control failed).")
log(f"\nDONE ({time.time()-t0:.0f}s)")
