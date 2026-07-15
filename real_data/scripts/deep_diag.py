import os, glob, numpy as np, nibabel as nib, sys
from nilearn import datasets, image
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.banded_ridge import fit_r2
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
ref=nib.load(subs[0]); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(name):
    idx=list(hoc["labels"]).index(name)
    m=image.resample_to_img(image.new_img_like(hoc["maps"],(np.asarray(hoc["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True)
    return np.asarray(m.dataobj)>0.5
mFUS=mv("Temporal Occipital Fusiform Cortex"); mV1=mv("Occipital Pole")
FUS=[];V1=[]
for p in subs:
    B=np.asarray(nib.load(p).dataobj,dtype=np.float32)
    for M,acc in [(mFUS,FUS),(mV1,V1)]:
        ts=B[M].T; acc.append(((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))
    del B
T=min(a.shape[0] for a in FUS); FUS=np.mean([a[:T] for a in FUS],0); V1=np.mean([a[:T] for a in V1],0)  # pooled hi-SNR
av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz")
def prep(F,scale,k):
    Nm=F.shape[0]; mt=np.clip(scale*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6)
    if R.shape[1]>k:
        _,_,Vt=np.linalg.svd(R-R.mean(0),full_matrices=False); R=R@Vt[:k].T; R=(R-R.mean(0))/(R.std(0)+1e-6)
    return R.astype(np.float32)
def r2(F,y,scale,k):
    X=prep(F,scale,k); r,_=fit_r2([X],y,lambdas_grid=[10.,100.,1000.,1e4,1e5],n_folds=5); return float(np.nanmean(r))
print("Decisive: EmoNet(frame-avg) -> POOLED fusiform (hi-SNR). broken vs underpowered?\n")
print("PCA dims:   ", "  ".join(f"k={k}" for k in [10,30,80,150]))
for name,F,scale in [("EmoNet->FUS",av["E"],1.042),("ViT->FUS",av["VT"],1.042),("lowlevel->FUS",old["LL"],1.042)]:
    print(f"  {name:16s} " + "  ".join(f"{r2(F,FUS,scale,k):+.4f}" for k in [10,30,80,150]))
print("\nFine-scale check EmoNet->pooled fusiform (k=30):")
for s in [1.036,1.039,1.042,1.045,1.048,1.000]:
    print(f"   scale {s:.3f}: R^2 {r2(av['E'],FUS,s,30):+.4f}")
