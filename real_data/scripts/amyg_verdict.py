import os, glob, numpy as np, nibabel as nib, sys
from nilearn import datasets, image
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.banded_ridge import fit_r2
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]; SCALE=1.042; K=30
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
ref=nib.load(subs[0]); hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"L-amyg":mv(hos,"Left Amygdala"),"R-amyg":mv(hos,"Right Amygdala"),"Fusiform(ctrl)":mv(hoc,"Temporal Occipital Fusiform Cortex")}
per={k:[] for k in masks}
for p in subs:
    B=np.asarray(nib.load(p).dataobj,dtype=np.float32)
    for k,M in masks.items(): ts=B[M].T; per[k].append(((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))
    del B
T=min(a.shape[0] for k in per for a in per[k])
def isc(region):
    m=[a[:T].mean(1) for a in per[region]]; return float(np.mean([np.corrcoef(m[i],np.mean([m[j] for j in range(len(m)) if j!=i],0))[0,1] for i in range(len(m))]))
bold={k:np.mean([a[:T] for a in v],0) for k,v in per.items()}
av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz"); F0={"emonet":av["E"],"vit":av["VT"],"resnet":av["RN"],"lowlevel":old["LL"]}
def prep(F,k=K):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32); R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6)
    if R.shape[1]>k: _,_,Vt=np.linalg.svd(R-R.mean(0),full_matrices=False); R=R@Vt[:k].T; R=(R-R.mean(0))/(R.std(0)+1e-6)
    return R.astype(np.float32)
feat={n:prep(F) for n,F in F0.items()}
def R2(keys,y): r,_=fit_r2([feat[k] for k in keys],y,lambdas_grid=[10.,100.,1000.,1e4,1e5],n_folds=5); return float(np.nanmean(r))
print(f"AMYGDALA ENCODER-INVARIANCE VERDICT | ds002837 n={len(subs)} pooled | PAL {SCALE} | frame-avg deep feats | PCA-{K}\n")
print(f"ISC (shared-signal strength): Fusiform {isc('Fusiform(ctrl)'):+.3f} | L-amyg {isc('L-amyg'):+.3f} | R-amyg {isc('R-amyg'):+.3f}")
print(f"\nPOSITIVE CONTROL  Fusiform <- EmoNet {R2(['emonet'],bold['Fusiform(ctrl)']):+.4f} | <- ViT {R2(['vit'],bold['Fusiform(ctrl)']):+.4f} | <- low {R2(['lowlevel'],bold['Fusiform(ctrl)']):+.4f}")
print("\n=== AMYGDALA ===")
for hn in ["L-amyg","R-amyg"]:
    y=bold[hn]; s={k:R2([k],y) for k in feat}
    uel=R2(["emonet","lowlevel"],y)-R2(["lowlevel"],y); uei=R2(["emonet","vit","resnet"],y)-R2(["vit","resnet"],y)
    print(f"  {hn}: emo {s['emonet']:+.4f} vit {s['vit']:+.4f} resnet {s['resnet']:+.4f} low {s['lowlevel']:+.4f}")
    print(f"        EmoNet-unique vs low-level {uel:+.4f} | EmoNet-unique OVER ImageNet {uei:+.4f}")
