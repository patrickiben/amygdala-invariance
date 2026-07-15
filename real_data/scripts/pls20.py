import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]; SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5
OUT=open(f"{SP}/pls20_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
log(f"20-subject PLS (n_comp={NCOMP}, PCA={PCADIM}, {NFOLD}-fold block CV, PAL {SCALE}); n={len(subs)}")
ref=nib.load(subs[0]); hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"L":mv(hos,"Left Amygdala"),"R":mv(hos,"Right Amygdala"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex")}
av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz"); F0={"emonet":av["E"],"vit":av["VT"],"resnet":av["RN"],"lowlevel":old["LL"],"random":old["RC"]}
def pca(X,k):
    X=X-X.mean(0); 
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(F,T):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6); R=pca(R,PCADIM); return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float32)
def blocks(n,k): e=np.linspace(0,n,k+1).astype(int); return [np.arange(e[i],e[i+1]) for i in range(k)]
def r2set(feat,keys,y):
    X=np.concatenate([feat[k] for k in keys],1); n=X.shape[0]; fo=blocks(n,NFOLD); pred=np.zeros_like(y)
    nc=min(NCOMP,X.shape[1])
    for i in range(NFOLD):
        te=fo[i]; tr=np.concatenate([fo[j] for j in range(NFOLD) if j!=i])
        pls=PLSRegression(n_components=nc,scale=False); pls.fit(X[tr],y[tr]); pred[te]=pls.predict(X[te])
    ss=((y-pred)**2).sum(0); st=((y-y.mean(0))**2).sum(0); return float(np.nanmean(1-ss/np.where(st<1e-9,np.nan,st)))
res={k:[] for k in ["FUS_emo","L_emo","R_emo","L_uel","R_uel","L_uei","R_uei"]}
for p in subs:
    sid=p.split('sub-')[1].split('_')[0]; B=np.asarray(nib.load(p).dataobj,dtype=np.float32); T=B.shape[-1]
    reg={k:(lambda ts:((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))(B[M].T) for k,M in masks.items()}; del B
    feat={n:prep(F,T) for n,F in F0.items()}
    res["FUS_emo"].append(r2set(feat,["emonet"],reg["FUS"]))
    for hn,rk in [("L","L"),("R","R")]:
        y=reg[hn]
        res[f"{rk}_emo"].append(r2set(feat,["emonet"],y))
        res[f"{rk}_uel"].append(r2set(feat,["emonet","lowlevel"],y)-r2set(feat,["lowlevel"],y))
        res[f"{rk}_uei"].append(r2set(feat,["emonet","vit","resnet"],y)-r2set(feat,["vit","resnet"],y))
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): FUS<-emo {res['FUS_emo'][-1]:+.4f} | L emo {res['L_emo'][-1]:+.4f} ue_img {res['L_uei'][-1]:+.4f} | R emo {res['R_emo'][-1]:+.4f} ue_img {res['R_uei'][-1]:+.4f}")
def grp(k):
    a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
log(f"\n=== GROUP (n={len(subs)}, one-sample t vs 0) ===")
for k,lab in [("FUS_emo","POS-CTRL Fusiform<-EmoNet"),("L_emo","L-amyg<-EmoNet"),("R_emo","R-amyg<-EmoNet"),
              ("L_uel","L-amyg EmoNet-unique vs low-level"),("R_uel","R-amyg EmoNet-unique vs low-level"),
              ("L_uei","L-amyg EmoNet-unique OVER ImageNet"),("R_uei","R-amyg EmoNet-unique OVER ImageNet")]:
    m,se,t,pp=grp(k); log(f"  {lab:38s} mean {m:+.4f} +/- {se:.4f}  t={t:+.2f} p={pp:.3f}")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
