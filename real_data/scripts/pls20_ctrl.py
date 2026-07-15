import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]; SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5
OUT=open(f"{SP}/pls20_ctrl_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
log(f"CONTROLS for the +ue_img result | n={len(subs)} | regions: amyg L/R + FUS + V1")
ref=nib.load(subs[0]); hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"L":mv(hos,"Left Amygdala"),"R":mv(hos,"Right Amygdala"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex"),"V1":mv(hoc,"Occipital Pole")}
av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz"); F0={"emonet":av["E"],"vit":av["VT"],"resnet":av["RN"],"lowlevel":old["LL"],"random":old["RC"]}
def pca(X,k):
    X=X-X.mean(0)
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(F,T):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6); R=pca(R,PCADIM); return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float32)
def blocks(n,k): e=np.linspace(0,n,k+1).astype(int); return [np.arange(e[i],e[i+1]) for i in range(k)]
def r2set(feat,keys,y):
    X=np.concatenate([feat[k] for k in keys],1); n=X.shape[0]; fo=blocks(n,NFOLD); pred=np.zeros_like(y); nc=min(NCOMP,X.shape[1])
    for i in range(NFOLD):
        te=fo[i]; tr=np.concatenate([fo[j] for j in range(NFOLD) if j!=i])
        pls=PLSRegression(n_components=nc,scale=False); pls.fit(X[tr],y[tr]); pred[te]=pls.predict(X[te])
    ss=((y-pred)**2).sum(0); st=((y-y.mean(0))**2).sum(0); return float(np.nanmean(1-ss/np.where(st<1e-9,np.nan,st)))
res={}
def add(k,v): res.setdefault(k,[]).append(v)
for p in subs:
    sid=p.split('sub-')[1].split('_')[0]; B=np.asarray(nib.load(p).dataobj,dtype=np.float32); T=B.shape[-1]
    reg={k:(lambda ts:((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))(B[M].T) for k,M in masks.items()}; del B
    feat={n:prep(F,T) for n,F in F0.items()}
    for rk in ["L","R","FUS","V1"]:
        y=reg[rk]; base=r2set(feat,["vit","resnet"],y)
        add(f"{rk}_emoOVERimg", r2set(feat,["emonet","vit","resnet"],y)-base)     # EmoNet unique over ImageNet
        add(f"{rk}_randOVERimg", r2set(feat,["random","vit","resnet"],y)-base)     # ARTIFACT control: random over ImageNet
    for rk in ["L","R"]:
        y=reg[rk]; basel=r2set(feat,["vit","resnet","lowlevel"],y)
        add(f"{rk}_emoOVERimglow", r2set(feat,["emonet","vit","resnet","lowlevel"],y)-basel)  # stricter: over ImageNet+lowlevel
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): L emoOimg {res['L_emoOVERimg'][-1]:+.4f} randOimg {res['L_randOVERimg'][-1]:+.4f} | FUS emoOimg {res['FUS_emoOVERimg'][-1]:+.4f}")
def grp(k): a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
log(f"\n=== GROUP CONTROLS (n={len(subs)}, one-sample t vs 0) ===")
log("[region-specificity] EmoNet-unique OVER ImageNet, by region:")
for rk,lab in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform"),("V1","V1/occ")]:
    m,se,t,pp=grp(f"{rk}_emoOVERimg"); log(f"    {lab:10s} mean {m:+.4f} +/- {se:.4f}  t={t:+.2f} p={pp:.4f}")
log("[artifact control] RANDOM-unique OVER ImageNet (should be ~0 if effect is real signal):")
for rk,lab in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform"),("V1","V1/occ")]:
    m,se,t,pp=grp(f"{rk}_randOVERimg"); log(f"    {lab:10s} mean {m:+.4f} +/- {se:.4f}  t={t:+.2f} p={pp:.4f}")
log("[stricter baseline] EmoNet-unique OVER ImageNet+low-level (amygdala):")
for rk,lab in [("L","L-amyg"),("R","R-amyg")]:
    m,se,t,pp=grp(f"{rk}_emoOVERimglow"); log(f"    {lab:10s} mean {m:+.4f} +/- {se:.4f}  t={t:+.2f} p={pp:.4f}")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
