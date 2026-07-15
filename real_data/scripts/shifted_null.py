import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]; SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5
OUT=open(f"{SP}/shifted_null_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
log(f"TEMPORAL-NULL control: misaligned-EmoNet (same stats, broken time) vs aligned | n={len(subs)}")
ref=nib.load(subs[0]); hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"L":mv(hos,"Left Amygdala"),"R":mv(hos,"Right Amygdala"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex")}
av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz"); F0={"emonet":av["E"],"vit":av["VT"],"resnet":av["RN"]}
def pca(X,k):
    X=X-X.mean(0)
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(F,T):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6); R=pca(R,PCADIM); return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float32)
def blocks(n,k): e=np.linspace(0,n,k+1).astype(int); return [np.arange(e[i],e[i+1]) for i in range(k)]
def r2mat(X,y):
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
    E=prep(F0["emonet"],T); VR=np.concatenate([prep(F0["vit"],T),prep(F0["resnet"],T)],1); Tn=E.shape[0]
    shifts=[Tn//4,Tn//2,3*Tn//4]
    for rk in ["L","R","FUS"]:
        y=reg[rk]; base=r2mat(VR,y)
        add(f"{rk}_real", r2mat(np.concatenate([E,VR],1),y)-base)
        nulls=[r2mat(np.concatenate([np.roll(E,s,axis=0),VR],1),y)-base for s in shifts]
        add(f"{rk}_null", float(np.mean(nulls)))
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): L real {res['L_real'][-1]:+.4f} null {res['L_null'][-1]:+.4f} | R real {res['R_real'][-1]:+.4f} null {res['R_null'][-1]:+.4f} | FUS real {res['FUS_real'][-1]:+.4f} null {res['FUS_null'][-1]:+.4f}")
def grp1(k): a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
def grpd(kr,kn): a=np.array(res[kr])-np.array(res[kn]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
log(f"\n=== TEMPORAL-NULL RESULT (n={len(subs)}) ===")
for rk,lab in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform")]:
    m,se,t,pp=grp1(f"{rk}_real"); mn,sn,tn,pn=grp1(f"{rk}_null"); md,sd,td,pd=grpd(f"{rk}_real",f"{rk}_null")
    log(f"  {lab:9s} aligned {m:+.4f}(p{pp:.3f}) | misaligned-null {mn:+.4f}(p{pn:.3f}) | aligned-minus-null {md:+.4f} t={td:+.2f} p={pd:.4f}")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
