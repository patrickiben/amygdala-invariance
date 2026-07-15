import os,glob,numpy as np,nibabel as nib,sys,time
from nilearn import datasets,image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")); SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5
OUT=open(f"{SP}/audio_amyg_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
af=np.load(f"{SP}/audio/audio_features.npz"); av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz")
F={"audioAffect":af["audioAffect"],"audioLow":af["audioLow"],"emonet":av["E"],"vit":av["VT"],"resnet":av["RN"]}
log(f"audio->amygdala | n={len(subs)} | audioAffect {af['audioAffect'].shape} audioLow {af['audioLow'].shape}")
ref=nib.load(subs[0]); hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,names):
    idxs=[list(atl["labels"]).index(n) for n in names]; dat=np.asarray(atl["maps"].dataobj)
    m=image.resample_to_img(image.new_img_like(atl["maps"],np.isin(dat,idxs).astype("f4")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
ROIS={"L-amyg":mv(hos,["Left Amygdala"]),"R-amyg":mv(hos,["Right Amygdala"]),
      "Heschl(A1-ctrl)":mv(hoc,["Heschl's Gyrus (includes H1 and H2)"]),
      "STG":mv(hoc,["Superior Temporal Gyrus, anterior division","Superior Temporal Gyrus, posterior division"]),
      "Fusiform":mv(hoc,["Temporal Occipital Fusiform Cortex"]),"V1":mv(hoc,["Occipital Pole"])}
log("ROI voxels: "+" ".join(f"{k}={int(v.sum())}" for k,v in ROIS.items()))
def pca(X,k):
    X=X-X.mean(0)
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(Fm,T):
    Nm=Fm.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),Fm[:,c]) for c in range(Fm.shape[1])],1).astype(np.float32)
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
    reg={k:(lambda ts:((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))(B[M].T) for k,M in ROIS.items()}; del B
    AA=prep(F["audioAffect"],T); AL=prep(F["audioLow"],T); VIS=np.concatenate([prep(F["emonet"],T),prep(F["vit"],T),prep(F["resnet"],T)],1)
    for rk in ROIS:
        y=reg[rk]
        add(f"{rk}_AA", r2(AA,y)); add(f"{rk}_AL", r2(AL,y)); add(f"{rk}_VIS", r2(VIS,y))
        add(f"{rk}_AAoverAL", r2(np.concatenate([AA,AL],1),y)-r2(AL,y))
        add(f"{rk}_AAoverVIS", r2(np.concatenate([AA,VIS],1),y)-r2(VIS,y))
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): Heschl AA {res['Heschl(A1-ctrl)_AA'][-1]:+.4f} | L-amyg AA {res['L-amyg_AA'][-1]:+.4f} AAoverVIS {res['L-amyg_AAoverVIS'][-1]:+.4f}")
def g(k): a=np.array(res[k]); t,p=stats.ttest_1samp(a,0); return np.mean(a),np.std(a,ddof=1)/np.sqrt(len(a)),t,p
log(f"\n=== AUDIO -> ROIs (n={len(subs)}) ===")
for rk in ROIS:
    aa=g(f"{rk}_AA"); al=g(f"{rk}_AL"); vis=g(f"{rk}_VIS"); aoal=g(f"{rk}_AAoverAL"); aov=g(f"{rk}_AAoverVIS")
    log(f"  {rk:15s} audioAffect {aa[0]:+.4f}(p{aa[3]:.3f}) | audioLow {al[0]:+.4f}(p{al[3]:.3f}) | vision {vis[0]:+.4f}(p{vis[3]:.3f})")
    log(f"      audioAffect-unique/lowAudio {aoal[0]:+.4f}(p{aoal[3]:.3f}) | audioAffect-unique-OVER-VISION {aov[0]:+.4f}(p{aov[3]:.3f})")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
