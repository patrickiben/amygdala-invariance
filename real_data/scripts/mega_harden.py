import os, glob, numpy as np, nibabel as nib, sys, time, cv2
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]; SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5; GAP=8; MRAND=20
OUT=open(f"{SP}/mega_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
# face regressors
Tmax=6000; fc=np.zeros(Tmax,np.float32)
for ln in open(f"{SP}/500dos_face.1D").read().splitlines():
    pp=ln.replace(":"," ").replace(","," ").split()
    if len(pp)>=2:
        try: on=float(pp[0]); du=float(pp[1]); a=int(round(on)); b=int(round(on+du)); fc[a:min(b,Tmax)]=1.0
        except ValueError: pass
rf=np.load(f"{SP}/rich_face.npz"); RFcnt=rf["cnt"]; RFarea=rf["area"]
# pixel matrix for random encoders
frames=sorted(glob.glob(f"{SP}/frames/f*.png")); Pm=np.zeros((len(frames),3072),np.float32)
for i,fp in enumerate(frames):
    im=cv2.imread(fp); 
    if im is not None: Pm[i]=cv2.resize(im,(32,32)).astype(np.float32).reshape(-1)/255.
log(f"mega-hardening | face72%={100*fc.mean():.0f}% richface23% | pixmat {Pm.shape} | MRAND={MRAND} GAP={GAP} | n=20")
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
ref=nib.load(subs[0])
hos5=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"Le":mv(hos5,"Left Amygdala"),"Re":mv(hos5,"Right Amygdala"),"Lh":mv(hos5,"Left Hippocampus"),"Rh":mv(hos5,"Right Hippocampus"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex")}
log("mask voxels (thr50): "+" ".join(f"{k}={int(m.sum())}" for k,m in masks.items()))
av=np.load(f"{SP}/feat_avg_cache.npz")
def pca(X,k):
    X=X-X.mean(0)
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(F,T,doPCA=True):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6)
    if doPCA: R=pca(R,PCADIM)
    return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float32)
def col(v,T):
    x=convolve_hrf(v[:T].reshape(-1,1).astype(np.float32).copy(),1.0); return ((x-x.mean(0))/(x.std(0)+1e-6)).astype(np.float32)
def blocks(n,k): e=np.linspace(0,n,k+1).astype(int); return [np.arange(e[i],e[i+1]) for i in range(k)]
def _fit(X,y,idxfn):
    n=X.shape[0]; fo=blocks(n,NFOLD); pred=np.zeros_like(y); nc=min(NCOMP,X.shape[1])
    for i in range(NFOLD):
        te=fo[i]; tr=idxfn(fo,i,n)
        pls=PLSRegression(n_components=nc,scale=False); pls.fit(X[tr],y[tr]); pred[te]=pls.predict(X[te])
    ss=((y-pred)**2).sum(0); st=((y-y.mean(0))**2).sum(0); return float(np.nanmean(1-ss/np.where(st<1e-9,np.nan,st)))
def r2std(X,y): return _fit(X,y,lambda fo,i,n: np.concatenate([fo[j] for j in range(NFOLD) if j!=i]))
def r2p(X,y):  # purged: drop GAP TRs around test block from train
    def idx(fo,i,n):
        te=fo[i]; lo,hi=te[0],te[-1]; m=np.ones(n,bool); m[max(0,lo-GAP):min(n,hi+1+GAP)]=False; return np.where(m)[0]
    return _fit(X,y,idx)
res={}
def add(k,v): res.setdefault(k,[]).append(v)
for p in subs:
    sid=p.split('sub-')[1].split('_')[0]; B=np.asarray(nib.load(p).dataobj,dtype=np.float32); T=B.shape[-1]
    reg={k:(lambda ts:((ts-ts.mean(0))/(ts.std(0)+1e-6)).astype(np.float32))(B[M].T) for k,M in masks.items()}; del B
    E=prep(av["E"],T); V=prep(av["VT"],T); RN=prep(av["RN"],T); VR=np.concatenate([V,RN],1)
    F72=col(fc,T); FC=col(RFcnt,T); FA=col(RFarea,T); FACES=np.concatenate([F72,FC,FA],1)
    Tn=E.shape[0]; shifts=[Tn//4,Tn//2,3*Tn//4]
    rnd=[prep(Pm@np.random.default_rng(1000+m).standard_normal((3072,64)).astype(np.float32),T) for m in range(MRAND)]
    for rk in masks:
        y=reg[rk]
        bv=r2p(V,y); add(f"{rk}_emoOvit_p", r2p(np.concatenate([E,V],1),y)-bv); add(f"{rk}_resOvit_p", r2p(np.concatenate([RN,V],1),y)-bv)
    for rk in ["Le","Re"]:
        y=reg[rk]
        bVR=r2p(VR,y); add(f"{rk}_real_p", r2p(np.concatenate([E,VR],1),y)-bVR)
        add(f"{rk}_null_p", float(np.mean([r2p(np.concatenate([np.roll(E,s,0),VR],1),y)-bVR for s in shifts])))
        bF=r2p(np.concatenate([VR,FACES],1),y); add(f"{rk}_emoOVERrichface", r2p(np.concatenate([E,VR,FACES],1),y)-bF)
        b0=r2std(VR,y); add(f"{rk}_emoOVERimg", r2std(np.concatenate([E,VR],1),y)-b0)
        add(f"{rk}_randnull", [r2std(np.concatenate([rnd[m],VR],1),y)-b0 for m in range(MRAND)])
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): Le emoOvit {res['Le_emoOvit_p'][-1]:+.4f} resOvit {res['Le_resOvit_p'][-1]:+.4f} | real {res['Le_real_p'][-1]:+.4f} null {res['Le_null_p'][-1]:+.4f} | Lh emoOvit {res['Lh_emoOvit_p'][-1]:+.4f} resOvit {res['Lh_resOvit_p'][-1]:+.4f}")
def g(k): a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
def gd(k1,k2): a=np.array(res[k1])-np.array(res[k2]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
log(f"\n=== [H1/H2] PURGED-CV + ERODED(thr50) mask: 2nd-object swap (EmoNet vs ResNet over ViT) ===")
for rk,l in [("Le","L-amyg(eroded)"),("Re","R-amyg(eroded)"),("Lh","L-hipp(bleed-ctrl)"),("Rh","R-hipp(bleed-ctrl)"),("FUS","Fusiform(ref)")]:
    me,_,_,pe=g(f"{rk}_emoOvit_p"); mr,_,_,pr=g(f"{rk}_resOvit_p"); md,_,td,pd=gd(f"{rk}_emoOvit_p",f"{rk}_resOvit_p")
    log(f"  {l:20s} EmoNet-o-ViT {me:+.4f}(p{pe:.3f}) | ResNet-o-ViT {mr:+.4f}(p{pr:.3f}) | Emo-minus-Res {md:+.4f} t={td:+.2f} p={pd:.3f}")
log(f"\n=== [H2] PURGED-CV temporal-null + rich-face (eroded amygdala) ===")
for rk,l in [("Le","L-amyg"),("Re","R-amyg")]:
    m,_,_,pp=g(f"{rk}_real_p"); mn,_,_,pn=g(f"{rk}_null_p"); md,_,td,pd=gd(f"{rk}_real_p",f"{rk}_null_p"); mf,_,_,pf=g(f"{rk}_emoOVERrichface")
    log(f"  {l:9s} aligned {m:+.4f}(p{pp:.3f}) misaligned {mn:+.4f}(p{pn:.3f}) diff p={pd:.4f} | EmoNet-unique over ViT+ResNet+3xFACE {mf:+.4f}(p{pf:.3f})")
log(f"\n=== [H3] EMPIRICAL NULL ({MRAND} random encoders) + TOST artifact bound (eroded amygdala) ===")
for rk,l in [("Le","L-amyg"),("Re","R-amyg")]:
    emo=np.array(res[f"{rk}_emoOVERimg"]); rnda=np.array(res[f"{rk}_randnull"])  # (20, MRAND)
    subj_nullmean=rnda.mean(1); nm=subj_nullmean.mean(); nse=subj_nullmean.std(ddof=1)/np.sqrt(20)
    null_hi=nm+1.96*nse  # upper 95% bound on the artifact
    em=emo.mean(); ese=emo.std(ddof=1)/np.sqrt(20)
    diff=emo-subj_nullmean; td,pd=stats.ttest_1samp(diff,0)
    log(f"  {l:9s} EmoNet-o-img {em:+.4f}+/-{ese:.4f} | empirical-null(rand) {nm:+.4f} (95%-upper {null_hi:+.4f}) | EmoNet-minus-null {diff.mean():+.4f} t={td:+.2f} p={pd:.4f}")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
