import os, glob, numpy as np, nibabel as nib, sys, time
from nilearn import datasets, image
from sklearn.cross_decomposition import PLSRegression
from scipy import stats
sys.path.insert(0,next(str(_p) for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p/"amyg_inv").is_dir()))
from amyg_inv.modeling.hemodynamics import convolve_hrf
SP=os.environ["SP"]; SCALE=1.042; NCOMP=10; PCADIM=300; NFOLD=5
OUT=open(f"{SP}/combined_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
t0=time.time()
Tmax=6000; fc=np.zeros(Tmax,np.float32)
for ln in open(f"{SP}/500dos_face.1D").read().splitlines():
    pp=ln.replace(":"," ").replace(","," ").split()
    if len(pp)<2: continue
    try: on=float(pp[0]); du=float(pp[1])
    except ValueError: continue
    a=int(round(on)); b=int(round(on+du)); fc[a:min(b,Tmax)]=1.0
log(f"combined controls | face regressor {int(fc.sum())}s ({100*fc.mean():.0f}%) | n=20")
subs=sorted(glob.glob(f"{SP}/deriv_sub-*_bold.nii.gz"), key=lambda p:int(p.split('sub-')[1].split('_')[0]))
ref=nib.load(subs[0]); hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("float32")),ref,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
masks={"L":mv(hos,"Left Amygdala"),"R":mv(hos,"Right Amygdala"),"FUS":mv(hoc,"Temporal Occipital Fusiform Cortex")}
av=np.load(f"{SP}/feat_avg_cache.npz"); old=np.load(f"{SP}/feat_cache.npz")
def pca(X,k):
    X=X-X.mean(0)
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def prep(F,T):
    Nm=F.shape[0]; mt=np.clip(SCALE*np.arange(T),0,Nm-1)
    R=np.stack([np.interp(mt,np.arange(Nm),F[:,c]) for c in range(F.shape[1])],1).astype(np.float32)
    R=convolve_hrf(R,1.0); R=(R-R.mean(0))/(R.std(0)+1e-6); R=pca(R,PCADIM); return ((R-R.mean(0))/(R.std(0)+1e-6)).astype(np.float32)
def prepface(T):
    f=convolve_hrf(fc[:T].reshape(-1,1).copy(),1.0); return ((f-f.mean(0))/(f.std(0)+1e-6)).astype(np.float32)
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
    E=prep(av["E"],T); V=prep(av["VT"],T); RN=prep(av["RN"],T); LO=prep(old["LL"],T); FA=prepface(T)
    VR=np.concatenate([V,RN],1); VRL=np.concatenate([V,RN,LO],1); VRLF=np.concatenate([VRL,FA],1)
    Tn=E.shape[0]; shifts=[Tn//4,Tn//2,3*Tn//4]
    for rk in ["L","R","FUS"]:
        y=reg[rk]
        # temporal-null
        baseVR=r2(VR,y); add(f"{rk}_real", r2(np.concatenate([E,VR],1),y)-baseVR)
        add(f"{rk}_null", float(np.mean([r2(np.concatenate([np.roll(E,s,0),VR],1),y)-baseVR for s in shifts])))
        # face confound
        b0=r2(VRL,y); add(f"{rk}_emoOVERbase", r2(np.concatenate([E,VRL],1),y)-b0)
        bf=r2(VRLF,y); add(f"{rk}_emoOVERbaseFACE", r2(np.concatenate([E,VRLF],1),y)-bf)
        add(f"{rk}_faceR2", r2(FA,y))
        # 2nd-object
        bv=r2(V,y); add(f"{rk}_emoOVERvit", r2(np.concatenate([E,V],1),y)-bv); add(f"{rk}_resOVERvit", r2(np.concatenate([RN,V],1),y)-bv)
    log(f"  sub-{sid:>2} ({time.time()-t0:.0f}s): L real {res['L_real'][-1]:+.4f} null {res['L_null'][-1]:+.4f} | emoOb {res['L_emoOVERbase'][-1]:+.4f}->+FACE {res['L_emoOVERbaseFACE'][-1]:+.4f} | emoOvit {res['L_emoOVERvit'][-1]:+.4f} resOvit {res['L_resOVERvit'][-1]:+.4f}")
def g(k): a=np.array(res[k]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
def gd(k1,k2): a=np.array(res[k1])-np.array(res[k2]); t,pp=stats.ttest_1samp(a,0); return a.mean(),a.std(ddof=1)/np.sqrt(len(a)),t,pp
log(f"\n=== TEMPORAL-NULL: aligned vs misaligned-EmoNet (n=20) ===")
for rk,l in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform")]:
    m,se,t,pp=g(f"{rk}_real"); mn,sn,tn,pn=g(f"{rk}_null"); md,sd,td,pd=gd(f"{rk}_real",f"{rk}_null")
    log(f"  {l:9s} aligned {m:+.4f}(p{pp:.3f}) | misaligned {mn:+.4f}(p{pn:.3f}) | aligned-minus-misaligned {md:+.4f} t={td:+.2f} p={pd:.4f}")
log(f"\n=== FACE CONFOUND: EmoNet-unique over [ViT+ResNet+low] with vs without FACE in baseline ===")
for rk,l in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform")]:
    mf,sf,tf,pf=g(f"{rk}_faceR2"); m,se,t,pp=g(f"{rk}_emoOVERbase"); m2,s2,t2,p2=g(f"{rk}_emoOVERbaseFACE"); md,sd,td,pd=gd(f"{rk}_emoOVERbase",f"{rk}_emoOVERbaseFACE")
    log(f"  {l:9s} face-only R^2 {mf:+.4f}(p{pf:.3f}) | emoOVERbase {m:+.4f}(p{pp:.3f}) -> +FACE {m2:+.4f}(p{p2:.3f}) | absorbed {md:+.4f}(p{pd:.3f})")
log(f"\n=== 2nd-OBJECT: EmoNet-over-ViT vs ResNet-over-ViT (is a 2nd object encoder as good as EmoNet?) ===")
for rk,l in [("L","L-amyg"),("R","R-amyg"),("FUS","Fusiform")]:
    me,see,te,pe=g(f"{rk}_emoOVERvit"); mr,sr,tr,pr=g(f"{rk}_resOVERvit"); md,sd,td,pd=gd(f"{rk}_emoOVERvit",f"{rk}_resOVERvit")
    log(f"  {l:9s} EmoNet-o-ViT {me:+.4f}(p{pe:.3f}) | ResNet-o-ViT {mr:+.4f}(p{pr:.3f}) | Emo-minus-Res {md:+.4f} t={td:+.2f} p={pd:.4f}")
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
