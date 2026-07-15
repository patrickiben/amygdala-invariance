import os,glob,re,numpy as np,nibabel as nib,scipy.io as sio,time,pandas as pd
from nilearn import datasets,image
from nilearn.glm.first_level import FirstLevelModel
from sklearn.linear_model import Ridge
from scipy import stats
S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023")); TR=2.0; t0=time.time(); OUT=open(f"{S}/sun_univar_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
FE=[0,30,40,50,60,70,100]; R2F={1:100,2:70,3:60,4:50,5:40,6:30,7:0}
mf=np.load(f"{S}/morph_features.npz"); ENC=["faceEmoNet","kragelEmoNet","resnet","vit","lowlevel"]
subs=sorted(glob.glob(f"{S}/fmri/sub*/"),key=lambda p:int(re.search(r"sub(\d+)",p).group(1)))
subs=[p for p in subs if os.path.exists(os.path.join(p,"face_emotion_para3","SPM.mat"))]
grid=nib.load(f"{subs[0]}/face_emotion_para3/beta_0001.nii")
hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("f4")),grid,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
ROIS={"L-amyg":mv(hos,"Left Amygdala"),"R-amyg":mv(hos,"Right Amygdala"),"Fusiform":mv(hoc,"Temporal Occipital Fusiform Cortex"),"V1":mv(hoc,"Occipital Pole")}
def pca(X,k):
    X=X-X.mean(0); 
    if X.shape[1]<=k: return X
    _,_,Vt=np.linalg.svd(X,full_matrices=False); return X@Vt[:k].T
def loo_r2(Xf,y):  # leave-one-level-out ridge R^2 over 7 pts
    n=len(y); pred=np.zeros(n)
    for i in range(n):
        tr=[j for j in range(n) if j!=i]; r=Ridge(alpha=1.0).fit(Xf[tr],y[tr]); pred[i]=r.predict(Xf[i:i+1])[0]
    ss=((y-pred)**2).sum(); st=((y-y.mean())**2).sum(); return 1-ss/(st+1e-9)
prof={r:[] for r in ROIS}; enc_r2={f"{r}|{e}":[] for r in ROIS for e in ENC}
for sp in subs:
    sid=re.search(r"sub(\d+)",sp).group(1)
    SPM=sio.loadmat(f"{sp}/face_emotion_para3/SPM.mat",struct_as_record=False,squeeze_me=True)["SPM"]
    U=np.atleast_1d(SPM.Sess.U); face=[u for u in U if str(np.atleast_1d(u.name)[0]).lower().startswith("face")][0]
    ons=np.atleast_1d(face.ons); P=np.atleast_1d(face.P); fear=np.atleast_1d([p for p in P if str(p.name).lower()=="fear"][0].P)
    fe=np.array([R2F[int(round(f))] for f in fear])
    ev=pd.DataFrame({"onset":ons,"duration":0.0,"trial_type":[f"fe{v}" for v in fe]})
    vols=sorted(glob.glob(f"{sp}/sw*.img")); arr=np.stack([np.asarray(nib.load(v).dataobj) for v in vols],-1).astype(np.float32)
    img4=nib.Nifti1Image(arr,grid.affine); rp=np.loadtxt(glob.glob(f"{sp}/rp_*.txt")[0]); conf=pd.DataFrame(rp,columns=[f"m{i}" for i in range(rp.shape[1])])
    flm=FirstLevelModel(t_r=TR,hrf_model="spm",drift_model="cosine",high_pass=1/128.,minimize_memory=True).fit(img4,events=ev,confounds=conf)
    B=np.stack([np.asarray(flm.compute_contrast(f"fe{v}",output_type="effect_size").dataobj) for v in FE])  # (7,X,Y,Z)
    for rn,m in ROIS.items():
        a=B[:,m].mean(1)  # 7-level mean response
        prof[rn].append(a); az=(a-a.mean())/(a.std()+1e-9)
        for e in ENC:
            enc_r2[f"{rn}|{e}"].append(loo_r2(pca(mf[f"lvl_{e}"],3),az))
    log(f"  sub{sid} ({time.time()-t0:.0f}s): L-amyg profile(z)~ faceEmo R2 {enc_r2['L-amyg|faceEmoNet'][-1]:+.2f} resnet {enc_r2['L-amyg|resnet'][-1]:+.2f}")
# parametric replication: amygdala mean profile vs fear-intensity(|fe-? use fe as fear%: fe100=max fear) linear + ambiguity(-(fe-50)^2)
feA=np.array(FE,float); lin=(feA-feA.mean())/feA.std(); amb=-((feA-50)**2); amb=(amb-amb.mean())/amb.std()
log(f"\n=== PARAMETRIC replication (profile ~ fear-linear + ambiguity), group t (n={len(subs)}) ===")
for rn in ROIS:
    Pmat=np.array(prof[rn])  # (nsub,7)
    bl=[]; ba=[]
    for a in Pmat:
        X=np.c_[np.ones(7),lin,amb]; c=np.linalg.lstsq(X,(a-a.mean()),rcond=None)[0]; bl.append(c[1]); ba.append(c[2])
    tl,pl=stats.ttest_1samp(bl,0); ta,pa=stats.ttest_1samp(ba,0)
    log(f"  {rn:9s} fear-linear beta {np.mean(bl):+.3f}(t{tl:+.2f} p{pl:.3f}) | ambiguity beta {np.mean(ba):+.3f}(t{ta:+.2f} p{pa:.3f})")
log(f"\n=== UNIVARIATE ENCODING: amygdala profile predicted by each encoder (LOO R^2, group) ===")
def g(k): a=np.array(enc_r2[k]); t,p=stats.ttest_1samp(a,0); return np.mean(a),t,p
def gd(a,b): d=np.array(enc_r2[a])-np.array(enc_r2[b]); t,p=stats.ttest_1samp(d,0); return np.mean(d),t,p
for rn in ROIS:
    log(f"  [{rn}] "+" | ".join(f"{e} {g(f'{rn}|{e}')[0]:+.3f}(p{g(f'{rn}|{e}')[2]:.2f})" for e in ENC))
    d1=gd(f"{rn}|faceEmoNet",f"{rn}|resnet"); d2=gd(f"{rn}|kragelEmoNet",f"{rn}|resnet")
    log(f"       faceEmo-resnet {d1[0]:+.3f}(t{d1[1]:+.2f} p{d1[2]:.3f}) | kragel-resnet {d2[0]:+.3f}(p{d2[2]:.3f})")
np.savez(f"{S}/sun_univar_res.npz",prof={r:np.array(prof[r]) for r in ROIS},**{k:np.array(v) for k,v in enc_r2.items()})
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
