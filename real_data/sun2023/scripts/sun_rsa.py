import os,glob,re,numpy as np,nibabel as nib,scipy.io as sio,time
from nilearn import datasets,image
from nilearn.glm.first_level import FirstLevelModel
from scipy import stats
from scipy.spatial.distance import squareform
S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023")); TR=2.0; t0=time.time()
OUT=open(f"{S}/sun_rsa_result.txt","w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()
FE=[0,30,40,50,60,70,100]                 # ascending fe order (matches encoder lvl_*)
R2F={1:100,2:70,3:60,4:50,5:40,6:30,7:0}  # fear-rank (1=most fear) -> fe value  (step2: 1-7 = decreasing fear)
mf=np.load(f"{S}/morph_features.npz"); ENC=["faceEmoNet","kragelEmoNet","resnet","vit","lowlevel"]
def rdm(M):  # correlation-distance RDM over rows (7 levels)
    C=np.corrcoef(M); return 1-C
encRDM={k:rdm(mf[f"lvl_{k}"]) for k in ENC}
subs=sorted(glob.glob(f"{S}/fmri/sub*/"), key=lambda p:int(re.search(r"sub(\d+)",p).group(1)))
subs=[p for p in subs if os.path.exists(os.path.join(p,"face_emotion_para3","SPM.mat"))]
grid=nib.load(f"{subs[0]}/face_emotion_para3/beta_0001.nii")
hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name); m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("f4")),grid,interpolation="nearest",copy_header=True); return np.asarray(m.dataobj)>0.5
ROIS={"L-amyg":mv(hos,"Left Amygdala"),"R-amyg":mv(hos,"Right Amygdala"),"Fusiform":mv(hoc,"Temporal Occipital Fusiform Cortex"),"V1":mv(hoc,"Occipital Pole")}
log(f"n_subj={len(subs)} | ROIs={ {k:int(v.sum()) for k,v in ROIS.items()} }")
import pandas as pd
res={f"{r}|{e}":[] for r in ROIS for e in ENC}
for sp in subs:
    sid=re.search(r"sub(\d+)",sp).group(1)
    SPM=sio.loadmat(f"{sp}/face_emotion_para3/SPM.mat",struct_as_record=False,squeeze_me=True)["SPM"]
    U=np.atleast_1d(SPM.Sess.U); face=[u for u in U if str(np.atleast_1d(u.name)[0]).lower().startswith("face")][0]
    ons=np.atleast_1d(face.ons); P=np.atleast_1d(face.P); fear=np.atleast_1d([p for p in P if str(p.name).lower()=="fear"][0].P)
    fe=np.array([R2F[int(round(f))] for f in fear])
    ev=pd.DataFrame({"onset":ons,"duration":0.0,"trial_type":[f"fe{v}" for v in fe]})
    vols=sorted(glob.glob(f"{sp}/sw*.img")); arr=np.stack([np.asarray(nib.load(v).dataobj) for v in vols],-1).astype(np.float32)
    img4=nib.Nifti1Image(arr,grid.affine); rp=np.loadtxt(glob.glob(f"{sp}/rp_*.txt")[0]); conf=pd.DataFrame(rp,columns=[f"m{i}" for i in range(rp.shape[1])])
    flm=FirstLevelModel(t_r=TR,hrf_model="spm",drift_model="cosine",high_pass=1/128.,minimize_memory=True,n_jobs=1)
    flm.fit(img4,events=ev,confounds=conf)
    betas=[]
    for v in FE:
        b=flm.compute_contrast(f"fe{v}",output_type="effect_size"); betas.append(np.asarray(b.dataobj))
    B=np.stack(betas)  # (7, X,Y,Z)
    for rn,m in ROIS.items():
        pat=B[:,m]  # 7 x nvox
        pat=(pat-pat.mean(1,keepdims=True))/(pat.std(1,keepdims=True)+1e-6)
        nR=rdm(pat)
        tri=np.triu_indices(7,1)
        for e in ENC:
            rho=stats.spearmanr(nR[tri],encRDM[e][tri]).correlation
            res[f"{rn}|{e}"].append(rho)
    log(f"  sub{sid} ({time.time()-t0:.0f}s): Lamyg faceEmo {res['L-amyg|faceEmoNet'][-1]:+.3f} resnet {res['L-amyg|resnet'][-1]:+.3f} | Fus faceEmo {res['Fusiform|faceEmoNet'][-1]:+.3f} resnet {res['Fusiform|resnet'][-1]:+.3f}")
log(f"\n=== GROUP RSA (Spearman neural~encoder RDM, n={len(subs)}) ===")
def g(k): a=np.array(res[k]); t,p=stats.ttest_1samp(a,0); return np.nanmean(a),np.nanstd(a,ddof=1)/np.sqrt(len(a)),t,p
def gd(a,b): d=np.array(res[a])-np.array(res[b]); t,p=stats.ttest_1samp(d,0); return np.nanmean(d),t,p
for rn in ROIS:
    log(f"  [{rn}]")
    for e in ENC:
        m,se,t,p=g(f"{rn}|{e}"); log(f"     {e:14s} RSA {m:+.3f}+/-{se:.3f} (p{p:.3f})")
    dfe,tfe,pfe=gd(f"{rn}|faceEmoNet",f"{rn}|resnet"); dke,tke,pke=gd(f"{rn}|kragelEmoNet",f"{rn}|resnet")
    log(f"     -> faceEmoNet - resnet = {dfe:+.3f} (t={tfe:+.2f} p={pfe:.3f}) | kragelEmoNet - resnet = {dke:+.3f} (p{pke:.3f})")
np.savez(f"{S}/sun_rsa_res.npz",**{k:np.array(v) for k,v in res.items()})
log(f"\nDONE ({time.time()-t0:.0f}s)"); OUT.close()
