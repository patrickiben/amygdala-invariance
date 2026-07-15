import os
import glob, numpy as np, nibabel as nib
from nilearn import datasets, image
S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023"))
grid=nib.load(f"{S}/fmri/sub1/face_emotion_para3/beta_0001.nii")
print("EPI grid:", grid.shape, "| voxsize:", np.round(grid.header.get_zooms()[:3],2), "| affine diag:", np.round(np.diag(grid.affine)[:3],1))
hos=datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"); hoc=datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
def mv(atl,name):
    idx=list(atl["labels"]).index(name)
    m=image.resample_to_img(image.new_img_like(atl["maps"],(np.asarray(atl["maps"].dataobj)==idx).astype("f4")),grid,interpolation="nearest",copy_header=True)
    return np.asarray(m.dataobj)>0.5
masks={"L-amyg":mv(hos,"Left Amygdala"),"R-amyg":mv(hos,"Right Amygdala"),"Fusiform":mv(hoc,"Temporal Occipital Fusiform Cortex"),"V1":mv(hoc,"Occipital Pole")}
for k,m in masks.items(): print(f"  {k}: {int(m.sum())} vox in atlas ROI")
print("\n=== per-subject coverage (SPM mask) + tSNR (first 120 vols) ===")
for sub in ["sub1","sub2","sub3","sub8","sub15"]:
    try:
        spmmask=np.asarray(nib.load(f"{S}/fmri/{sub}/face_emotion_para3/mask.nii").dataobj)>0.5
        vols=sorted(glob.glob(f"{S}/fmri/{sub}/sw*.img"))[:120]
        arr=np.stack([np.asarray(nib.load(v).dataobj) for v in vols],-1).astype(np.float32)
        tsnr=arr.mean(-1)/(arr.std(-1)+1e-6)
        row=[]
        for k,m in masks.items():
            cov=100*(spmmask&m).sum()/max(m.sum(),1)
            ts=float(np.nanmedian(tsnr[m]))
            row.append(f"{k}: cov {cov:3.0f}% tSNR {ts:5.1f}")
        print(f"  {sub}: "+" | ".join(row))
    except Exception as e: print(f"  {sub}: ERR {str(e)[:80]}")
