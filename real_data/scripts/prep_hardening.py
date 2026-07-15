import os, glob, time, numpy as np
SP=os.environ["SP"]; t0=time.time()
# ---- rich face regressor: OpenCV Haar face count + total face-area fraction per frame ----
import cv2
frames=sorted(glob.glob(f"{SP}/frames/f*.png"))
cas=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
N=len(frames); cnt=np.zeros(N,np.float32); area=np.zeros(N,np.float32)
for i,fp in enumerate(frames):
    im=cv2.imread(fp,cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    H,W=im.shape; f=cas.detectMultiScale(im,1.2,4,minSize=(20,20))
    cnt[i]=len(f); area[i]=sum(w*h for (x,y,w,h) in f)/float(H*W) if len(f) else 0.0
    if i%1000==0: print(f"  face {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
np.savez(f"{SP}/rich_face.npz", cnt=cnt, area=area)
print(f"rich-face saved: {N} frames, mean count {cnt.mean():.2f}, mean area {area.mean():.3f}, %frames-with-face {100*(cnt>0).mean():.0f}%")
# ---- attempt a SECOND affect-supervised encoder (best-effort) ----
ok=False
try:
    import torch, timm
    print("timm available:", timm.__version__)
except Exception as e:
    print("timm not available:", str(e)[:80])
print(f"prep done ({time.time()-t0:.0f}s)")
