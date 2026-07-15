import os, sys, glob, time, subprocess, numpy as np, torch, cv2
SP=os.environ["SP"]; FILM=os.environ.get("FILM",""); FPS=6
DEV="mps" if torch.backends.mps.is_available() else "cpu"
t0=time.time(); log=lambda *a: print(*a, flush=True)
sys.path.insert(0, f"{SP}/toisoul_emonet")
from emonet.models import EmoNet
net=EmoNet(n_expression=8)
sd=torch.load(f"{SP}/toisoul_emonet/pretrained/emonet_8.pth", map_location="cpu")
net.load_state_dict(sd, strict=False); net.eval()
try: net.to(DEV); _=net(torch.zeros(2,3,256,256).to(DEV)); log(f"device={DEV}")
except Exception as e:
    DEV="cpu"; net.to(DEV); log(f"MPS failed ({str(e)[:60]}), using CPU")
buf={}
net.avg_pool_2.register_forward_hook(lambda m,i,o: buf.__setitem__("f", o.flatten(1).detach().cpu().numpy()))
# frames at 6 fps, square 256 (matches demo resize)
fdir=f"{SP}/frames6"
if not os.path.isdir(fdir) or len(glob.glob(f"{fdir}/f*.png"))<1000:
    os.makedirs(fdir, exist_ok=True); assert os.path.exists(FILM), "need FILM"
    log("ffmpeg 6fps 256x256 ..."); subprocess.run(["ffmpeg","-y","-loglevel","error","-i",FILM,"-vf",f"fps={FPS},scale=256:256",f"{fdir}/f%06d.png"], check=True)
frames=sorted(glob.glob(f"{fdir}/f*.png")); log(f"{len(frames)} frames @ {FPS}fps")
F=[]; B=64
for i in range(0,len(frames),B):
    ims=[]
    for fp in frames[i:i+B]:
        im=cv2.imread(fp); im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); ims.append(im)
    x=torch.from_numpy(np.stack(ims)).float().permute(0,3,1,2).to(DEV)/255.0
    with torch.no_grad(): net(x)
    F.append(buf["f"].copy())
    if i%3200==0: log(f"  {i}/{len(frames)} ({time.time()-t0:.0f}s)")
F=np.vstack(F).astype(np.float32); n=(len(F)//FPS)*FPS
A2=F[:n].reshape(-1,FPS,F.shape[1]).mean(1)
np.savez(f"{SP}/feat_affect2.npz", A2=A2)
log(f"DONE feat_affect2.npz A2 {A2.shape} (Toisoul face-EmoNet avg_pool_2 256-d) ({time.time()-t0:.0f}s)")
