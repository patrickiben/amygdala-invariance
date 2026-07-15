import os,sys,glob,re,numpy as np,torch,torch.nn as nn
from PIL import Image
import torchvision.transforms as T, torchvision.models as M
OLD=os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")); S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023")); DEV="mps" if torch.backends.mps.is_available() else "cpu"
tifs=sorted(glob.glob(f"{S}/stimuli/Fear-Happy/*.tif"))
def lvl(fn):  # index by fe-value itself
    return int(re.search(r"fe(\d+)",fn).group(1))
meta=[(os.path.basename(t), lvl(t)) for t in tifs]
print("images:",len(tifs),"| levels:",sorted(set(m[1] for m in meta)))
imgs=[Image.open(t).convert("RGB") for t in tifs]
feats={}
# ---- Toisoul face-EmoNet (256-d avg_pool_2) ----
sys.path.insert(0,f"{OLD}/toisoul_emonet"); from emonet.models import EmoNet
net=EmoNet(n_expression=8)
net.load_state_dict(torch.load(f"{OLD}/toisoul_emonet/pretrained/emonet_8.pth",map_location="cpu"),strict=False)
net.eval()
net=net.to(DEV)
buf={}; net.avg_pool_2.register_forward_hook(lambda m,i,o: buf.__setitem__("f",o.flatten(1).detach().cpu().numpy()))
X=torch.stack([torch.from_numpy(np.asarray(im.resize((256,256))).astype(np.float32)).permute(2,0,1)/255. for im in imgs]).to(DEV)
with torch.no_grad(): net(X)
feats["faceEmoNet"]=buf["f"].copy()
# ---- Kragel EmoNet (4096-d conv_6) ----
sys.path.insert(0,f"{OLD}/emonet"); from models import EmoNet as KEmoNet
ke=KEmoNet(); ke.load_state_dict_from_path(f"{OLD}/emonet/emonet_weights.pt"); ke.eval()
kb={}; ke.conv_6.register_forward_hook(lambda m,i,o: kb.__setitem__("f",o.flatten(1).cpu().numpy()))
tf227=T.Compose([T.Resize((227,227)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
with torch.no_grad(): ke(torch.stack([tf227(im) for im in imgs]))
feats["kragelEmoNet"]=kb["f"].copy()
# ---- ResNet50 + ViT (object) ----
tf224=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
xb=torch.stack([tf224(im) for im in imgs]).to(DEV)
rn=M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2); rn.fc=nn.Identity(); rn.eval().to(DEV)
vit=M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1); vit.heads=nn.Identity(); vit.eval().to(DEV)
with torch.no_grad(): feats["resnet"]=rn(xb).cpu().numpy(); feats["vit"]=vit(xb).cpu().numpy()
# ---- low-level ----
ll=[]
for im in imgs:
    g=np.asarray(im.convert("L").resize((64,64)),np.float32)/255.; gx=np.diff(g,axis=0); gy=np.diff(g,axis=1)
    ll.append([g.mean(),g.std(),np.abs(gx).mean(),np.abs(gy).mean()])
feats["lowlevel"]=np.array(ll,np.float32)
# ---- average to 7 levels ----
levels=np.array([m[1] for m in meta]); order=sorted(set(levels))
lvlfeats={k:np.stack([feats[k][levels==L].mean(0) for L in order]) for k in feats}
np.savez(f"{S}/morph_features.npz", order=np.array(order), **{f"lvl_{k}":lvlfeats[k] for k in lvlfeats}, **{f"img_{k}":feats[k] for k in feats}, names=np.array([m[0] for m in meta]))
print("saved morph_features.npz | per-level dims:", {k:lvlfeats[k].shape for k in lvlfeats})
