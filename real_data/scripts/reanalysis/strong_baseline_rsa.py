#!/usr/bin/env python
"""G4 STRONG-BASELINE FLOOR / representation-invariance red-team, applied REFLEXIVELY to this
paper's own NEGATIVE claim ("affect encoders match the amygdala single-neuron geometry no better
than generic object/low-level encoders").

The false-floor central law: a negative inherits the weakness of its strongest-FAILING baseline. So
we re-run the amygdala single-neuron RSA with paradigmatically-different STRONGER encoders on BOTH sides:
  - stronger GENERIC:  DINOv2-base (self-distilled) + CLIP ViT-B/32 (contrastive)  [vs the paper's ResNet/ViT]
  - stronger AFFECT :  a modern facial-expression ViT (trpakov/vit-face-expression) [vs EmoNet/Toisoul]
Verdict is ROBUST iff affect(best) still does NOT beat generic(best) even with these stronger baselines.

Reuses the pseudo-population + RDM logic of amyg_neuron_rsa.py. Reads $SUN2023_DIR (default $AMYG_DATA/sun2023),
needs morph_features.npz + neuron/FiringRate_Amygdala.mat + stimuli/Fear-Happy/*.tif. Writes
strong_baseline_result.txt next to this script."""
import os, re, glob, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from PIL import Image
import scipy.io as sio
from scipy import stats
S = os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023"))
DEV = "cpu"
OUT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strong_baseline_result.txt"), "w")
def log(*a): s=" ".join(map(str,a)); print(s, flush=True); OUT.write(s+"\n"); OUT.flush()

tifs = sorted(glob.glob(f"{S}/stimuli/Fear-Happy/*.tif"))
levels = np.array([int(re.search(r"fe(\d+)", os.path.basename(t)).group(1)) for t in tifs])
order = sorted(set(levels.tolist()))
imgs = [Image.open(t).convert("RGB") for t in tifs]
log(f"morph images: {len(tifs)} | levels(order): {order}")
def to_levels(feat): return np.stack([feat[levels == L].mean(0) for L in order])

newfeat = {}
try:
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    m = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEV); F=[]
    with torch.no_grad():
        for im in imgs: F.append(m(**proc(images=im, return_tensors="pt").to(DEV)).last_hidden_state[0,0].cpu().numpy())
    newfeat["dinov2"] = to_levels(np.array(F)); log("  DINOv2-base OK", newfeat["dinov2"].shape); del m
except Exception as e: log("  DINOv2 FAILED:", repr(e)[:120])
try:
    from transformers import CLIPModel, CLIPProcessor
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DEV); F=[]
    with torch.no_grad():
        for im in imgs: F.append(cm.get_image_features(**cp(images=im, return_tensors="pt").to(DEV))[0].cpu().numpy())
    newfeat["clip"] = to_levels(np.array(F)); log("  CLIP ViT-B/32 OK", newfeat["clip"].shape); del cm
except Exception as e: log("  CLIP FAILED:", repr(e)[:120])
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor as AIP
    mid = "trpakov/vit-face-expression"; fp = AIP.from_pretrained(mid)
    fm = AutoModelForImageClassification.from_pretrained(mid, output_hidden_states=True).eval().to(DEV); F=[]
    with torch.no_grad():
        for im in imgs: F.append(fm(**fp(images=im, return_tensors="pt").to(DEV)).hidden_states[-1][0,0].cpu().numpy())
    newfeat["ferViT"] = to_levels(np.array(F)); log("  FER-ViT (affect) OK", newfeat["ferViT"].shape); del fm
except Exception as e: log("  FER-ViT FAILED:", repr(e)[:120])

mf = np.load(f"{S}/morph_features.npz")
lvlfeat = {e: mf[f"lvl_{e}"] for e in ["faceEmoNet","kragelEmoNet","resnet","vit","lowlevel"]}; lvlfeat.update(newfeat)
def rdm(M): return 1 - np.corrcoef(M)
encRDM = {e: rdm(lvlfeat[e]) for e in lvlfeat}; tri = np.triu_indices(7,1)

m = sio.loadmat(f"{S}/neuron/FiringRate_Amygdala.mat", struct_as_record=False, squeeze_me=True)
FR=np.atleast_1d(m["FR"]); beh=np.atleast_1d(m["beh"]); vCell=np.atleast_1d(m["vCell"]); tun=[]
for u in range(len(FR)):
    b=beh[int(vCell[u])-1]; tt=np.atleast_1d(b.trialTypes); cl=np.atleast_1d(b.codeL)
    ca=np.atleast_1d(FR[u].countAll).astype(float); base=np.atleast_1d(FR[u].countBaseline).astype(float)
    n=min(len(ca),len(cl),len(tt),len(base)); ca,base,cl,tt=ca[:n],base[:n],cl[:n],tt[:n]
    fh=(tt==1); x=ca-base
    vec=np.array([np.nanmean(x[fh&(cl==L)]) if np.any(fh&(cl==L)) else np.nan for L in range(1,8)])
    if np.any(np.isnan(vec)): continue
    tun.append((vec-vec.mean())/(vec.std()+1e-9))
tun=np.array(tun); log(f"\namygdala pseudo-pop units: {len(tun)}")

ALL=list(lvlfeat.keys())
def pop_rsa(P):
    nR=rdm(P.T); return {e:stats.spearmanr(nR[tri],encRDM[e][tri]).correlation for e in ALL}
base=pop_rsa(tun); log("\n=== amygdala pseudo-pop RSA (point) ===")
for e in ALL: log(f"   {e:14s} RSA {base[e]:+.3f}")
AFF0=["faceEmoNet","kragelEmoNet"]; OBJ0=["resnet","vit"]
AFFs=AFF0+(["ferViT"] if "ferViT" in newfeat else []); GENs=OBJ0+[e for e in ("dinov2","clip") if e in newfeat]
rng=np.random.RandomState(0); B=2000
def contrast(A,G):
    d=[]
    for _ in range(B):
        r=pop_rsa(tun[rng.randint(0,len(tun),len(tun))])
        d.append(max(r[a] for a in A)-max(r[g] for g in G))
    d=np.array(d); return np.median(d), np.percentile(d,[2.5,97.5]), 2*min((d<=0).mean(),(d>=0).mean())
log("\n=== affect(best) - generic(best): does the negative survive STRONGER baselines? ===")
for name,A,G in [("ORIGINAL   affect{face,kragel} - object{resnet,vit}",AFF0,OBJ0),
    ("+STRONG GEN affect{face,kragel} - generic{resnet,vit,dinov2,clip}",AFF0,GENs),
    ("+STRONG AFF affect{face,kragel,ferViT} - object{resnet,vit}",AFFs,OBJ0),
    ("+BOTH STRONG affect{+ferViT} - generic{+dinov2,clip}",AFFs,GENs)]:
    md,ci,p=contrast(A,G); v="affect still NOT better (negative HOLDS)" if md<=0 or p>=.05 else "affect BEATS generic (FLIPS!)"
    log(f"   {name}\n       = {md:+.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}] boot-p={p:.3f}  -> {v}")
log("\nDONE"); OUT.close()
