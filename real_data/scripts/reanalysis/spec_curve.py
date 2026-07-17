#!/usr/bin/env python
"""Specification-curve (multiverse) for the single-neuron anchor verdict (affect(best)-object(best) RSA).
Runs the verdict across the grid of DEFENSIBLE analytic choices (Type E = principled-equivalent, Type U =
uncertain; Type N facets reported separately), per Steegen 2016 / Simonsohn 2020 / Del Giudice-Gangestad 2021.
Shows the negative is stable across the multiverse, and which choices move it. Reuses strong encoders
(DINOv2/CLIP/FER) cached to morph_strong.npz. Writes spec_curve_result.txt + spec_curve.png."""
import os, re, glob, itertools, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from PIL import Image
import scipy.io as sio
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023")); DEV="cpu"
OUT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "spec_curve_result.txt"), "w")
def log(*a): s=" ".join(map(str,a)); print(s,flush=True); OUT.write(s+"\n"); OUT.flush()

# ---- encoders: existing + strong (cache) ----
mf = np.load(f"{S}/morph_features.npz")
lvl = {e: mf[f"lvl_{e}"] for e in ["faceEmoNet","kragelEmoNet","resnet","vit","lowlevel"]}
cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "morph_strong_cache_LOCAL.npz")
if os.path.exists(cache):
    d=np.load(cache); [lvl.__setitem__(k, d[k]) for k in d.files]; log("loaded strong encoders from cache")
else:
    tifs=sorted(glob.glob(f"{S}/stimuli/Fear-Happy/*.tif"))
    levels=np.array([int(re.search(r"fe(\d+)",os.path.basename(t)).group(1)) for t in tifs]); order=sorted(set(levels.tolist()))
    imgs=[Image.open(t).convert("RGB") for t in tifs]; tl=lambda f: np.stack([f[levels==L].mean(0) for L in order])
    from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor, AutoModelForImageClassification
    p=AutoImageProcessor.from_pretrained("facebook/dinov2-base"); m=AutoModel.from_pretrained("facebook/dinov2-base").eval()
    with torch.no_grad(): lvl["dinov2"]=tl(np.array([m(**p(images=im,return_tensors="pt")).last_hidden_state[0,0].numpy() for im in imgs]))
    cp=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"); cm=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    with torch.no_grad(): lvl["clip"]=tl(np.array([cm.get_image_features(**cp(images=im,return_tensors="pt"))[0].numpy() for im in imgs]))
    fp=AutoImageProcessor.from_pretrained("trpakov/vit-face-expression"); fm=AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression",output_hidden_states=True).eval()
    with torch.no_grad(): lvl["ferViT"]=tl(np.array([fm(**fp(images=im,return_tensors="pt")).hidden_states[-1][0,0].numpy() for im in imgs]))
    np.savez(cache, dinov2=lvl["dinov2"], clip=lvl["clip"], ferViT=lvl["ferViT"]); log("extracted + cached strong encoders")

def rdm(M, metric):
    if metric=="corr": return 1-np.corrcoef(M)
    if metric=="spear": return 1-stats.spearmanr(M.T).correlation
    if metric=="euclid":
        from scipy.spatial.distance import squareform, pdist; return squareform(pdist(M))
tri=np.triu_indices(7,1)

# ---- pseudo-populations: pooled / L / R ----
def build(area=None):
    m=sio.loadmat(f"{S}/neuron/FiringRate_Amygdala.mat",struct_as_record=False,squeeze_me=True)
    FR=np.atleast_1d(m["FR"]); beh=np.atleast_1d(m["beh"]); vCell=np.atleast_1d(m["vCell"]); areaCell=np.atleast_1d(m["areaCell"]); tun=[]
    for u in range(len(FR)):
        if area is not None and area not in str(areaCell[u]): continue
        b=beh[int(vCell[u])-1]; tt=np.atleast_1d(b.trialTypes); cl=np.atleast_1d(b.codeL)
        ca=np.atleast_1d(FR[u].countAll).astype(float); ba=np.atleast_1d(FR[u].countBaseline).astype(float)
        n=min(len(ca),len(cl),len(tt),len(ba)); ca,ba,cl,tt=ca[:n],ba[:n],cl[:n],tt[:n]
        fh=(tt==1); x=ca-ba
        v=np.array([np.nanmean(x[fh&(cl==L)]) if np.any(fh&(cl==L)) else np.nan for L in range(1,8)])
        if np.any(np.isnan(v)): continue
        tun.append(v)
    return np.array(tun)
pops={"pooled":build(None),"left":build("L"),"right":build("R")}
log("pseudo-pop sizes: "+" ".join(f"{k}={len(v)}" for k,v in pops.items()))

# ---- multiverse grid ----
AFFECT = {"orig":["faceEmoNet","kragelEmoNet"], "+ferViT":["faceEmoNet","kragelEmoNet","ferViT"]}   # Type U
OBJECT = {"orig":["resnet","vit"], "+strong":["resnet","vit","dinov2","clip"], "+strong+low":["resnet","vit","dinov2","clip","lowlevel"]}  # U/E
BESTOP = {"max":max, "mean":lambda xs: sum(xs)/len(xs)}   # Type E
METRIC = ["corr","spear","euclid"]                        # Type E
NORM   = ["zscore","raw"]                                 # Type E
def spec_value(pop, aset, oset, bop, metric, norm):
    P = pop.copy()
    if norm=="zscore": P = np.array([(r-r.mean())/(r.std()+1e-9) for r in P])
    nR = rdm(P.T, metric)
    enc = {e: stats.spearmanr(nR[tri], rdm(lvl[e], metric)[tri]).correlation for e in set(aset+oset)}
    return bop([enc[a] for a in aset]) - bop([enc[o] for o in oset])

rows=[]
for hemi,pop in [("pooled",pops["pooled"]),("left",pops["left"]),("right",pops["right"])]:
    for (an,aset),(on,oset),(bn,bop),metric,norm in itertools.product(AFFECT.items(),OBJECT.items(),BESTOP.items(),METRIC,NORM):
        try: val=spec_value(pop,aset,oset,bop,metric,norm)
        except Exception: continue
        if not np.isfinite(val): continue   # skip degenerate specs (constant RDM row etc.)
        rows.append((hemi,an,on,bn,metric,norm,val))
vals=np.array([r[6] for r in rows])
log(f"\n=== SPECIFICATION-CURVE ({len(rows)} defensible specs across E/U choices; N-facet=hemisphere) ===")
for hemi in ["pooled","left","right"]:
    hv=np.array([r[6] for r in rows if r[0]==hemi])
    log(f"  {hemi:7s}: n={len(hv)}  median {np.median(hv):+.3f}  range [{hv.min():+.3f},{hv.max():+.3f}]  frac<=0: {np.mean(hv<=0):.0%}")
log(f"  ALL POOLED+LEFT (the anchored claim): frac<=0 = {np.mean(np.array([r[6] for r in rows if r[0] in ('pooled','left')])<=0):.0%}")
# which axis moves it most (variance of mean effect across each axis' levels)
log("  choice sensitivity (spread of median effect across each axis' levels):")
for ax,idx in [("affect",1),("object",2),("bestop",3),("metric",4),("norm",5),("hemisphere",0)]:
    levs={}
    for r in rows: levs.setdefault(r[idx],[]).append(r[6])
    sp=max(np.median(v) for v in levs.values())-min(np.median(v) for v in levs.values())
    log(f"     {ax:11s} spread {sp:.3f}  ({', '.join(f'{k}:{np.median(v):+.3f}' for k,v in levs.items())})")

# ---- figure ----
po=[r for r in rows if r[0] in ("pooled","left")]; po=sorted(po,key=lambda r:r[6]); pv=np.array([r[6] for r in po])
fig,ax=plt.subplots(figsize=(7,3.2)); x=np.arange(len(pv))
ax.axhline(0,color="#bbb",lw=.8); ax.bar(x,pv,width=1.0,color=["#D55E00" if v<=0 else "#0072B2" for v in pv])
ax.set_xlabel(f"defensible specification (sorted; pooled+left, n={len(pv)})"); ax.set_ylabel("affect(best) − object(best)")
ax.set_title(f"Specification-curve: single-neuron verdict stable across the multiverse ({np.mean(pv<=0):.0%} ≤ 0)",fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "spec_curve.png"),dpi=200)
log("\nwrote spec_curve.png"); log("DONE"); OUT.close()
