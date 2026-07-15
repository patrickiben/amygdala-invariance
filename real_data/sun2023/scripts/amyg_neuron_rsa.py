import os
import scipy.io as sio, numpy as np
from scipy import stats
S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023"))
mf=np.load(f"{S}/morph_features.npz"); ENC=["faceEmoNet","kragelEmoNet","resnet","vit","lowlevel"]
def rdm(M): return 1-np.corrcoef(M)
encRDM={e:rdm(mf[f"lvl_{e}"]) for e in ENC}   # 7x7, fe ascending; codeL mapped ascending (contrast is direction-robust)
tri=np.triu_indices(7,1)
def run(path,label,areasel=None):
    m=sio.loadmat(path,struct_as_record=False,squeeze_me=True)
    FR=np.atleast_1d(m["FR"]); beh=np.atleast_1d(m["beh"]); vCell=np.atleast_1d(m["vCell"]); areaCell=np.atleast_1d(m["areaCell"])
    # per-unit 7-level baseline-corrected mean FR (fear-happy trials only)
    tun=[]; sess_of=[]
    for u in range(len(FR)):
        if areasel is not None and str(areaCell[u]) not in areasel: continue
        b=beh[int(vCell[u])-1]; tt=np.atleast_1d(b.trialTypes); cl=np.atleast_1d(b.codeL)
        ca=np.atleast_1d(FR[u].countAll).astype(float); base=np.atleast_1d(FR[u].countBaseline).astype(float)
        n=min(len(ca),len(cl),len(tt),len(base)); ca,base,cl,tt=ca[:n],base[:n],cl[:n],tt[:n]
        fh=(tt==1); x=ca-base
        vec=np.array([np.nanmean(x[fh&(cl==L)]) if np.any(fh&(cl==L)) else np.nan for L in range(1,8)])
        if np.any(np.isnan(vec)): continue
        v=(vec-vec.mean())/(vec.std()+1e-9)   # tuning shape
        tun.append(v); sess_of.append(int(vCell[u]))
    tun=np.array(tun); sess_of=np.array(sess_of)
    # pseudo-population RDM (all units) + bootstrap over units
    def pop_rsa(P):
        nR=rdm(P.T)   # 7x7 across conditions (cols)
        return {e:stats.spearmanr(nR[tri],encRDM[e][tri]).correlation for e in ENC}
    base_rsa=pop_rsa(tun)
    rng=np.random.RandomState(0); B=2000; boot={e:[] for e in ENC}; bootd=[]
    for _ in range(B):
        idx=rng.randint(0,len(tun),len(tun)); r=pop_rsa(tun[idx])
        for e in ENC: boot[e].append(r[e])
        bootd.append(max(r["faceEmoNet"],r["kragelEmoNet"])-max(r["resnet"],r["vit"]))
    bootd=np.array(bootd)
    print(f"\n=== {label} (n_units={len(tun)}) — pseudo-pop RSA (bootstrap 95% CI over units) ===")
    for e in ENC:
        lo,hi=np.percentile(boot[e],[2.5,97.5]); print(f"   {e:14s} RSA {base_rsa[e]:+.3f} [{lo:+.3f},{hi:+.3f}]")
    p_ao=2*min((bootd<=0).mean(),(bootd>=0).mean())
    print(f"   affect(best) - object(best): {np.median(bootd):+.3f}  boot-p={p_ao:.3f}")
    fe_res=np.array([b["faceEmoNet"] for b in [dict(zip(ENC,[boot[e][i] for e in ENC])) for i in range(B)]])  # not used
    # per-session group test (mirror fMRI per-subject)
    sess=np.unique(sess_of); rows={e:[] for e in ENC}
    for sv in sess:
        P=tun[sess_of==sv]
        if len(P)<3: continue
        nR=rdm(P.T)
        for e in ENC: rows[e].append(stats.spearmanr(nR[tri],encRDM[e][tri]).correlation)
    ns=len(rows[ENC[0]]); 
    def g(e): a=np.array(rows[e]); t,p=stats.ttest_1samp(a,0,nan_policy='omit'); return np.nanmean(a),p
    def gd(a,b): d=np.array(rows[a])-np.array(rows[b]); t,p=stats.ttest_1samp(d,0,nan_policy='omit'); return np.nanmean(d),t,p
    print(f"   per-session group (n_sessions={ns}, >=3 units):")
    for e in ENC: mm,pp=g(e); print(f"      {e:14s} {mm:+.3f} (p{pp:.3f})")
    for pair in [("faceEmoNet","resnet"),("kragelEmoNet","resnet"),("faceEmoNet","vit"),("kragelEmoNet","vit")]:
        dd,tt2,pp=gd(*pair); print(f"      {pair[0]}-{pair[1]}: {dd:+.3f} (t{tt2:+.2f} p{pp:.3f})")
run(f"{S}/neuron/FiringRate_Amygdala.mat","AMYGDALA (all: LA+RA)")
run(f"{S}/neuron/FiringRate_Amygdala.mat","AMYGDALA — LEFT only",{"LA"})
run(f"{S}/neuron/FiringRate_Amygdala.mat","AMYGDALA — RIGHT only",{"RA"})
run(f"{S}/neuron/FiringRate_MFC.mat","MFC (ACC+SMA) — comparison region")
