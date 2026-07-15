import os
import scipy.io as sio, numpy as np
from scipy import stats
S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023"))
mf=np.load(f"{S}/morph_features.npz"); ENC=["faceEmoNet","kragelEmoNet","resnet","vit"]
encRDM={e:1-np.corrcoef(mf[f"lvl_{e}"]) for e in ENC}; tri=np.triu_indices(7,1)
m=sio.loadmat(f"{S}/neuron/FiringRate_Amygdala.mat",struct_as_record=False,squeeze_me=True)
FR=np.atleast_1d(m["FR"]); beh=np.atleast_1d(m["beh"]); vCell=np.atleast_1d(m["vCell"])
tun=[]; sess=[]
for u in range(len(FR)):
    b=beh[int(vCell[u])-1]; tt=np.atleast_1d(b.trialTypes); cl=np.atleast_1d(b.codeL)
    ca=np.atleast_1d(FR[u].countAll).astype(float); base=np.atleast_1d(FR[u].countBaseline).astype(float)
    n=min(len(ca),len(cl),len(tt),len(base)); ca,base,cl,tt=ca[:n],base[:n],cl[:n],tt[:n]; fh=(tt==1); x=ca-base
    vec=np.array([np.nanmean(x[fh&(cl==L)]) if np.any(fh&(cl==L)) else np.nan for L in range(1,8)])
    if np.any(np.isnan(vec)): continue
    tun.append((vec-vec.mean())/(vec.std()+1e-9)); sess.append(int(vCell[u]))
tun=np.array(tun); sess=np.array(sess)
def ao(P):  # affect(best) - object(best) pseudo-pop RSA difference
    nR=1-np.corrcoef(P.T)
    r={e:stats.spearmanr(nR[tri],encRDM[e][tri]).correlation for e in ENC}
    return max(r["faceEmoNet"],r["kragelEmoNet"])-max(r["resnet"],r["vit"])
pt=ao(tun); rng=np.random.RandomState(0)
bd=np.array([ao(tun[rng.randint(0,len(tun),len(tun))]) for _ in range(4000)])
lo90,hi90=np.percentile(bd,[5,95]); lo95,hi95=np.percentile(bd,[2.5,97.5]); se=bd.std()
DELTA=0.073  # smallest affect-specificity the single-neuron test itself detected (MFC); fusiform-BOLD was +0.070
print(f"n_units={len(tun)}  amygdala affect(best)-object(best) = {pt:+.4f}")
print(f"  bootstrap SE={se:.4f} | 90% CI [{lo90:+.4f},{hi90:+.4f}] | 95% CI [{lo95:+.4f},{hi95:+.4f}]")
print(f"  boot-p (two-sided vs 0) = {2*min((bd<=0).mean(),(bd>=0).mean()):.3f}")
print(f"  TOST equivalence margin Delta = +{DELTA:.3f} (MFC-sized affect effect)")
print(f"  95% upper bound {hi95:+.4f}  ->  {'BELOW' if hi95<DELTA else 'NOT below'} Delta  =>  {'amygdala affect-specificity of MFC size is RULED OUT (equivalence established)' if hi95<DELTA else 'cannot exclude an MFC-sized effect'}")
print(f"  sensitivity: a true +{DELTA:.3f} effect would be z={DELTA/se:.1f} SE from 0 (power ~{100*stats.norm.cdf(DELTA/se-1.96):.0f}% at alpha=.05) -> design {'COULD' if DELTA/se>1.96 else 'could NOT reliably'} detect it")
# per-session (primary inference)
svals=[]
for sv in np.unique(sess):
    P=tun[sess==sv]
    if len(P)>=3: svals.append(ao(P))
svals=np.array(svals); t,p=stats.ttest_1samp(svals,0)
slo,shi=np.percentile(svals,[2.5,97.5]) if len(svals)>3 else (np.nan,np.nan)
print(f"\nper-session (n={len(svals)}): affect-object mean {svals.mean():+.4f} (t={t:+.2f} p={p:.3f}); 95% upper {svals.mean()+1.96*svals.std(ddof=1)/np.sqrt(len(svals)):+.4f}")
