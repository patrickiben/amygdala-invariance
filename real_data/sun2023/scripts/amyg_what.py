import os
import scipy.io as sio, numpy as np
from scipy import stats
S=os.environ.get("SUN2023_DIR", os.path.join(os.environ.get("AMYG_DATA", os.path.expanduser("~/amyg_data")), "sun2023"))
def zc(x):
    x=np.asarray(x,float); s=np.nanstd(x); return (x-np.nanmean(x))/s if s>0 else x*0
def analyze(path,label,areasel=None):
    m=sio.loadmat(path,struct_as_record=False,squeeze_me=True)
    FR=np.atleast_1d(m["FR"]); beh=np.atleast_1d(m["beh"]); vCell=np.atleast_1d(m["vCell"]); areaCell=np.atleast_1d(m["areaCell"])
    PREDS=["valence","intensity","choice","RT","confidence","correct"]
    T={p:[] for p in PREDS}; cvr={"stimulus":[],"decision":[],"full":[]}
    nunit=0
    for u in range(len(FR)):
        if areasel is not None and str(areaCell[u]) not in areasel: continue
        b=beh[int(vCell[u])-1]
        tt=np.atleast_1d(b.trialTypes).astype(float); cl=np.atleast_1d(b.codeL).astype(float)
        vr=np.atleast_1d(b.vResp).astype(float); rt=np.atleast_1d(b.RT).astype(float)
        cr=np.atleast_1d(b.vCR).astype(float); cc=np.atleast_1d(b.vCorr).astype(float)
        y=np.atleast_1d(FR[u].countAll).astype(float)-np.atleast_1d(FR[u].countBaseline).astype(float)
        n=min(len(y),len(cl),len(tt),len(vr),len(rt),len(cr),len(cc))
        sl=slice(0,n); tt,cl,vr,rt,cr,cc,y=tt[sl],cl[sl],vr[sl],rt[sl],cr[sl],cc[sl],y[sl]
        mask=(tt==1)&~np.isnan(cl)&~np.isnan(vr)&~np.isnan(rt)&~np.isnan(cr)&~np.isnan(cc)&~np.isnan(y)
        if mask.sum()<40: continue
        cl,vr,rt,cr,cc,y=cl[mask],vr[mask],rt[mask],cr[mask],cc[mask],y[mask]
        X=np.c_[np.ones(len(y)), zc(cl-4), zc(np.abs(cl-4)), zc(vr), zc(rt), zc(cr), zc(cc)]  # int + 6 preds
        # OLS + t per predictor
        beta,_,_,_=np.linalg.lstsq(X,y,rcond=None); resid=y-X@beta; dof=len(y)-X.shape[1]
        if dof<5: continue
        sigma2=(resid@resid)/dof; cov=sigma2*np.linalg.pinv(X.T@X); se=np.sqrt(np.diag(cov))
        tv=beta/(se+1e-12)
        for i,p in enumerate(PREDS): T[p].append(tv[i+1])
        # CV model comparison: stimulus(valence,intensity) vs decision(choice,RT,conf,correct) vs full
        cols={"stimulus":[1,2],"decision":[3,4,5,6],"full":[1,2,3,4,5,6]}
        idx=np.arange(len(y)); fo=np.array_split(idx,5)
        for mdl,cs in cols.items():
            Xm=np.c_[np.ones(len(y)),X[:,cs]]; pred=np.zeros(len(y))
            for f in range(5):
                te=fo[f]; tr=np.concatenate([fo[j] for j in range(5) if j!=f])
                bb,_,_,_=np.linalg.lstsq(Xm[tr],y[tr],rcond=None); pred[te]=Xm[te]@bb
            ss=((y-pred)**2).sum(); st=((y-y.mean())**2).sum(); cvr[mdl].append(1-ss/(st+1e-9))
        nunit+=1
    print(f"\n=== {label} (n_units={nunit}) — per-unit tuning (fraction |t|>1.96, signed) ===")
    for p in PREDS:
        t=np.array(T[p]); frac=np.mean(np.abs(t)>1.96); pos=np.mean(t[np.abs(t)>1.96]>0) if np.any(np.abs(t)>1.96) else np.nan
        bp=stats.binomtest(int(np.sum(np.abs(t)>1.96)),len(t),0.05,alternative='greater').pvalue
        mt,pt=stats.ttest_1samp(t,0)
        print(f"   {p:11s}: {100*frac:4.1f}% tuned (binom p={bp:.1e}) | mean t {np.mean(t):+.2f}(p{pt:.3f}) | {100*(pos if not np.isnan(pos) else 0):.0f}% of tuned are +")
    st_=np.array(cvr["stimulus"]); de_=np.array(cvr["decision"]); fu_=np.array(cvr["full"])
    d,pp=stats.ttest_rel(de_,st_)
    print(f"   CV R^2: stimulus {st_.mean():+.4f} | decision {de_.mean():+.4f} | full {fu_.mean():+.4f} | decision-stimulus {de_.mean()-st_.mean():+.4f} (t{d:+.2f} p{pp:.3f})")
analyze(f"{S}/neuron/FiringRate_Amygdala.mat","AMYGDALA (all)")
analyze(f"{S}/neuron/FiringRate_Amygdala.mat","AMYGDALA — LEFT",{"LA"})
analyze(f"{S}/neuron/FiringRate_Amygdala.mat","AMYGDALA — RIGHT",{"RA"})
analyze(f"{S}/neuron/FiringRate_MFC.mat","MFC (decision region, validity contrast)")
