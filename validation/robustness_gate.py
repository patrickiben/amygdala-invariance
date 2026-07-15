#!/usr/bin/env python3
"""Robustness gate (amygdala manuscript). Deterministically re-derives the two load-bearing
statistical claims from reported summary stats: (1) BY-FDR over the confirmatory positive-control
family -> all survive; (2) the single-neuron TOST -> 95% upper < margin AND ~82% power. Pure
stdlib (math). Analog of the A False Floor CP property/differential test. Exit 0 = both reproduce."""
import math, sys
def ncdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
ok=True
# (1) BY-FDR (dependence-robust) over the confirmatory positive-control detections
fam={"movie fusiform":0.003,"faces fusiform":0.005,"audio STG":0.010}
p=sorted(fam.values()); m=len(p); c=sum(1.0/i for i in range(1,m+1)); prev=1.0; q={}
for rank in range(m-1,-1,-1):
    val=min(prev,p[rank]*m*c/(rank+1)); prev=val; q[p[rank]]=val
allsurv=all(q[v]<0.05 for v in fam.values())
print(f"  BY-FDR: controls q={[round(q[v],3) for v in fam.values()]} -> {'ALL SURVIVE' if allsurv else 'FAIL'}")
ok &= allsurv
# (2) TOST re-derivation: est=-0.043, 95% CI upper=+0.009, se=0.0256, margin=0.073
se=0.0256; upper=0.009; margin=0.073; power=ncdf(margin/se-1.96)
bounded = upper < margin
print(f"  TOST: 95%-upper {upper:+.3f} < margin {margin:+.3f} -> {'BOUNDED' if bounded else 'FAIL'} | power={power:.2f} (~{round(power*100)}%)")
ok &= bounded and 0.78 < power < 0.86
print(f"\nrobustness gate: {'PASS' if ok else 'FAIL'} (both load-bearing claims re-derive)")
sys.exit(0 if ok else 1)
