#!/usr/bin/env python3
"""G2 numbers-gate (amygdala manuscript). Deterministic, not an LLM. Exit 0 = all numbers trace.

WHAT THIS CHECKS (and what it does NOT): each entry below is a hand-transcribed headline number
from the manuscript paired with a substring set; the gate asserts that substring set co-occurs
within some committed result log under real_data/**. It confirms every headline number is BACKED
BY committed evidence (the number exists in a log). It does NOT open the manuscript and cannot, on
its own, confirm the manuscript quotes the number in the right place or describes its estimator /
CI / sign correctly -- that is a manual/reviewer responsibility. Keep the transcribed entries in
sync with the manuscript by hand.

All four arms were re-run under corrected preprocessing; re-run headline numbers trace to
real_data/scripts/reanalysis/results/*.txt (the same-named pre-correction logs elsewhere are
superseded and carry a SUPERSEDED header)."""
import os, glob, sys
HERE=os.path.dirname(os.path.abspath(__file__)); RD=os.path.join(HERE,"..","real_data")
RE=os.path.abspath(os.path.join(HERE,"..",".."))
def logs():
    d={}
    for f in glob.glob(os.path.join(RD,"**","*.txt"),recursive=True): d[f]=open(f,errors="ignore").read()
    for f in glob.glob(os.path.join(RE,"REAL-RESULT_amygdala-*.md")): d[f]=open(f,errors="ignore").read()
    return d
L=logs()
def has(subs): return any(all(s in t for s in subs) for t in L.values())

# ---- CORRECTED re-run headline numbers (single-neuron anchor + softened movie) ----
REANALYSIS=[
 ("single-neuron amygdala affect-object -0.086 boot-p=.033",["affect(best) - object(best): -0.086","boot-p=0.033"]),
 ("single-neuron LEFT amygdala -0.109 boot-p=.003",["affect(best) - object(best): -0.109","boot-p=0.003"]),
 ("single-neuron RIGHT amygdala +0.021 (n.s.)",["affect(best) - object(best): +0.021"]),
 ("MFC comparison affect-object +0.100 boot-p=.040",["affect(best) - object(best): +0.100","boot-p=0.040"]),
 ("left faceEmoNet-resnet -0.089 t=-5.04",["-0.089 (t-5.04"]),
 ("all-amyg kragelEmoNet-resnet -0.037",["kragelEmoNet-resnet: -0.037"]),
 ("TOST margin +0.100, upper -0.0065, power 72%",["Delta = +0.100","-0.0065","72%"]),
 ("movie EmoNet-unique OVER ImageNet L +0.0022",["L-amyg EmoNet-unique OVER ImageNet","+0.0022"]),
 ("movie EmoNet-minus-ResNet L +0.0009 p=.002",["Emo-Res +0.0009(p0.002)"]),
 ("movie EmoNet-minus-ResNet R +0.0006 p=.003",["Emo-Res +0.0006(p0.003)"]),
 ("movie fusiform EmoNet-minus-ResNet +0.0105",["Emo-Res +0.0105(p0.000)"]),
 ("movie 2nd-affect A2-ResNet L -0.0012 p=.287",["A2-Res -0.0012(p0.287)"]),
 ("movie 2nd-affect A2-ResNet R -0.0024 p=.000",["A2-Res -0.0024(p0.000)"]),
 ("two-encoder convergence L-amyg -0.28",["corr(emo,a2)=-0.28"]),
 ("two-encoder convergence fusiform +0.79",["corr(emo,a2)=+0.79"]),
 ("banded ridge over-regularizes: fusiform +0.0001",["Fusiform (pos. control)","+0.0001"]),
 ("banded ridge L-amyg collapses to -0.0000",["L-amygdala","EmoNet-minus-ResNet -0.0000"]),
 ("noise-ceiling fusiform convergence +0.790 CI [+0.394,+0.965]",["Fusiform","+0.790","[+0.394, +0.965]"]),
 ("noise-ceiling L-amyg -0.278 CI includes 0",["L-amyg","-0.278","[-0.737, +0.100]"]),
 ("faces RSA re-run: fusiform kragel-resnet +0.093 p=.002",["kragelEmoNet - resnet = +0.093","p0.002"]),
 ("faces RSA re-run: amyg kragel-resnet lone-hint uncorroborated p=.72",["kragelEmoNet - resnet = +0.010 (p0.716)"]),
 ("audio re-run: STG audio-over-vision +0.0189 p=.001",["audioAffect-unique-OVER-VISION +0.0189","p0.001"]),
 ("audio re-run: L-amyg audio-over-vision -0.0028 (negative)",["audioAffect-unique-OVER-VISION -0.0028"]),
 ("univariate BY-FDR recompute: both q=0.063 (neither survives)",["BY q=0.063","does NOT survive"]),
]
# ---- Arms NOT re-run this pass (prior committed logs; reported provisional in the manuscript) ----
CARRIED_OVER=[
 ("faces param L fear p=.042",["fear-linear","0.042"]),
 ("faces fusiform kragel-resnet +0.070",["kragelEmoNet - resnet","+0.070"]),
 ("intensity mean t +0.20 fraction 6.3%",["intensity","6.3%","+0.20"]),
 ("confidence LA 10.5% p=.0002",["confidence","10.5%"]),
 ("MFC RT 10.8%",["MFC","RT","10.8%"]),
]
CHECKS=REANALYSIS+CARRIED_OVER
bad=[n for n,s in CHECKS if not has(s)]
print("== corrected re-run ==")
for n,s in REANALYSIS: print(f"  [{'ok ' if has(s) else 'MISS'}] {n}")
print("== not re-run this pass (prior logs) ==")
for n,s in CARRIED_OVER: print(f"  [{'ok ' if has(s) else 'MISS'}] {n}")
print(f"\nG2 numbers-gate: {len(CHECKS)-len(bad)}/{len(CHECKS)} traced to a committed log")
sys.exit(1 if bad else 0)
