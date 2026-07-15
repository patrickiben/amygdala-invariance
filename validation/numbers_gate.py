#!/usr/bin/env python3
"""G2 numbers-gate (amygdala manuscript). Deterministic: asserts every headline number in
MANUSCRIPT_amygdala-false-floor_v2.md co-occurs in a committed result log. Analog of the
A False Floor g2_cp_consistency.py. Not an LLM. Exit 0 = all numbers trace."""
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
CHECKS=[("EmoNet-unique over ImageNet +0.0014 p=.0004",["+0.0014","0.0004"]),
 ("movie EmoNet-ResNet primary R p=.17",["Emo-minus-Res","0.169"]),
 ("movie EmoNet-ResNet eroded L p=.22",["Emo-minus-Res","0.223"]),
 ("affect anti-correlate r=-0.26",["corr(emo,a2)","-0.26"]),
 ("movie fusiform Emo-Res +0.0066 p=.003",["+0.0066","0.003"]),
 ("faces param L fear p=.042",["fear-linear","0.042"]),
 ("faces fusiform kragel-resnet +0.070",["kragelEmoNet - resnet","+0.070"]),
 ("single-neuron affect-object -0.043 CI",["-0.0429","95% CI"]),
 ("single-neuron per-session -0.005 p=.66",["per-session","-0.0051"]),
 ("left faceEmo-resnet -0.089 t=-5.04",["-0.089","-5.04"]),
 ("TOST 95% upper +0.0091 82% power",["+0.0091","82%"]),
 ("audio STG +0.0088 p=.010",["STG","+0.0088","0.010"]),
 ("intensity mean t +0.20 fraction 6.3%",["intensity","6.3%","+0.20"]),
 ("confidence LA 10.5% p=.0002",["confidence","10.5%"]),
 ("MFC RT 10.8%",["MFC","RT","10.8%"])]
bad=[n for n,s in CHECKS if not has(s)]
for n,s in CHECKS: print(f"  [{'ok ' if has(s) else 'MISS'}] {n}")
print(f"\nG2 numbers-gate: {len(CHECKS)-len(bad)}/{len(CHECKS)} traced to a log")
sys.exit(1 if bad else 0)
