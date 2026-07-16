#!/usr/bin/env python3
"""Dependence-robust Benjamini-Yekutieli (BY) FDR for the two exploratory univariate
parametric contrasts in the face-morph arm (Section 2.2 / Methods 4.9).

Raw p-values trace to sun2023 sun_univar_result.txt: parametric fear intensity (left) p=0.042,
ambiguity (right) p=0.028. m=2, harmonic constant c = sum(1/i, i=1..m) = 1.5.
BY step-up (== statsmodels multipletests method='fdr_by'): q_i = p_(i) * m * c / i, then
enforce monotonicity from the largest rank down (cumulative minimum). Writes univariate_byfdr_result.txt.

This corrects a prior hand-reported pair (q=.10, .08) that did not reproduce under BY; the qualitative
conclusion (neither contrast survives q<.05) is unchanged."""
import os
import numpy as np

# (name, raw p) from sun_univar_result.txt
TESTS = [("parametric fear intensity (L)", 0.042), ("ambiguity (R)", 0.028)]

names = [t[0] for t in TESTS]
p = np.array([t[1] for t in TESTS], float)
m = len(p)
c = float(np.sum(1.0 / np.arange(1, m + 1)))          # harmonic constant, m=2 -> 1.5
order = np.argsort(p)                                   # ascending
ps = p[order]
ranks = np.arange(1, m + 1)
q_raw = ps * m * c / ranks                              # BY raw
q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]       # monotone from top
q_mono = np.clip(q_mono, 0, 1)
q = np.empty(m); q[order] = q_mono                      # back to input order

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "univariate_byfdr_result.txt")
lines = [f"BY-FDR (Benjamini-Yekutieli) univariate family: m={m}, harmonic c={c:.3f}"]
for i in range(m):
    lines.append(f"  {names[i]:32s} raw p={p[i]:.3f}  BY q={q[i]:.3f}  {'survives' if q[i] < 0.05 else 'does NOT survive'} (q<.05)")
lines.append(f"  -> neither exploratory univariate contrast survives dependence-robust BY-FDR (both q approx {q.max():.2f})")
txt = "\n".join(lines) + "\n"
open(OUT, "w").write(txt)
print(txt, end="")
