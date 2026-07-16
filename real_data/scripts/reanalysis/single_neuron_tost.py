#!/usr/bin/env python3
"""Single-neuron one-sided non-superiority (TOST-style) bound, corrected margin.

Inputs are the committed bootstrap-over-units outputs for the 442-unit amygdala
affect(best)-object(best) contrast (see amyg_neuron_rsa_result.txt / amyg_neuron_tost_result.txt):
  effect  = -0.0831   (unit-bootstrap point estimate; the best-minus-best headline is -0.086)
  SE      =  0.0391   (unit-bootstrap standard deviation)
  ci95    = [-0.1558, -0.0065]
  persess_upper = +0.0063  (per-session 95% upper limit; primary frame)
The equivalence margin is the affect effect the SAME instrument detects in the MFC comparison
region. The re-run MFC affect(best)-object(best) is +0.100 (amyg_neuron_rsa_result.txt), so the
margin is DELTA=+0.100. (An earlier version of this script hardcoded the prior-run value +0.073;
the amygdala upper bound falls far below either, and the power figure is moot because the observed
effect is significantly negative.)

Deterministic recompute (no data load); writes single_neuron_tost_result.txt."""
import os
from scipy.stats import norm

EFFECT       = -0.0831
SE           =  0.0391
CI95         = (-0.1558, -0.0065)
PERSESS_UP   =  0.0063
DELTA        =  0.100          # re-run MFC affect(best)-object(best)
DELTA_PRIOR  =  0.073          # prior-run MFC benchmark, for reference

def power(delta, se):          # descriptive: P(detect a true +delta at alpha=.05 two-sided)
    return float(norm.cdf(delta / se - 1.96))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_neuron_tost_result.txt")
lines = [
    "n_units=442  amygdala affect(best)-object(best) = %+.4f" % EFFECT,
    "  bootstrap SE=%.4f | 95%% CI [%+.4f,%+.4f] | boot-p (two-sided vs 0) = 0.033" % (SE, CI95[0], CI95[1]),
    "  TOST equivalence margin Delta = %+.3f (re-run MFC-sized affect effect; prior-run benchmark was %+.3f)" % (DELTA, DELTA_PRIOR),
    "  unit-bootstrap 95%% upper bound %+.4f  ->  BELOW Delta  =>  MFC-sized affect-specificity RULED OUT" % CI95[1],
    "  per-session 95%% upper bound %+.4f  ->  BELOW Delta (primary frame)" % PERSESS_UP,
    "  observed effect is significantly NEGATIVE (boot-p=.033): non-superiority is doubly satisfied",
    "  descriptive power vs Delta=%+.3f: %.0f%%  (vs prior benchmark %+.3f: %.0f%%) -- MOOT (effect is significantly negative)"
        % (DELTA, 100 * power(DELTA, SE), DELTA_PRIOR, 100 * power(DELTA_PRIOR, SE)),
]
txt = "\n".join(lines) + "\n"
open(OUT, "w").write(txt)
print(txt, end="")
