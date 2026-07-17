import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
from scipy.stats import norm
FIG=os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.spines.top":False,
    "axes.spines.right":False,"axes.linewidth":0.8,"figure.dpi":300})
AMY="#D55E00"; CTL="#0072B2"; GREY="#7f7f7f"  # Okabe-Ito subset (CVD-safe)
def star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
def se_from_p(val,p):
    # representative error bar derived from the reported two-sided p (normal approx);
    # exact per-participant/unit SEMs are in the re-run logs. z = |val|/SE.
    p=min(max(p,1e-6),0.999); z=norm.isf(p/2.0); return abs(val)/z if z>0 else abs(val)
def save(fig,name):
    fig.savefig(f"{FIG}/{name}.png",bbox_inches="tight",dpi=300)
    fig.savefig(f"{FIG}/{name}.pdf",bbox_inches="tight"); plt.close(fig)

# ---- Fig 1: four-arm comparison (affect encoder advantage over object encoder) ----
# CORRECTED (Mac re-run, canonical 0-255 EmoNet preprocessing). All four arms re-run under corrected
# preprocessing (rerun=True for every arm; no arm is provisional/hatched).
# (title, ctl_label, v_amy, v_ctl, p_amy, p_ctl, rerun)
arms=[("Movie fMRI\n(encoding, ΔR²)","fusiform", 0.0009, 0.0105, 0.002, 0.0005, True),
      ("Face morphs\n(RSA, Δρ)","fusiform",        0.010, 0.093,  0.72,  0.002,  True),
      ("Single neurons\n(RSA, Δρ)","MFC",         -0.086, 0.100,  0.033, 0.040,  True),
      ("Audio\n(ΔR² over vision)","STG",          -0.0028,0.0189, 0.001, 0.001,  True)]
fig,axes=plt.subplots(1,4,figsize=(9.4,2.8))
for ax,(title,ctl,va,vc,pa,pc,rerun) in zip(axes,arms):
    sea,sec=se_from_p(va,pa),se_from_p(vc,pc)
    ax.axhline(0,color="#bbb",lw=0.8,zorder=0)
    bars=ax.bar([0,1],[va,vc],width=0.62,color=[GREY,CTL],edgecolor="white",linewidth=1,
           yerr=[sea,sec],error_kw=dict(lw=1,capsize=3,ecolor="#444"),zorder=2)
    if not rerun:
        for b in bars: b.set_hatch("////"); b.set_edgecolor("#999")
    ax.set_xticks([0,1]); ax.set_xticklabels(["amygdala",ctl],fontsize=8)
    ax.set_title(title,fontsize=8.3,pad=6)
    span=max(abs(va)+sea,abs(vc)+sec)
    # sign-aware significance over each bar
    ax.text(0, va+(sea+span*0.08 if va>=0 else -sea-span*0.14), star(pa), ha="center",
            fontsize=8.5, color=(GREY if pa>=.05 else AMY), fontweight="bold")
    ax.text(1, vc+(sec+span*0.08 if vc>=0 else -sec-span*0.14), star(pc), ha="center",
            fontsize=9, color=CTL, fontweight="bold")
    ax.margins(y=0.30)
axes[0].set_ylabel("affect − object encoder")
fig.suptitle("The affect−object advantage is far larger in sensory cortex than in the amygdala, in every arm",
             fontsize=9.6,y=1.05,fontweight="bold")
fig.text(0.5,-0.09,"Sensory-cortex control (blue) shows a substantial affect advantage; the amygdala (grey) is at or near zero — "
         "significantly negative for single neurons and audio, and only tiny non-significant/near-floor positive increments in the movie "
         "(+0.0009, ~10× below fusiform) and face-morph (+0.010, n.s.) arms. All four arms were re-run under corrected preprocessing. Units differ per arm.",
         ha="center",fontsize=6.6,color="#555",wrap=True)
save(fig,"Fig1_four_modality")

# ---- Fig 2: bounded null + significant negative (single-neuron TOST) ----
fig,ax=plt.subplots(figsize=(6.6,2.2))
est,lo,hi,margin,hi_ps=-0.083,-0.156,-0.0065,0.100,0.0063   # equivalence-frame point owns the CI; per-session upper +0.0063
ax.axvspan(0,margin,color="#0072B2",alpha=0.08,zorder=0,label="non-superiority margin (+0.100)")
ax.axvline(0,color="#bbb",lw=0.8)
ax.axvline(margin,color=CTL,lw=1.2,ls="--"); ax.text(margin,1.46,"margin = +0.100\n(medial-frontal effect)",color=CTL,fontsize=7.5,ha="center")
ax.errorbar([est],[1],xerr=[[est-lo],[hi-est]],fmt="o",color=AMY,ms=8,capsize=4,lw=1.6,zorder=3,label="amygdala affect − object (unit boot, n=442)")
ax.axvline(hi_ps,color=AMY,lw=1.1,ls=":"); ax.text(hi_ps,1.44,"per-session\nupper +0.0063",color=AMY,fontsize=7.0,ha="center")
ax.plot([margin],[0.55],"D",color=CTL,ms=7,zorder=3); ax.text(margin,0.30,"medial-frontal\neffect +0.100",color=CTL,fontsize=7.3,ha="center")
ax.annotate("equivalence-frame −0.083 (composite −0.086, boot-p=.033).\n"
            "Per-session: worse than objects on 3/4 contrasts (left p<.001).\n"
            "95% upper (unit −0.0065; per-session +0.0063) far below margin.",
            xy=(est,1),xytext=(-0.150,1.66),fontsize=7.1,color="#333",ha="left",
            arrowprops=dict(arrowstyle="->",color="#888",lw=0.8))
ax.set_yticks([]); ax.set_ylim(0,2.15); ax.set_xlim(-0.175,0.135)
ax.set_xlabel("affect − object encoder (RSA Δρ)")
ax.set_title("The single-neuron amygdala affect−object contrast is bounded far below a detectable effect (negative per-session)",fontsize=8.4,pad=8)
save(fig,"Fig2_TOST_bounded_null")

# ---- Fig 3: two-encoder convergence with reliability CIs ----
fig,ax=plt.subplots(figsize=(4.8,2.6))
regs=["L-amyg","R-amyg","Fusiform"]; rr=[-0.28,-0.00,0.79]
lo=[-0.74,-0.63,0.39]; hi=[0.10,0.59,0.97]; cols=[AMY,AMY,CTL]
err=[[rr[i]-lo[i] for i in range(3)],[hi[i]-rr[i] for i in range(3)]]
ax.axhline(0,color="#bbb",lw=0.8)
ax.bar(regs,rr,width=0.6,color=cols,edgecolor="white",linewidth=1,
       yerr=err,error_kw=dict(lw=1.1,capsize=4,ecolor="#333"),zorder=2)
for i,v in enumerate(rr):
    reliable = lo[i]>0 or hi[i]<0
    ax.text(i, hi[i]+0.06 if v>=0 else lo[i]-0.10, f"{v:+.2f}", ha="center", fontsize=8,
            fontweight="bold" if reliable else "normal")
ax.text(2,0.20,"reliable",ha="center",fontsize=7,color="white",fontweight="bold")
ax.text(0,0.14,"CI incl. 0",ha="center",fontsize=6.5,color="#444"); ax.text(1,0.14,"CI incl. 0",ha="center",fontsize=6.5,color="#444")
ax.set_ylim(-0.95,1.12); ax.set_ylabel("agreement of two affect encoders  (r)")
ax.set_title("Two independent affect encoders converge reliably in fusiform,\nbut not in the amygdala (CIs include 0)",fontsize=8.8,pad=8)
save(fig,"Fig3_two_encoder_convergence")

# ---- Fig 4: positive account (single-neuron population tuning) — unchanged (not re-run; encoder-independent) ----
fig,ax=plt.subplots(figsize=(5.2,2.5))
pred=["intensity","valence","choice","correct","confidence","RT"]; t=[0.20,0.08,0.03,-0.01,-0.08,-0.10]
sig=[True,False,False,False,False,False]
cols=[CTL if s else GREY for s in sig]
ax.axhline(0,color="#bbb",lw=0.8)
ax.bar(pred,t,width=0.66,color=cols,edgecolor="white",linewidth=1)
ax.text(0,0.205,"***",ha="center",fontsize=11,color=CTL,fontweight="bold")
ax.set_ylabel("amygdala population mean t"); ax.set_ylim(-0.16,0.26)
ax.set_title("What the amygdala does encode: a population bias toward\nstimulus intensity/salience (not graded emotion)",fontsize=8.6,pad=8)
ax.tick_params(axis="x",labelsize=8)
fig.text(0.5,-0.02,"Population mean-t (n=442 units); per-unit tuned fraction for intensity is n.s. (6.3%). "
         "The MFC control instead encodes reaction time.",ha="center",fontsize=6.8,color="#555")
save(fig,"Fig4_positive_account")
print("wrote:", ", ".join(sorted(f for f in os.listdir(FIG) if f.endswith('.png'))))
