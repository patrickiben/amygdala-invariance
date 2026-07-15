import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
FIG=os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.spines.top":False,
    "axes.spines.right":False,"axes.linewidth":0.8,"figure.dpi":300})
AMY="#D55E00"; CTL="#0072B2"; GREY="#7f7f7f"  # Okabe-Ito subset (CVD-safe)
def star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "n.s."
def save(fig,name):
    fig.savefig(f"{FIG}/{name}.png",bbox_inches="tight",dpi=300)
    fig.savefig(f"{FIG}/{name}.pdf",bbox_inches="tight"); plt.close(fig)

# ---- Fig 1: four-modality dissociation (affect encoder advantage over object encoder) ----
arms=[("Movie fMRI\n(encoding, ΔR²)","amygdala","fusiform",0.0002,0.0066,0.40,0.003,0.0019),
      ("Face morphs\n(RSA, Δρ)","amygdala","fusiform",-0.005,0.070,0.83,0.005,0.024),
      ("Single neurons\n(RSA, Δρ)","amygdala","MFC",-0.043,0.073,0.11,0.044,0.026),
      ("Audio\n(encoding, ΔR² over vision)","amygdala","STG",-0.0023,0.0088,0.5,0.010,0.0034)]
fig,axes=plt.subplots(1,4,figsize=(9.2,2.6))
for ax,(title,r1,r2,va,vc,pa,pc,se) in zip(axes,arms):
    ax.axhline(0,color="#bbb",lw=0.8,zorder=0)
    ax.bar([0,1],[va,vc],width=0.62,color=[GREY,CTL],edgecolor="white",linewidth=1,
           yerr=[se,se],error_kw=dict(lw=1,capsize=3,ecolor="#444"),zorder=2)
    ax.set_xticks([0,1]); ax.set_xticklabels([r1,r2],fontsize=8)
    ax.set_title(title,fontsize=8.3,pad=6)
    top=max(vc,va)+se; ax.text(0,va+se+top*0.06 if va>0 else se*0.4,"n.s.",ha="center",fontsize=7.5,color=GREY)
    ax.text(1,vc+se+top*0.06,star(pc),ha="center",fontsize=9,color=CTL,fontweight="bold")
    ax.margins(y=0.22)
axes[0].set_ylabel("affect − object encoder")
fig.suptitle("The amygdala is not emotion-encoder-specific; the sensory-cortex control is",
             fontsize=10,y=1.04,fontweight="bold")
fig.text(0.5,-0.06,"Emotion-supervised encoders beat a matched object encoder in fusiform/STG (colored, significant) "
         "but not the amygdala (grey, n.s.). Units differ per arm.",ha="center",fontsize=7,color="#555")
save(fig,"Fig1_four_modality")

# ---- Fig 2: bounded null (single-neuron TOST) ----
fig,ax=plt.subplots(figsize=(6.4,2.1))
est,lo,hi,margin,hi_ps=-0.043,-0.092,0.009,0.073,0.017
ax.axvspan(0,margin,color="#0072B2",alpha=0.08,zorder=0,label="non-superiority margin (+0.073)")
ax.axvline(0,color="#bbb",lw=0.8)
ax.axvline(margin,color=CTL,lw=1.2,ls="--"); ax.text(margin,1.42,"margin = +0.073\n(MFC-sized effect)",color=CTL,fontsize=7.5,ha="center")
ax.errorbar([est],[1],xerr=[[est-lo],[hi-est]],fmt="o",color=AMY,ms=8,capsize=4,lw=1.6,zorder=3,label="amygdala affect − object (unit boot, n=442)")
ax.axvline(hi_ps,color=AMY,lw=1.1,ls=":"); ax.text(hi_ps,1.42,"per-session\nupper +0.017",color=AMY,fontsize=7.0,ha="center")
ax.plot([0.073],[0.55],"D",color=CTL,ms=7,zorder=3); ax.text(0.073,0.30,"MFC effect",color=CTL,fontsize=7.5,ha="center")
ax.annotate("upper bound < margin in both frames\n(per-session +0.017; unit-boot +0.009)\none-sided non-superiority; ~82% power (descriptive)",
            xy=(hi_ps,1),xytext=(-0.125,1.72),fontsize=7.3,color="#333",ha="left",
            arrowprops=dict(arrowstyle="->",color="#888",lw=0.8))
ax.set_yticks([]); ax.set_ylim(0,2.1); ax.set_xlim(-0.13,0.13)
ax.set_xlabel("affect − object encoder (RSA Δρ)")
ax.set_title("A bounded null: the amygdala affect advantage is smaller than a detectable effect",fontsize=9.3,pad=8)
save(fig,"Fig2_TOST_bounded_null")

# ---- Fig 3: two-encoder convergence ----
fig,ax=plt.subplots(figsize=(4.6,2.5))
regs=["L-amyg","R-amyg","Fusiform"]; rr=[-0.26,-0.51,0.94]; cols=[AMY,AMY,CTL]
ax.axhline(0,color="#bbb",lw=0.8)
ax.bar(regs,rr,width=0.6,color=cols,edgecolor="white",linewidth=1)
for i,v in enumerate(rr): ax.text(i,v+(0.05 if v>0 else -0.09),f"{v:+.2f}",ha="center",fontsize=8)
ax.set_ylim(-0.75,1.05); ax.set_ylabel("agreement of two affect encoders  (r)")
ax.set_title("Two independent affect encoders converge in fusiform,\ndiverge in the amygdala",fontsize=9,pad=8)
save(fig,"Fig3_two_encoder_convergence")

# ---- Fig 4: positive account (single-neuron population tuning) ----
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
