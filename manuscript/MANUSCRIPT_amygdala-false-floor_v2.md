# No emotion-specific image-computable code in the human left amygdala

**Patrick Iben¹**
¹ Independent Researcher, Saint Petersburg, FL, USA · ORCID: 0009-0005-2443-7973 · Correspondence: patrickiben@gmail.com

---

## Abstract

A common inference in computational neuroscience holds that if a deep network trained to recognize construct
*X* predicts a brain region, that region represents *X*. We test this for the human amygdala with a
representation-invariance criterion: a construct-supervised encoder counts only if it beats a capacity-matched
non-construct encoder and converges across two independent construct encoders, with a positive control. The anchor
is 442 human amygdala neurons: affect encoders match their single-neuron geometry no better than object and
low-level encoders, and on most per-session contrasts significantly worse than object encoders, an effect
strongest and most consistent in the left amygdala (face-affect encoder minus object, per-session p<.001); the pooled affect-minus-object composite (−0.086,
bootstrap-p=.033) is a non-superiority null bounded below the (marginal) affect effect the same instrument detected
in medial-frontal cortex. In naturalistic movie fMRI the amygdala shows only a tiny EmoNet advantage over a matched
object encoder (+0.0009; significant but about ten times smaller than in fusiform), which a second affect encoder
does not reproduce and on which the two affect encoders do not reliably converge. Fusiform (vision) and superior
temporal (audio) cortex show genuine modality-specific structure. The amygdala response is better described by a
modest population bias toward stimulus intensity and salience plus a decision-uncertainty signal. The contribution
is a reusable invariance test; the negative finding converges with recent work relocating the image-computable
emotion code to cortex.

Keywords: amygdala; emotion; encoding models; representation-invariance; single-neuron; EmoNet; RSA; salience.

---

## 1. Introduction

Image-computable encoding models map a fixed deep-network feature space onto brain activity to test whether a
region represents the network's training objective [1,2]. A recent application argues that the human amygdala
hosts an image-computable emotion code: Jang & Kragel (2025) map features from EmoNet, a network trained to
classify emotion categories from images [3], onto amygdala BOLD [4].

Such claims rest on the encoder-dependence assumption, that predicting a region from a construct-trained network
licenses attributing the construct to the region. This inference is not automatically valid: above-chance
decoding or encoding of feature X in region Y does not entail that Y represents X [5]. A network differs from a
control in architecture, training distribution, feature smoothness, and dimensionality, any of which can drive a
correlation with brain activity, and object-trained networks alone can reproduce apparent emotion structure
[6, 7]. The safeguard is representation-invariance: a coding claim is credible only if it survives re-running
across paradigmatically different encoders of the same construct.

We operationalize a two-part test: a construct encoder must (A) exceed a capacity-matched non-construct encoder
and (B) converge across two independent construct encoders, with a positive-control region where the construct
code is known to exist so that a regional null is interpretable rather than merely underpowered. We apply it to
the amygdala emotion-coding claim across three datasets and two modalities, and we ask, if the amygdala does not
host an image-computable emotion code, what its response encodes instead.

We frame the contribution as methodological. Recent work from the originating group localizes the image-computable
facial-emotion code to cortical face and visual-context regions (Soderberg, Jang & Kragel, 2026 [18]), so the
present cortical-localization finding is convergent with theirs rather than contrarian. What we add is the disciplined, falsifiable form of the test and its generalization to audio and to
single neurons with positive controls in two modalities.

---

## 2. Results

### 2.1 Naturalistic movie fMRI: a far weaker, encoder-inconsistent amygdala affect signal than in sensory cortex
On the Naturalistic Neuroimaging Database (ds002837 [8]; 20 participants, 1.5T, TR=1s, *500 Days of Summer*), we
fit per-participant encoding models (partial least squares, PLS) predicting amygdala BOLD from EmoNet,
ImageNet ViT/ResNet, a low-level model, and a dimensionality-matched random encoder. EmoNet features were
extracted with the preprocessing of the reference PyTorch implementation of EmoNet (emonet-pytorch;
github.com/ecco-laboratory/emonet-pytorch), which resizes frames to 227 px and passes them as raw 0-255 RGB with
no scaling and no normalization (its network applies only a float cast, `x.to(torch.float)`; Methods 4.2). This
corrects a prior per-frame 1/255 scaling and materially changes the movie-arm result, and we report the change
plainly below. We used the unsmoothed (no_blur) denoised MNI derivative. Stimulus-to-BOLD alignment required
correcting a PAL-to-NTSC frame-rate drift: the dataset descriptor documents the film was presented at 25 fps [8],
versus the 23.976 fps source, i.e. a 4.27% linear time stretch (scale 25/23.976 = 1.042). The correction is
independently validated by low-level prediction of V1 and EmoNet prediction of fusiform peaking at that scale. We
report PLS variance partitions rather than the preregistered banded-ridge estimator because, at this
single-participant noise floor, banded ridge over-regularizes: it shrinks the fusiform EmoNet-minus-ResNet
increment roughly a hundredfold (from +0.0105 under PLS to +0.0001) and collapses the amygdala increment to zero,
compressing the very fusiform-versus-amygdala contrast at issue. PLS, with a fixed low-dimensional component
budget, preserves interpretable magnitudes and recovers a well-powered fusiform positive control; the banded ridge
is reported as a robustness check that yields the same qualitative verdict (Methods 4.3, 4.10).

EmoNet added unique variance over ImageNet in the amygdala (+0.0022 L, +0.0017 R, both p<.001), and that signal
was real (time-locked, with a time-shifted temporal null dropping to zero) and not a dimensionality artifact (a
random encoder was null in the amygdala where it was positive in visual cortex). With the corrected preprocessing,
EmoNet also shows a small but significant increment over a matched generic object encoder: EmoNet minus ResNet
over a ViT baseline was +0.0009 (p=.002) on the left and +0.0006 (p=.003) on the right. We flag this honestly,
because it reverses a claim in an earlier version of this work: using non-canonical EmoNet preprocessing, that
contrast was non-significant (L/R p=.38/.17), and that clean "an object encoder recovers the same signal" null
was a preprocessing artifact.

The amygdala affect signal is nonetheless far weaker, near the noise floor, and encoder-inconsistent compared with
a genuine sensory-cortex affect code, on three grounds. First, the amygdala increment is roughly an order of
magnitude smaller than the fusiform positive control, where the same EmoNet-minus-ResNet contrast is +0.0105
(p<.001), and it sits near the single-participant noise floor. On its own this cross-region magnitude ratio is
only suggestive, because the amygdala is a smaller, lower-SNR subcortical region whose encoding R² is lower for
every encoder; the decisive point is the next one, which is a within-region test that mere proportional
down-scaling of a genuine affect code cannot explain. Second, a second independent affect encoder (a face
valence/arousal network [9]) does not beat generic vision in the amygdala at all (face-EmoNet minus ResNet −0.0012
left, p=.29, n.s.; −0.0024 right, p<.001, i.e. significantly worse than the object encoder), whereas in fusiform
both affect encoders beat objects. Third, the two affect encoders do not converge in the amygdala: their
per-participant unique contributions correlate r=−0.28 left and −0.00 right, versus +0.79 in fusiform. Critically,
this amygdala non-convergence is not statistically reliable (its bootstrap 95% CIs include zero; see Section 2.2
and Figure 3), so it should be read as an absence of the reliable cross-encoder convergence seen in fusiform, not
as positive evidence of encoder-specific divergence. The overall pattern survived an adversarial battery (eroded
high-threshold mask with a hippocampus neighbor control; purged/gap cross-validation; a rich three-regressor
face-presence baseline; and an empirical null over 20 random encoders). Variance partitions are differences of
held-out R² on identical targets, valid despite negative absolute R² because the total sum of squares cancels.

A strong-baseline test confirmed that this small increment is not a weak-generic-encoder artifact: EmoNet retained
a unique increment over the strongest generic encoder, DINOv2, in the amygdala (EmoNet minus DINOv2 +0.0021 left,
+0.0015 right, both p<.001), even larger than its increment over ResNet. This sharpens rather than softens the
reading. The movie amygdala robustly meets criterion (A) — a construct encoder beating even a strong non-construct
encoder — but still fails criterion (B), because the second affect encoder does not reproduce it and the two affect
encoders do not converge; a signal that survives strong generic baselines yet is not affect-encoder-invariant is
precisely the encoder-specific, not construct-specific, pattern the invariance test is built to expose. The
increment also remains several-fold smaller than in the fusiform positive control, where EmoNet beats DINOv2 by
+0.0079 (Methods 4.11).

The honest movie-arm claim is therefore not a clean absence of an image-computable affect code in the amygdala,
but a far weaker, near-floor, encoder-inconsistent one: a single affect encoder yields a tiny significant increment
over generic vision, about ten times smaller than in sensory cortex, that a second affect encoder does not
reproduce and that the two affect encoders do not reliably agree on.

### 2.2 Task-evoked face morphs: no encoder matches the amygdala geometry (RSA)
To test whether the movie result reflected only a low-SNR passive-viewing regime, we analyzed a task-evoked
fear-to-happy face-morph dataset (Sun et al., 2023 [10]; n=19, 3T; amygdala tSNR ~140 to 190, full coverage). The
amygdala's emotion response was present at the univariate level (parametric fear intensity, left p=.042;
ambiguity, right p=.028); these are secondary sanity checks establishing that the amygdala responds to the
morphs, neither survives dependence-robust BY-FDR across the family (both q=.06), and neither bears on the
encoder-specificity claim. By RSA, however, no encoder reliably matched the amygdala's representational
geometry (all p>0.2), so the affect-versus-object comparison had nothing to attach to, and the lone hint (left
face-EmoNet minus ResNet, p=.06) was uncorroborated by the scene-emotion encoder (kragelEmoNet minus ResNet, p=.72). The fusiform again
showed genuine affect-specificity (scene-emotion encoder minus ResNet +0.093, p=.002; the face-affect encoder (faceEmoNet) was
concordant in sign but weaker, p=.06, so this fusiform positive control is single-encoder-significant here, with
reliable two-encoder convergence established in the movie arm, r=+0.79, bootstrap 95% CI [+0.39, +0.97]), a small increment on top of a largely
generic-visual code (ResNet RSA +0.279, emotion increment +0.093), confirming the test discriminates. This arm was
re-run end-to-end under the corrected canonical EmoNet preprocessing (raw 0-255 RGB at 227 px): the neural RSA
reproduces the prior faceEmoNet and object-encoder values to three decimals (the neural patterns are unchanged),
and the corrected kragelEmoNet encoder if anything sharpens the fusiform positive control (scene-emotion encoder
minus ResNet +0.070 in the prior run rises to +0.093, p=.002) while leaving the amygdala null intact, so the
qualitative verdict is confirmed rather than merely carried over.

### 2.3 Single neurons: affect encoders match the amygdala geometry no better, and mostly worse, than object encoders
We removed the resolution and SNR limits of BOLD using 442 human amygdala neurons (left 277, right 165, 22
sessions, epilepsy patients) recorded on the same morph task [10, 11], now with the corrected EmoNet preprocessing
(raw 0-255 RGB at 227 px; Methods 4.2). Our primary inference is the per-session frame, which respects the nesting
of units within recording sessions; the unit bootstrap, which pseudoreplicates units across sessions, is reported
as descriptive corroboration (Methods 4.5). In the per-session frame, affect encoders matched the amygdala
single-neuron geometry significantly worse than object encoders on three of four encoder-pair contrasts across all
442 units (scene-emotion encoder (kragelEmoNet) minus ResNet −0.037, p=.040; kragelEmoNet minus ViT −0.038,
p=.004; face-affect encoder (faceEmoNet) minus ViT −0.039, p=.038; the fourth, faceEmoNet minus ResNet −0.038, was
non-significant, p=.134). The effect was strongest and most consistent in the left amygdala (277 units), where both
object contrasts were highly significant (faceEmoNet minus ResNet −0.089, p<.001; kragelEmoNet minus ResNet −0.071,
p=.001); this left-lateralized, per-session result is the correction-robust anchor. The right amygdala (165 units)
showed no affect-object difference in either direction (per-session faceEmoNet minus ResNet +0.007, p=.83;
kragelEmoNet minus ResNet +0.007, p=.78), so we scope the single-neuron deficit to the pooled and left-amygdala
samples and do not read a whole-region claim into the right hemisphere (see Discussion).

As descriptive corroboration, the pooled affect(best)-minus-object(best) composite over all 442 units — a
max-of-affect-minus-max-of-object statistic that is defined only in the unit bootstrap — was −0.086 (boot-p=.033),
strengthening a previously non-significant estimate (−0.043, boot-p=.10). We flag plainly that this composite
reaches significance only in the descriptive bootstrap frame: in the primary per-session frame its 95% interval
crosses zero (upper +0.0063), because the max-max composite has no single per-session test. The per-session support
for the verdict is the three significant encoder-pair contrasts and the left-amygdala result above, not the
bootstrap composite p; and because the confirmatory amygdala contrast is interpreted directly rather than
BY-FDR-corrected (Methods 4.9), the correction-robust claim rests on the left amygdala (p<.001), with the pooled
composite (boot-p=.033) and the medial-frontal benchmark (boot-p=.040) treated as uncorrected, marginal
corroboration.

Beyond significance, this affect deficit is bounded against a positive benchmark. Treating the +0.100 affect
effect that the same test detected in a medial-frontal comparison region (268 units) as the equivalence margin,
the amygdala affect-object contrast is −0.083 as a point estimate (bootstrap SE=0.039, 95% CI [−0.156, −0.0065],
boot-p=.033); this is the same max(affect)−max(object) composite reported above (affect encoders {faceEmoNet,
kragelEmoNet}, object encoders {ResNet, ViT}, low-level excluded from both), whose bootstrap median is −0.086, so
the two values differ only by estimator (point versus bootstrap median), not by encoder set. Its 95% upper bound of
−0.0065 in the unit bootstrap and
+0.0063 in the per-session frame both fall far below the +0.100 margin. A descriptive unit-level sensitivity
calculation indicates about 72% power to detect the +0.100 margin-sized effect (the bootstrap SE grew to 0.039
under the corrected features), but this is moot. Crucially, the "not a sensitivity failure" argument does not need
the benchmark at all: a region whose per-session encoder contrasts detect a reliably negative effect (above) is
demonstrably sensitive, so the amygdala null is a genuine result rather than a detection failure. The
medial-frontal comparison region is moreover itself only a marginal positive (affect-object +0.100 in the unit
bootstrap, boot-p=.040, but per-session faceEmoNet minus ResNet p=.108 and kragelEmoNet minus ResNet p=.054), a
hint rather than a firm positive; we treat it as such, and rest the dissociation on the significantly-negative
per-session amygdala contrasts, with the bounded-null comparison as a secondary check that holds in either frame.

**Strong-baseline robustness (representation invariance).** A negative claim is only as strong as the strongest
baseline that fails to beat it, so we re-ran the anchor RSA with paradigmatically different, stronger frozen encoders
on both sides: two stronger generic encoders (DINOv2, self-distilled; CLIP ViT-B/32, contrastive) and a modern
facial-expression network as a stronger affect encoder. The strongest generic encoder (DINOv2) matched the amygdala
single-neuron geometry as well as the object encoders (RSA +0.592, versus ViT +0.592) and better than either affect
encoder, while the modern affect encoder matched it worse (+0.140), so the affect encoders used here are not weak
strawmen. The affect(best)−object(best) deficit held across all four encoder configurations and, if anything,
strengthened under the stronger generic encoders (−0.086 with the original panel, −0.088 adding DINOv2 and CLIP,
−0.084 adding the stronger affect encoder, −0.091 with stronger encoders on both sides; all bootstrap-p<.05). The
verdict is therefore robust to encoder strength rather than an artifact of a weak affect probe (Methods 4.11).

We further ran a specification-curve (multiverse) over the anchor verdict across 136 defensible analytic choices for
the pooled and left-amygdala samples — encoder set (original, plus the stronger encoders), best-of aggregation
(max/mean), RDM metric (correlation, Spearman, Euclidean), and neural-pattern normalization — and the
affect(best)−object(best) contrast was negative in 100% of them (median −0.11; Figure S1). No analytic choice
reversed the sign; the only large mover was hemisphere (the right amygdala is null, as reported). The single-neuron
verdict is thus robust across the multiverse of reasonable pipelines, not an artifact of the specific one (Methods 4.11).

This analysis targets an image-computable code, meaning a stimulus-to-neural mapping. It does not test whether
an emotion code is recoverable from the population geometry of experimenter-defined affective states under a
task, which a recent study reports for basolateral amygdala (O'Neill, Posani et al., 2026 [12], mouse). That is a
different and stricter-in-a-different-way question, and it is a genuine open threat to any unit-level or
stimulus-encoding negative; we return to it in the Limitations.

### 2.4 Audio: a single-encoder cross-modal control
Because the preceding encoders were vision-only, we tested audition using an emotion-fine-tuned wav2vec2 encoder
[13] on the movie soundtrack (ds002837, n=20), with auditory cortex as positive control. We note that this arm
uses one audio affect encoder and therefore does not satisfy criterion (B) of the invariance test; it is a
single-encoder cross-modal control, not a fourth full application of the two-part test. Audio-affect predicted
superior temporal gyrus uniquely over vision (+0.0189, p=.001; primary auditory cortex +0.0040, p=.12) but added
nothing to the amygdala (left −0.0028, right −0.0023, both p<.001 in the negative direction), which patterned
with visual, not auditory, cortex. So the amygdala's affect response is no more auditory than visual. This arm was
re-run under the corrected EmoNet vision baseline: the audio-affect and audio-low features are unaffected by the
EmoNet correction and reproduce exactly, and the corrected vision baseline sharpened the STG positive control
(audio-affect over vision +0.0088 in the prior run rises to +0.0189, p=.001) while the amygdala remained negative,
so the verdict is confirmed rather than merely carried over.

### 2.5 What the amygdala does encode: a bias toward intensity and salience, plus uncertainty
Using behavioral variables recorded with the single-neuron spikes (choice, reaction time, confidence,
correctness) alongside stimulus valence and intensity, per-unit regression revealed a modest population bias
toward stimulus intensity: emotional extremes drove firing more than the ambiguous middle (population mean t
+0.20, p<.001; 75% of tuned units positive), replicating Wang et al. 2017 [11]. This mean-t bias was bilateral
and numerically larger in the right amygdala (+0.26, p<.001) than the left (+0.16, p=.021). The individually
intensity-tuned fraction was modest and left-lateralized: it exceeded chance in the left amygdala (7.9%, p=.02)
but not the right (3.6%, n.s.) or the pooled population (6.3%, p=.12). So this is a population-level bias with
only modest left-hemisphere single-unit tuning, not strong single-unit intensity coding. Two other signals reached the fraction threshold: a left-amygdala decision-
uncertainty signal (confidence, 10.5% of units, p=.0002; firing more on low-confidence trials) and right-
amygdala valence and choice signals (12.1% and 11.5%, p<.001). The medial-frontal comparison region instead
encoded reaction time, that is decision effort (10.8%, p<.0001), a double dissociation that supports the
region-specificity of the amygdala pattern. Single-trial cross-validated prediction was at the noise floor for
all models, so its overfitting-dominated model comparison was not interpreted. Thus the amygdala tracks a coarse
nonlinear salience/intensity quantity plus judgment uncertainty, a characterization consistent with (not a proof
of) the view that it is a relevance detector rather than a feedforward stimulus-to-emotion encoder [14]. Effects
are modest; see Limitations.

---

## 3. Discussion

Across three datasets and two modalities, an emotion-supervised encoder never earns a reliable construct-specific
claim on the amygdala under the invariance test, while the modality-appropriate sensory cortex does (fusiform for
vision, superior temporal gyrus for audio). We anchor this verdict on the single-neuron arm, which is the
highest-resolution test and which strengthened under the corrected EmoNet preprocessing. In the primary
per-session frame, affect encoders match the amygdala single-neuron geometry significantly worse than object
encoders on three of four encoder-pair contrasts across 442 units, an effect strongest and correction-robust in
the left amygdala (faceEmoNet minus ResNet −0.089, per-session p<.001); the pooled affect(best)-minus-object(best)
composite (−0.086, bootstrap-p=.033) corroborates this in the descriptive bootstrap frame. This is not a
sensitivity failure: a region whose per-session contrasts detect a reliably negative effect is demonstrably
sensitive, and the same instrument moreover registers a (marginal) positive affect effect in the medial-frontal
comparison region (+0.100, bootstrap-p=.040; per-session p=.05–.11), against which the amygdala effect is bounded
(per-session and unit-bootstrap upper limits +0.0063 and −0.0065, far below +0.100). The right amygdala shows no
affect-object difference in either direction (per-session p=.78–.83), so we scope the single-neuron claim to the
pooled and left-amygdala samples.

The naturalistic-movie arm is weaker, and we report it honestly rather than as a clean absence. The previously
reported "EmoNet does not beat a matched object encoder in the amygdala" (p=.38/.17) was a preprocessing artifact
of the earlier feature inputs. Under the corrected canonical preprocessing, EmoNet does add a small unique
increment over a matched object encoder in the amygdala (EmoNet minus ResNet over a ViT baseline, +0.0009 left,
p=.002; +0.0006 right, p=.003; EmoNet-unique over ImageNet +0.0022 left, +0.0017 right, both p<.001). We therefore
no longer claim the amygdala EmoNet advantage is null in the movie. But three facts keep this from supporting an
amygdala affect code. First, the increment is roughly an order of magnitude smaller than in the fusiform positive
control (EmoNet minus ResNet +0.0105, p<.001), that is, near the single-participant noise floor. Second, the
second, independent affect encoder shows no such advantage in the amygdala (−0.0012 left, n.s.; −0.0024 right, in
the worse direction), so the two affect encoders disagree there. Third, that amygdala disagreement is not
statistically reliable (Figure 3): the between-encoder convergence CIs include zero, whereas the fusiform
convergence is reliable (r=+0.79, CI [+0.39, +0.97]). So the honest movie-arm claim is a far weaker, near-floor,
encoder-inconsistent affect signal in the amygdala than in sensory cortex. This softening does not overturn the
thesis; it locates its strength in the single-neuron arm and treats the movie arm as consistent-but-weak
corroboration.

This reconciles two literatures. Object-trained networks alone reproduce apparent emotion structure (object-net
features can outperform EmoNet at predicting emotion ratings on diverse scenes [7]), and even a macaque study
that predicts amygdala valence does so with ventral-stream object-vision models, describing a visual-drive to
valence linkage rather than a graded emotion category and without running a two-encoder control (Peter et al.,
2026 [15], macaque). Read through the invariance lens, "an object-vision model predicts amygdala valence" is
evidence for the object-confound, not against it, and the single-neuron result makes the same point more sharply:
where the affect and object encoders can be compared at the highest resolution, the object encoder fits the
amygdala geometry better. The image-computable emotion code recovered by deep networks is best localized to
cortical regions, consistent with the originating group's own recent report localizing the facial-emotion code to
cortical face and visual-context regions (Soderberg, Jang & Kragel, 2026 [18]); the present result is convergent
with that cortical localization, and our contribution is the disciplined test plus multimodal generalization
rather than a reversal of consensus. We state the scope carefully, and it is why we scope the title
to the left amygdala: across the fMRI arms the amygdala bilaterally carries no reliable emotion-encoder-specific,
image-computable code of the kind sensory cortex carries, but the strongest, correction-robust positive evidence
that generic object and low-level encoders match its single-neuron geometry better than affect encoders is
left-lateralized (the right amygdala shows no affect-object difference in either direction). This is not a claim
that the amygdala is uninvolved in emotion.

The complementary single-neuron characterization gives a coherent positive account: the amygdala carries a
population bias toward stimulus intensity and salience and a decision-uncertainty signal, consistent with a
relevance-detection role [14] and with its parametric intensity and ambiguity coding [11], rather than a
feedforward stimulus-to-emotion transform.

For encoding-model practice, we recommend a routine guard: (A) compare against a capacity-matched non-construct
encoder, and (B) require convergence across two independent construct encoders, with a positive-control region.
The estimator matters at the single-participant noise floor: the preregistered banded-ridge path, when run,
over-regularizes so hard that it compresses the fusiform EmoNet-minus-ResNet increment roughly a hundredfold (to
+0.0001) and shrinks the amygdala increment to indistinguishable from zero, so it yields the same qualitative
verdict as PLS but sacrifices the interpretable effect magnitudes that make the fusiform-versus-amygdala
dissociation legible. This justifies reporting the fixed low-dimensional PLS budget as primary, while leaving the
anchoring single-neuron RSA (which does not depend on the encoding estimator) untouched. The components (variance
partitioning, RSA model comparison, empirical nulls) are standard [16,17]; the contribution is their disciplined
combination as a false-floor guard applied to a specific affective-coding claim. For affective neuroscience,
programs seeking a mechanistically interpretable image-computable model of amygdala affect should retarget the
image-computable component to cortical regions and model the amygdala as a salience and uncertainty integrator, a
computation not captured by any single feedforward sensory encoder, whose remaining candidate substrate
(interoception) is not encoder-shaped and requires a different paradigm.

---

## 4. Methods

### 4.1 Datasets and preprocessing

**Naturalistic movie fMRI (ds002837).** We analyzed the Naturalistic Neuroimaging Database (OpenNeuro ds002837; Aliko et al., 2020 [8]), 20 participants (sub-1 through sub-20) who viewed the ~90-min audiovisual film '500 Days of Summer'. Acquisition was at 1.5T with a TR of 1 s (multiband EPI; the multiband factor and remaining acquisition parameters are as reported in the ds002837 dataset descriptor, Aliko et al., 2020 [8]). We used the released MNI152 derivatives (files matching `*task-500daysofsummer*no_blur_no_censor.nii.gz`), which are denoised and spatially unsmoothed (the `no_blur` derivative), with runs concatenated with scanning-pause timing correction, yielding on the order of 5,700 volumes per participant (5,703–5,704 in the committed feature and BOLD-length headers). Derivatives were fetched from the OpenNeuro S3 mirror (`s3.amazonaws.com/openneuro.org/ds002837`). Each participant's BOLD series was loaded as float32 with nibabel; region voxels were selected by mask and then z-scored over time (see 4.3). No re-registration was performed on the released derivatives. Full acquisition parameters (echo time, flip angle, voxel size, slice count) are as reported in the ds002837 data descriptor (Aliko et al., 2020 [8]).

**Task-evoked face-morph fMRI (Sun et al., 2023).** We analyzed the fMRI arm of the uniform multimodal emotion dataset (OSF 26RHZ; Sun et al., 2023 [10]). Participants viewed static faces morphed along a fear-to-happy continuum sampled at seven expression levels. Acquisition was at 3T with a TR of 2 s in an event-related design with parametric fear and ambiguity. The dataset was supplied SPM12-preprocessed (normalized, smoothed `sw*.img` volumes with per-run `rp_*.txt` realignment parameters and an author-supplied first-level `SPM.mat`); the original SPM first-level design specified a single 'Face' condition (plus 'Fix') with two parametric modulators (fear, ambiguity), a canonical HRF basis of length 32 s (order 1, no Volterra expansion beyond order 1), 16 time bins per scan (xBF.T=16), a 128 s high-pass filter, AR(1) autocorrelation modeling, and the six realignment parameters as nuisance covariates. Rather than reuse the SPM contrast images, we re-fit a first-level GLM in nilearn `FirstLevelModel` (t_r=2.0, hrf_model='spm', drift_model='cosine', high_pass=1/128 Hz, minimize_memory=True) with events of duration 0 s coded by expression level and the six `rp_*.txt` columns as confounds, then extracted seven per-level beta maps (output_type='effect_size'). Analysis retained the 19 participants whose directory contained a `face_emotion_para3/SPM.mat` first-level design. Amygdala coverage and temporal SNR were verified with a coverage gate: on the first 120 volumes of each participant's normalized series, voxelwise temporal SNR was the temporal mean divided by the temporal standard deviation (mean / (std + 1e-6)), summarized as the median over amygdala-mask voxels, with coverage the percentage of amygdala-mask voxels retained in the SPM analysis mask; the reported tSNR range (approximately 140 to 190, computed at runtime by `coverage_gate.py` and not separately logged) is the spread of these per-participant medians, and full coverage indicates the amygdala was not truncated by the echo-planar field of view.

**Single-neuron recordings.** We analyzed processed single-unit firing rates released with Sun et al., 2023 [10] (originating recordings from Wang et al., 2017 [11]), read from `FiringRate_Amygdala.mat` and `FiringRate_MFC.mat`. The result logs report 442 amygdala units (277 left, 165 right) across 22 sessions and 268 medial-frontal (MFC = ACC+SMA) units, each with per-trial behavioral fields. Sessions and unit counts are computed at runtime from the `.mat` files. The raw intracranial recordings (Wang et al., 2017 [11]) are request-only for clinical-privacy reasons; only the released processed firing rates were used here.

### 4.2 Stimulus feature extraction

**Film frame sampling.** Movie frames were extracted with ffmpeg at 6 fps (FPS=6) as square 256x256 PNGs (filter `fps=6,scale=256:256`), giving 6 frames per 1 s TR. Per-frame features were reduced to TR resolution by trimming to a multiple of FPS and averaging within each TR block (`reshape(-1, 6, dim).mean(axis=1)`, n_TR = n_frames // 6). All feature caches were stored TR-averaged, before HRF convolution and before the PAL alignment described below.

**Vision encoders (movie arm).** Each encoder produced a per-frame feature vector, subsequently TR-averaged.
- *EmoNet* (Kragel et al., 2019 [3]): the emotion-supervised AlexNet-based network, weights from OSF (osf.io/amdju). A forward hook on the penultimate 4096-d linear layer (`classifier[-2]`, the fc7-equivalent activation; the repository README labels the same layer 'conv_6') returned a 4096-d vector. Input frames were resized to 227x227 and converted BGR to RGB, then passed as raw 0-255 RGB values with no scaling and no ImageNet mean/std normalization, matching the reference PyTorch implementation of EmoNet (emonet-pytorch; github.com/ecco-laboratory/emonet-pytorch), whose pipeline resizes to 227x227 and applies only a float cast to the network input (`x.to(torch.float)` in its `models.py`), i.e. no /255 and no mean subtraction. This replaces the 1/255 scaling used in the previous version of this arm; the correction changes the input intensity range EmoNet receives but not the layer or dimensionality. In the Results this network is referred to as the scene-emotion encoder.
- *ViT-B/16* (object control): torchvision `vit_b_16` with `ViT_B_16_Weights.IMAGENET1K_V1`, classification heads replaced by Identity, returning the 768-d CLS-token embedding.
- *ResNet-50* (object control): torchvision `resnet50` with `ResNet50_Weights.IMAGENET1K_V2`, `fc` replaced by Identity, returning the 2048-d global-average-pool feature.
- *Low-level* control (10-d): a Gabor energy bank of 4 orientations x 2 scales (8 filters) plus mean luminance and frame-to-frame motion energy. Gabor kernels used `cv2.getGaborKernel` with ksize=15, per-scale sigma=2.0*(s+1) and lambda=4.0*(s+1), orientation theta=pi*o/4, gamma=0.5; per-filter energy = sqrt(mean(relu(fftconvolve(gray, kernel))^2)); luminance = gray.mean(); motion = mean(|gray - prev_gray|), with grayscale = frame/255.
- *Random* (dimensionality-matched null): torchvision `alexnet(weights=None)` with `torch.manual_seed(0)` set before instantiation and the final classifier layer stripped so the output is the penultimate 4096-d activation.

The ViT, ResNet, and random-AlexNet encoders shared an input transform: frames resized to 224x224, converted BGR to RGB, scaled by 1/255, then ImageNet-normalized (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).

- *Second affect encoder* (movie arm, criterion-B convergence; referred to in the Results as the face-affect encoder): a face valence/arousal network (Toisoul et al., 2021 [9]), `EmoNet(n_expression=8)` with weights `emonet_8.pth`; a hook on `avg_pool_2` returned a 256-d vector. Frames were sampled at 6 fps, resized square to 256x256, converted BGR to RGB and scaled by 1/255 (no ImageNet normalization), then TR-averaged.

**Vision encoders (face-morph arm).** Features were extracted per stimulus image and averaged within each of the seven expression levels to form the level-by-feature matrices used by RSA. faceEmoNet: the Toisoul `EmoNet(n_expression=8)` (`emonet_8.pth`, loaded strict=False), `avg_pool_2` -> 256-d, input resized to 256x256 and scaled by 1/255 (no ImageNet normalization). kragelEmoNet: the Kragel EmoNet, conv_6 layer -> 4096-d, input resized to 227x227 and passed as raw 0-255 RGB with no normalization (matching the emonet-pytorch reference implementation, github.com/ecco-laboratory/emonet-pytorch), replacing the prior ImageNet-normalized input. ResNet-50 (`ResNet50_Weights.IMAGENET1K_V2`, fc=Identity, 2048-d) and ViT-B/16 (`ViT_B_16_Weights.IMAGENET1K_V1`, heads=Identity, 768-d) shared a Resize(256) -> CenterCrop(224) -> ImageNet-normalize transform. Low-level (4-d): [grayscale mean, grayscale std, mean |grad_x|, mean |grad_y|] computed on 64x64 grayscale. Note the face-morph low-level control differs by construction from the 10-d movie low-level control.

**PAL frame-rate alignment (scale 1.042).** Movie features were resampled onto the BOLD TR grid with a fixed linear time-stretch of 1.042. The value is the ratio of the presentation frame rate to the source rip rate: the film was presented at PAL 25 fps while the analyzed rip was 23.976 fps, so 25 / 23.976 = 1.0427 ~ 1.042, a 4.27% stretch. The constant is hardcoded (SCALE=1.042) and was validated empirically by a scale sweep (1.036, 1.039, 1.042, 1.045, 1.048, 1.000) in which low-level prediction of V1 and EmoNet prediction of pooled fusiform both peaked at 1.042. Alignment was applied inside `prep()`: for a region time series of length T and a cache of Nm movie rows, target indices `mt = clip(1.042 * arange(T), 0, Nm-1)` were formed and each feature column was linearly resampled onto `mt` with `np.interp`, so the feature at BOLD TR t is the movie feature at fractional index 1.042*t (clipped at the end).

### 4.3 Encoding models and variance partitioning

**Feature preparation.** Within each encoding fit, every encoder feature block passed through the same `prep()` pipeline: PAL resampling (4.2), HRF convolution, z-scoring over time, PCA to 300 dimensions, then a second z-scoring. PCA `pca(X, k)` mean-centered columns and, if the column count exceeded k=300, projected onto the top-300 right singular vectors via full SVD (`np.linalg.svd(X, full_matrices=False)`, `X @ Vt[:300].T`); blocks with 300 or fewer columns were returned centered but otherwise unchanged. Z-scoring used (x - x.mean(0)) / (x.std(0) + 1e-6) and was applied at three points: each region's voxel time series (per column), the features after HRF and before PCA, and the features after PCA.

**Hemodynamic model.** Features on the TR grid were convolved with a canonical SPM double-gamma HRF sampled at TR=1 s. `spm_hrf(tr)` used t = arange(0, 32+dt, dt) with dt=tr, hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16)/6.0, normalized to unit sum; `convolve_hrf` convolved each feature column with `np.convolve` and truncated to the first T samples (causal).

**Estimator and cross-validation.** Encoding used `sklearn.cross_decomposition.PLSRegression` with n_components = min(10, X.shape[1]) and scale=False, fit per training fold and used to predict the held-out fold. Cross-validation was 5-fold contiguous-block (leave-one-block-out): fold edges = `np.linspace(0, n, 6).astype(int)`, each fold the contiguous index range between consecutive edges, training the concatenation of the other four blocks. Held-out predictions were assembled across folds and a single R2 computed. Model fitting used one PLS per feature-set / region / participant.

**Preregistered banded-ridge estimator and its behavior at the noise floor.** The preregistered production path was a banded ridge regression with a separate regularization band per feature block (per-band penalty chosen by inner cross-validation over a 10^1–10^7 grid, closed-form ridge solve). We ran it on all 20 participants. It reproduces the same qualitative dissociation as PLS but over-regularizes at the single-participant noise floor: the EmoNet-minus-ResNet contrast is +0.0001 in the fusiform positive control (still p=.003, given a correspondingly tiny standard error) and −0.0000 in the amygdala (p=.40 left, .52 right), i.e. it compresses the fusiform increment roughly a hundredfold relative to PLS (+0.0105 under PLS) and shrinks the amygdala increment to zero. Because banded ridge sacrifices the interpretable effect magnitudes that make the fusiform-versus-amygdala contrast legible, the PLS estimator with a fixed low-dimensional component budget (n_components = 10) is used as the primary encoding path; the banded ridge is reported as a robustness check that yields the same verdict (indeed a cleaner amygdala null).

**Held-out R2.** R2 was the mean over voxels of (1 - SS_res/SS_tot) on the assembled held-out predictions: per voxel ss = sum((y - pred)^2, axis=0), st = sum((y - y.mean(0))^2, axis=0), with st < 1e-9 set to NaN and `np.nanmean` taken over voxels. Absolute held-out R2 can be negative at the single-participant noise floor.

**Variance partitioning.** Unique variance was computed as the difference of two held-out R2 values evaluated on the identical z-scored voxel target, with the same folds, feature blocks concatenated column-wise before PLS. The reported partitions include EmoNet-unique-over-ImageNet = R2([emonet, vit, resnet]) - R2([vit, resnet]); EmoNet-unique-over-low-level = R2([emonet, lowlevel]) - R2([lowlevel]); and a stricter R2([emonet, vit, resnet, lowlevel]) - R2([vit, resnet, lowlevel]). Because both terms score the same target with the same denominator, R2(full) - R2(reduced) = (SS_res,reduced - SS_res,full)/SS_tot; the total sum of squares cancels, so the sign is a valid relative comparison of held-out residuals even when both absolute R2 values are negative.

**ROI masks.** Masks were derived from the nilearn Harvard-Oxford atlases: amygdala from the subcortical maxprob-thr25-2mm atlas ('Left Amygdala', 'Right Amygdala'); the fusiform positive control from the cortical maxprob-thr25-2mm atlas ('Temporal Occipital Fusiform Cortex'); and V1 as 'Occipital Pole' (cortical thr25). Each label was resampled to the participant's BOLD reference with `resample_to_img(interpolation='nearest')` and binarized at > 0.5.

**Group inference.** Group statistics were one-sample t-tests versus 0 (`scipy.stats.ttest_1samp`, df = 19) over the per-participant held-out R2 or variance-partition values across the 20 participants, reporting mean and SEM = std(ddof=1)/sqrt(n).

### 4.4 Representational similarity analysis

**RDM construction and metric.** For both the face-morph fMRI and single-neuron arms, a representational dissimilarity matrix (RDM) was the correlation distance 1 - Pearson correlation matrix computed over the seven condition rows. The same `rdm()` was applied to neural patterns and to encoder feature matrices. For fMRI, the neural pattern was the 7 x n_voxel beta matrix restricted to an ROI, with each condition (row) z-scored across voxels (mean/std over the voxel axis, eps 1e-6) before the RDM. Encoder RDMs were built the same way over the per-level averaged feature matrices `lvl_<encoder>`.

**Comparison.** Neural-encoder similarity was the Spearman rank correlation between the upper triangles (`np.triu_indices(7, 1)`) of the neural and encoder RDMs, computed per participant, ROI, and encoder. The seven conditions were the ascending morph expression levels FE = [0, 30, 40, 50, 60, 70, 100], mapped from the SPM parametric fear rank via {1:100, 2:70, 3:60, 4:50, 5:40, 6:30, 7:0} so beta ordering matched the encoder level ordering. Encoders compared were faceEmoNet, kragelEmoNet, resnet, vit, and lowlevel; ROIs were L-amyg, R-amyg, Fusiform (positive control), and V1, using the same Harvard-Oxford masks as in 4.3. Group inference was one-sample t versus 0 across participants (mean, SE = nanstd(ddof=1)/sqrt(n)), and the encoder contrasts faceEmoNet - resnet and kragelEmoNet - resnet were per-participant paired differences tested by one-sample t.

**Secondary univariate checks.** As sanity checks (labeled secondary, not bearing on encoder specificity), the seven-level amygdala ROI-mean profile was regressed on a z-scored fear-linear regressor and an ambiguity regressor amb = -((fe - 50)^2) (inverted-U peaked at fe=50) via ordinary least squares on the mean-removed profile, with group one-sample t of each beta. A leave-one-level-out Ridge (alpha=1.0) R2 predicting the z-scored amygdala profile from each encoder's per-level features reduced to 3 PCs was also computed.

### 4.5 Single-neuron pseudo-population analysis and equivalence testing

**Per-unit tuning and pseudo-population.** For each unit, the baseline-corrected trial response was x = countAll - countBaseline, restricted to fear-happy trials (trialTypes == 1). A seven-level tuning vector was the NaN-mean of x within each morph level codeL in 1..7; units with any NaN level were dropped, and the tuning shape was z-scored ((vec - mean)/(std + 1e-9)). The pseudo-population was the units-by-seven matrix `tun`; the population RDM was 1 - corrcoef(P.T), i.e. a 7x7 matrix across conditions with units as features. Population RSA was the Spearman correlation of upper triangles against each encoder RDM.

**Bootstrap over units versus per-session group test.** Two inferential frames were computed. The unit bootstrap resampled units with replacement (`RandomState(0)`, B=2000 in the RSA script and B=4000 in the equivalence script), recomputing the pseudo-population RSA to give percentile 95% CIs; the affect-object statistic per bootstrap was ao = max(faceEmoNet, kragelEmoNet) - max(resnet, vit), with a two-sided bootstrap p = 2*min(mean(ao <= 0), mean(ao >= 0)). The per-session group test, which respects the nesting of units within sessions and is the primary inference, computed the pseudo-population RSA within each of the 20 (of 22) sessions that had at least 3 units and applied a one-sample t versus 0 across sessions (`ttest_1samp`, nan_policy='omit'); encoder-contrast pairs tested were faceEmoNet-resnet, kragelEmoNet-resnet, faceEmoNet-vit, and kragelEmoNet-vit. The same procedure was run for all amygdala units, left-only, right-only, and the MFC (ACC+SMA) comparison region.

**One-sided non-superiority (TOST-style) bound.** The equivalence margin was DELTA = 0.100, defined as the affect-specificity the single-neuron instrument itself detected in the MFC comparison region (affect-object +0.100, boot-p=.040). The statistic was the affect-object contrast ao on the pseudo-population (here the encoder set excludes lowlevel). The per-session primary bound was the 95% upper limit mean + 1.96 * std(ddof=1)/sqrt(n) across sessions; the descriptive unit-bootstrap bound was the 97.5th percentile of the bootstrap distribution and its SE the bootstrap standard deviation. Under the corrected features this contrast is significantly negative in the unit-bootstrap frame (point estimate −0.083, bootstrap median −0.086, boot-p=.033; unit-bootstrap SE 0.039, 95% CI [−0.156, −0.0065]; both the point and the bootstrap-median value use affect encoders {faceEmoNet, kragelEmoNet} and object encoders {ResNet, ViT}, differing only by estimator). The one-sided non-superiority claim is then doubly satisfied: the 95% upper bound (−0.0065 in the unit bootstrap, +0.0063 in the primary per-session frame) falls far below DELTA, and the point estimate is on the opposite side of zero from an affect advantage; note the pooled composite is significantly negative only in this descriptive bootstrap frame (its per-session interval crosses zero), so the primary-frame support is the per-session encoder-pair contrasts (Section 2.3). A descriptive sensitivity figure was reported as power = norm.cdf(DELTA/se - 1.96) at alpha=.05 (two-sided 1.96), using the data-derived bootstrap SE; this yields about 72% power to detect the +0.100 margin-sized effect given the corrected bootstrap SE of 0.039, and is labeled a unit-level sensitivity figure rather than a primary inference, both because the unit bootstrap pseudoreplicates across sessions and because the per-session contrasts already detect a reliably negative effect, which makes the power figure moot.

### 4.6 Single-neuron characterization

Each unit's response y = countAll - countBaseline (baseline-corrected, not z-scored) was regressed on an intercept plus six z-scored predictors: valence = codeL centered at 4, intensity = |codeL - 4|, choice = vResp, RT, confidence = vCR, and correctness = vCorr, with z-scoring zc = (x - nanmean)/nanstd. Trials were restricted to fear-happy (trialTypes == 1) with all predictors and y non-NaN, requiring at least 40 valid trials per unit. OLS was solved by least squares; t = beta/se with se from sigma^2 * pinv(X'X), residual dof = n - 7, units with dof < 5 dropped. A unit was 'tuned' for a predictor if |t| > 1.96; the fraction tuned was tested against chance 5% with a one-sided binomial test (`stats.binomtest`, alternative='greater'), and the population mean-t was a one-sample `ttest_1samp` of per-unit t versus 0 (the sign fraction of tuned units was also reported). A secondary 5-fold cross-validated R2 model comparison contrasted a stimulus model (valence, intensity), a decision model (choice, RT, confidence, correct), and the full model (folds via `np.array_split(idx, 5)`, decision-versus-stimulus by paired `ttest_rel`); single-trial CV R2 was at the noise floor and this comparison was not interpreted. The regression was run for all amygdala units, left-only, right-only, and MFC.

### 4.7 Audio arm

**Encoder.** The audio-affect encoder was the audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` model (Wagner et al., 2023 [13]), loaded via `AutoFeatureExtractor` and `Wav2Vec2Model` (eval mode, mps if available else cpu). Features were the final-layer `last_hidden_state` (1024-d per frame at approximately 50 Hz; no intermediate hidden-state selection). The soundtrack `film_16k.wav` was read at SR=16000 (int16 scaled by 1/32768) and processed in 20 s chunks; within each chunk the ~50 Hz frame embeddings were mean-pooled to exactly one value per second (fps = frames/secs; second k pooled over frames [int(k*fps):int((k+1)*fps)]), yielding a 1 Hz audioAffect matrix (n_sec, 1024).

**Acoustic control.** A per-second 13-d low-level acoustic vector (audioLow) comprised RMS, spectral centroid, spectral bandwidth, spectral rolloff at 85% cumulative power, zero-crossing rate, and 8 log-mel energies. The STFT used frame length 400 samples, hop 160, Hanning window; the log-mel filterbank used 8 filters, fmin=50 Hz, fmax=8000 Hz with HTK mel scaling (2595*log10(1+f/700)). Seconds shorter than one STFT frame were set to zeros.

**Encoding and ROIs.** Audio features passed through the same `prep()` pipeline (SCALE=1.042 seconds-per-TR linear interpolation, HRF convolution at TR=1, z-score, PCA to 300, z-score) and the same PLSRegression (n_components=10, scale=False) with 5-fold contiguous-block CV and the mean-over-voxels R2. Positive-control ROIs were primary auditory cortex ('Heschl's Gyrus (includes H1 and H2)') and STG (union of 'Superior Temporal Gyrus, anterior division' and 'Superior Temporal Gyrus, posterior division') from the Harvard-Oxford cortical thr25 atlas, alongside L/R amygdala (subcortical thr25), fusiform, and V1. Two nested variance partitions were computed inline: AAoverAL = R2([audioAffect, audioLow]) - R2(audioLow), and AAoverVIS = R2([audioAffect, VISION]) - R2(VISION), where VISION was the column concatenation of the PCA-prepared emonet, vit, and resnet features. Group inference was one-sample t versus 0 across the globbed participants. This arm uses a single audio affect encoder and therefore does not satisfy criterion (B); it is a single-encoder cross-modal control.

### 4.8 Control analyses

- **Random-encoder null** (`pls20_ctrl`): random-over-ImageNet = R2([random, vit, resnet]) - R2([vit, resnet]) for L/R amygdala, fusiform, and V1, expected ~0 if the EmoNet contribution is signal rather than a dimensionality artifact. This rules out a capacity/dimensionality confound.
- **Temporal (time-shifted) null** (`shifted_null`): the same EmoNet features with time broken by circular shift, shifts = [Tn//4, Tn//2, 3*Tn//4] via `np.roll(E, s, axis=0)`; the null was the mean over the three shifts of R2([roll(E,s), ViT, ResNet]) - R2([ViT, ResNet]), compared to the aligned value by a group paired t of aligned-minus-null. This rules out a purely spectral or drift-driven correlation.
- **Face-presence regressors**: `combined_controls` built a 72%-presence binary face regressor from `500dos_face.1D` (onset/duration coded), HRF-convolved and z-scored, and tested whether EmoNet-unique shrank when FACE was added to the baseline. `mega_harden` used three regressors, F72 plus per-frame face count (FC) and face area (FA) from `rich_face.npz`, each HRF-convolved and z-scored, and tested EmoNet-unique over ViT+ResNet+3xFACE. This rules out mere face-tracking.
- **Second-object swap**: ResNet-over-ViT = R2([resnet, vit]) - R2([vit]) versus EmoNet-over-ViT = R2([emonet, vit]) - R2([vit]), with a paired group difference (Emo minus Res) per region. This tests whether a second generic object encoder recovers the same signal as EmoNet.
- **Eroded high-threshold mask plus hippocampus neighbor** (`mega_harden`): amygdala from Harvard-Oxford subcortical thr50 ('Left/Right Amygdala') with 'Left/Right Hippocampus' added as a spatial-bleed neighbor control (fusiform remained cortical thr25). This rules out smoothing bleed from adjacent structures.
- **Purged/gap cross-validation** (`mega_harden`): a purged 5-fold contiguous-block variant dropping GAP=8 TRs on each side of the test block from the training set. This rules out temporal-autocorrelation leakage across folds.
- **Empirical null plus equivalence bound** (`mega_harden`): 20 random encoders (MRAND=20) formed by projecting a per-frame 32x32x3 = 3072-d pixel matrix through fixed Gaussian random matrices (3072 -> 64, seed `default_rng(1000+m)`). Per participant the 20 random over-ImageNet deltas were averaged; the group null mean nm, SE nse = std(ddof=1)/sqrt(20), and upper 95% artifact bound null_hi = nm + 1.96*nse were reported, with a paired t of EmoNet-minus-subject-null across participants. This places the artifact ceiling relative to the observed EmoNet effect.
- **Second affect-encoder swap** (`second_affect_swap`): a three-way comparison (EmoNet vs ResNet vs the second affect encoder A2 over a ViT baseline), unique-over-ImageNet for EmoNet and A2, redundancy contrasts, and the across-participant correlation of EmoNet-over-ImageNet with A2-over-ImageNet, under the same PAL-1.042 / HRF / z-score / PCA-300 / PLS-10 / 5-fold pipeline with thr50 amygdala and thr25 fusiform. This tests criterion (B), whether two independent affect encoders converge.

### 4.9 Multiple-comparison correction

Contrasts were partitioned into a confirmatory family and an exploratory family. The confirmatory family comprised the amygdala affect-object contrast and the matched within-modality positive-control contrast; these were interpreted directly. We state explicitly that the confirmatory amygdala affect-object contrasts were interpreted uncorrected: the pooled single-neuron composite (boot-p=.033) and the medial-frontal benchmark (boot-p=.040) are therefore uncorrected marginals, and the correction-robust single-neuron result is the left amygdala (per-session faceEmoNet-ResNet p<.001, kragelEmoNet-ResNet p=.001), which we treat as the anchor. Multiple-comparison control over the confirmatory positive-control detection family (movie fusiform, faces fusiform, audio STG) used the dependence-robust Benjamini-Yekutieli (BY) FDR procedure: with m p-values sorted ascending and harmonic constant c = sum(1/i, i=1..m), the step-up adjusted values q = min(prev, p_(rank) * m * c / (rank+1)) were enforced monotone from largest to smallest, with survival at q < 0.05. All three positive-control members were computed under corrected preprocessing: movie fusiform (p<.001, which survives BY on its own), faces fusiform (scene-emotion encoder minus ResNet +0.093, p=.002; re-run), and audio STG (audio-affect over vision +0.0189, p=.001; re-run). Exploratory contrasts (hemispheric lateralization, single-predictor tuning, univariate fear/ambiguity) falling in the .02 to .05 band were reported as exploratory and are noted as not all surviving FWER/FDR control across the family. Specifically, the two univariate parametric contrasts (parametric fear intensity and ambiguity) were treated as a small exploratory family and BY-corrected within it (m = 2, harmonic constant c = 1.5), yielding q = 0.063 for both (raw p = 0.042 and 0.028; Benjamini-Yekutieli step-up with monotonicity, as implemented in the committed univariate_byfdr.py), so neither survives dependence-robust correction.

### 4.10 Software and reproducibility

Analyses used Python (real-data stack, 3.9 to 3.12; numpy<2 required), numpy 1.26.4, scipy (stats, signal), scikit-learn (`cross_decomposition.PLSRegression`, `linear_model.Ridge`), torch 2.2.2, torchvision 0.17.2 (ViT-B/16 IMAGENET1K_V1, ResNet-50 IMAGENET1K_V2, AlexNet), nibabel (>=5,<6), nilearn (>=0.10,<0.13; Harvard-Oxford atlas fetch, `FirstLevelModel`, resampling), opencv-python 4.10.0.84, Pillow (>=10), transformers (>=4.40; audeering wav2vec2), and ffmpeg for frame extraction. The design was preregistered in a frozen region-agnostic specification (version 2, seed 20260705) whose SHA-256 is recorded so post-hoc edits are detectable; the executed primary encoding path used per-participant PLS (n_components=10, PCA-300) rather than the preregistered banded-ridge production path. The banded ridge was run (4.3) and over-regularizes at the single-participant noise floor, compressing even the fusiform positive control roughly a hundredfold, which is why PLS is reported as primary; the banded ridge yields the same qualitative verdict. All data roots are read from environment variables and no machine-specific paths are committed in the executable code. The committed per-analysis result logs are the archived record of the analyses; the corrected-preprocessing re-run logs live under `real_data/scripts/reanalysis/results/` and supersede the same-named pre-correction logs retained elsewhere for provenance (each of the latter carries a `SUPERSEDED` header). The complete analysis code is provided; the feature-extraction step is supplied as a reference implementation that reproduces the pipeline and all qualitative verdicts (signs, significance, the amygdala affect-object comparison, and the equivalence bound). As with any fMRI pipeline, exact numerical reproduction across compute environments is not expected (BLAS/GPU/MPS differences); the original encoder feature caches were lost and were regenerated with a reference extractor (this includes the movie random-encoder features, previously regenerated in place of the original projection seed), and the verdict reproduces at the group level under the regenerated caches. All analysis code, per-analysis result logs, and a reproducibility runner are openly available at https://github.com/patrickiben/amygdala-invariance and archived at Zenodo (doi:10.5281/zenodo.21367398). Public inputs: OpenNeuro ds002837 [8]; OSF 26RHZ [10]; EmoNet weights (OSF amdju); the Toisoul face-EmoNet weights (GitHub); torchvision weights; and the audeering wav2vec2 checkpoint (HuggingFace).

### 4.11 Strong-baseline red-team
To test whether the single-neuron negative inherits weak encoders, we added paradigmatically different stronger frozen encoders and re-ran the pseudo-population RSA. Generic: DINOv2-base (`facebook/dinov2-base`, CLS token, 768-d) and CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, projected image embedding, 512-d). Affect: a facial-expression ViT (`trpakov/vit-face-expression`, penultimate CLS, 768-d). Features were extracted on the seven morph levels exactly as in 4.2, RDMs and Spearman RSA computed as in 4.4, and the affect(best)−object(best) bootstrap contrast (4.5) recomputed for four encoder configurations (original; +stronger generic; +stronger affect; both). The load-bearing conclusion is the generic-side result (a stronger generic encoder matches the amygdala geometry at least as well as the affect encoders) and the representation-invariance across paradigmatically different affect encoders, not that this facial-expression network is the strongest possible affect probe. Script and log: `real_data/scripts/reanalysis/strong_baseline_rsa.py` -> `results/strong_baseline_result.txt`. As a companion multiverse, a specification-curve varied the encoder set, the best-of aggregation (max/mean), the RDM metric (1−Pearson correlation, Spearman, Euclidean), and the neural-pattern normalization over the pooled and left-amygdala samples — analytic choices classified as principled-equivalent (Type E) or uncertain (Type U), with hemisphere reported as a separate facet — and recomputed the affect(best)−object(best) sign for each defensible specification (`spec_curve.py` -> `results/spec_curve_result.txt`, `figures/FigS1_spec_curve.png`). For the movie arm, the same stronger generic encoders were run through the PLS variance-partition pipeline (4.3): per participant, EmoNet-minus-DINOv2 = R2([emonet,vit]) minus R2([dinov2,vit]) for the L/R amygdala and fusiform, testing whether the amygdala's small EmoNet increment survives the strongest generic baseline (`movie_strongbaseline.py` -> `results/movie_strongbaseline_result.txt`).

---

## 5. Limitations
- **Modest effect sizes.** BOLD deep-feature encoding sits near the single-participant noise floor; claims rest
  on the held-out variance partition, corroborated by positive controls. The single-neuron positive account is a
  population mean-t bias, not strong single-unit decoding, and the intensity fraction did not exceed chance. The
  verdict is anchored on the single-neuron affect-object RSA, which is the least noise-floor-limited arm and which
  is significantly negative for an affect code.
- **The movie-arm EmoNet increment and the preprocessing correction.** With canonical EmoNet preprocessing (raw
  0-255 RGB at 227 px, no normalization), EmoNet adds a small unique increment over a matched object encoder in
  the amygdala (EmoNet minus ResNet +0.0009 left, +0.0006 right; EmoNet-unique over ImageNet +0.0022 left, +0.0017
  right). The earlier clean "no EmoNet advantage" movie result (p=.38/.17) was a preprocessing artifact and should
  not be cited. We therefore do not claim a clean movie-arm null; the increment is roughly ten times smaller than
  in the fusiform positive control (+0.0105), is not matched by the second affect encoder (−0.0012 left, n.s.;
  −0.0024 right), and is near the single-participant noise floor. The movie arm is weak, encoder-inconsistent
  corroboration, not the load-bearing evidence.
- **The Figure-3 amygdala divergence is unreliable.** The two-encoder convergence is reliable only in the fusiform
  (r=+0.79, bootstrap 95% CI [+0.39, +0.97], excludes zero). In the amygdala the point estimates (r=−0.28 left,
  −0.00 right) have bootstrap 95% CIs that include zero ([−0.74, +0.10] and [−0.63, +0.59]), so the amygdala result
  is an absence of reliable cross-encoder convergence, not positive evidence of an anti-correlated affect code, and
  should not be read as such.
- **Estimator choice at the noise floor (banded ridge vs PLS).** The preregistered banded-ridge production path
  over-regularizes at the single-participant noise floor: it compresses the fusiform EmoNet-minus-ResNet increment
  roughly a hundredfold (from +0.0105 under PLS to +0.0001, though still p=.003 given a correspondingly tiny
  standard error) and shrinks the amygdala increment to indistinguishable from zero (−0.0000, p=.40 left, .52
  right). We therefore report the fixed-budget PLS estimator as primary, because it preserves the interpretable
  effect magnitudes that make the fusiform-versus-amygdala dissociation legible; the banded ridge is a robustness
  check that yields the same qualitative verdict, indeed a cleaner amygdala null. This estimator dependence affects
  only the BOLD encoding arms; the anchoring single-neuron RSA does not use the encoding estimator.
- **Feature caches regenerated; all four arms re-run under corrected preprocessing.** The corrected-preprocessing
  feature caches were regenerated with a reference extractor (the originals were lost); the group-level verdict
  reproduces. All four arms were re-run end-to-end under corrected EmoNet preprocessing. The face-morph neural RSA
  and the audio affect/low features reproduced their prior values to three decimals (the neural patterns and
  audio features are unchanged); the corrected scene-emotion encoder sharpened the fusiform positive control
  (RSA +0.070 to +0.093), and the corrected vision baseline sharpened the audio STG positive control (audio-affect
  over vision +0.0088 to +0.0189). No arm is reported at a prior-preprocessing value.
- **The movie criterion-(B) partner is a face encoder.** In the movie arm the significant EmoNet increment comes
  from the scene-trained EmoNet, the construct-appropriate encoder for largely non-facial naturalistic affect,
  whereas the second affect encoder (grounds 2-3) is a face-only valence/arousal network. The fusiform two-encoder
  convergence is expected there because fusiform is face-selective cortex; a scene-trained second affect encoder
  would be the stronger criterion-(B) partner for the movie arm, and we flag this as a targeted future control. The
  asymmetry does not apply to the face-stimulus anchor arms (face morphs and single neurons), where the face
  encoder's failure is fully diagnostic.
- **The negative survives a strong-baseline red-team.** Because a limitation claim inherits the strength of its
  strongest-failing baseline, we tested the amygdala verdict against paradigmatically different stronger encoders
  (DINOv2, CLIP; a modern facial-expression network). The affect–object deficit was robust and if anything
  strengthened, and the modern affect encoder matched the geometry worse than EmoNet/Toisoul, so the negative is not
  a weak-affect-encoder artifact — the representation-invariance test the paper applies to others, turned on its own
  claim (§2.3, Methods 4.11).
- **The single-neuron null is for image-computable codes.** It does not exclude an emotion code recoverable from
  population geometry of experimenter-defined states under a task (O'Neill, Posani et al., 2026 [12], mouse). Testing
  cross-condition population-geometry decodability in these 442 units is the most important next analysis.
- **Stimulus dimensionality.** The face-morph datasets are 1-D continua, so affect versus low-level dissociate
  only in multivariate geometry; the finding that object and low-level encoders match the amygdala geometry better
  than affect encoders is the wrong direction for a true emotion code, but a higher-dimensional emotional-stimulus
  battery would strengthen it.
- **Two encoders per construct** for the image arms; the audio arm has one affect encoder and so does not meet
  criterion (B). Vision-only for the image analyses; the audio encoder is prosody-trained.
- **Pseudo-population** single-neuron matrices pool across patients; corroborated by per-session tests.
- **Shared stimulus.** The face-morph fMRI and single-neuron arms use the same seven-level morph stimulus, so
  they are one stimulus measured two ways rather than independent replications; the naturalistic-movie arm is the
  only fully independent dataset.
- **Interoception untested**, and not encoder-shaped.
- **Dataset scope.** A well-powered shared-stimulus emotion-*movie* dataset with timepoint-level annotations now
  exists (Emo-FilM, Morgenroth et al., 2025 [19]) and is an obvious replication target; the genuine gap is a
  well-powered, diverse-emotional-*image*, single-trial dataset with amygdala coverage and shared stimuli.
- **Not a claim that the amygdala is uninvolved in emotion.** It responds to emotional stimuli and its neurons
  are morph-tuned; the claim is specifically about an image- or sound-computable emotion-encoder representation,
  which the amygdala's single-neuron geometry matches worse than generic object and low-level encoders.

---

## Ethics
All data analyzed here are publicly available and de-identified (OpenNeuro ds002837; OSF 26RHZ). The original
data-collection studies (Aliko et al., 2020 [8]; Sun et al., 2023 [10]; Wang et al., 2017 [11]) obtained
institutional ethics approval and participant informed consent. Secondary analysis of public, de-identified data
does not constitute human-subjects research and required no additional institutional-review-board approval or
consent.

## Data & Code Availability
Public datasets: OpenNeuro ds002837 [8]; OSF 26RHZ [10] (fMRI, single-neuron firing rates, stimuli). The amygdala
single-neuron *raw* recordings (Wang et al. 2017 [11]) are request-only for clinical-privacy reasons; the
processed firing rates analyzed here are public in [10]. Encoders: EmoNet [3] (OSF), face-EmoNet [9] (GitHub),
torchvision, audeering wav2vec2 [13] (HuggingFace). All analysis code, per-analysis result logs, and a
reproducibility runner are openly available at https://github.com/patrickiben/amygdala-invariance, archived
at Zenodo (doi:10.5281/zenodo.21367398).

## Competing interests
The author declares no competing interests.

## Funding
This research received no external funding.

## Use of AI
See the AI-use disclosure: analyses and drafting were performed with substantial AI assistance (Anthropic Claude)
under the author's direction and verification; no data were generated or altered by AI (all data are public);
every quantitative claim was verified against committed logs and every reference against primary records. The
author takes full responsibility for the content.

## Author note on scope
This is an exploratory-to-manuscript synthesis; effect sizes are modest and the work is presented with its full
adversarial-control and limitation ledger. Findings are bounded to the datasets analyzed and are offered as a
falsification of a specific class of claim plus a positive characterization, not as a complete model of amygdala
function.

---

## Figure legends

**Figure 1. The affect-minus-object advantage is far larger in modality-appropriate sensory cortex than in the
amygdala, in every arm.** Affect encoder advantage over a capacity-matched object encoder (affect − object), for
the amygdala versus the within-study positive-control region, across the four analyses (movie fMRI encoding;
face-morph fMRI RSA; single-neuron RSA; audio encoding over vision). In every arm the sensory-cortex control
(fusiform for vision, STG for audio; blue) shows a substantial affect advantage, whereas the amygdala (grey) is at
or near zero: significantly negative in the single-neuron arm (affect encoders match the geometry worse than
objects on 3 of 4 per-session contrasts, strongest in the left amygdala; pooled −0.086 vs medial-frontal +0.100)
and in the audio arm, and only tiny, non-significant or near-floor positive increments in the movie (+0.0009,
about ten times smaller than fusiform +0.0105 and not reproduced by a second affect encoder) and face-morph
(+0.010, n.s.) arms. Error bars are approximate ±SE reconstructed from the reported p-values (not measured SEMs;
the significance stars derive from the actual p-values); units differ per arm (ΔR² for encoding, Spearman Δρ for
RSA). The face-morph fMRI RSA arm is an uninformative null (no encoder matched the amygdala geometry at all, p>.2),
distinct from the bounded single-neuron result in Figure 2. All four arms were re-run end-to-end under the
corrected preprocessing (verdicts confirmed; see Limitations).

**Figure 2. The single-neuron amygdala affect-object contrast is bounded far below a detectable effect and
negative in the primary per-session frame.** The 442-unit amygdala affect − object contrast (vermillion;
point estimate −0.083, which owns the plotted bootstrap 95% CI [−0.156, −0.0065]; the same low-level-excluding
max(affect)−max(object) composite has bootstrap median −0.086), against a margin (+0.100) set to the
(marginal) affect effect the same instrument detected in medial-frontal cortex (blue diamond). The 95% upper bound
falls far below the margin in both the primary per-session frame (+0.0063) and the descriptive unit bootstrap
(−0.0065). The verdict rests on the per-session frame, where affect encoders are significantly worse than objects
on three of four encoder contrasts and most clearly in the left amygdala (faceEmoNet−ResNet p<.001); the pooled
composite is significantly negative only in the descriptive bootstrap frame (boot-p=.033). A descriptive
sensitivity analysis indicates ~72% power for the +0.100 margin-sized effect, moot given the per-session result.

**Figure 3. Two independent affect encoders converge reliably in the fusiform but not in the amygdala.** Agreement
(correlation r) between the per-participant unique contributions of two paradigmatically different affect
encoders, by region, with bootstrap 95% CIs. Fusiform shows strong, reliable positive agreement (r = +0.79, CI
[+0.39, +0.97], excludes zero). In the amygdala the point estimates are near zero or negative (r = −0.28 left,
−0.00 right) but the CIs ([−0.74, +0.10] and [−0.63, +0.59]) include zero, so the amygdala shows no reliable
convergence rather than reliable divergence: the absence of a stable, encoder-invariant affect signal is an
unreliable-convergence result, not positive evidence of an anti-correlated code. We report the observed (raw)
bootstrap convergence and its CI rather than a split-half-disattenuated estimate: the per-region encoder
reliabilities are modest (fusiform rel ≈ 0.23–0.54), so the disattenuated correction is unstable (it exceeds the
[−1, 1] range for fusiform and is undefined where a reliability is near zero); the raw fusiform CI excluding zero
is the basis for the reliable-convergence claim.

**Figure 4. What the amygdala does encode: a population bias toward stimulus intensity/salience.** Single-neuron
population mean-t by task variable (n = 442 amygdala units). The intensity/salience term is a significant
population bias (blue); note the per-unit tuned *fraction* for intensity does not exceed chance (6.3%), so this
is a population-level bias, not strong single-unit tuning. The MFC comparison region instead encodes reaction
time (decision effort).

**Figure S1 (supplementary). Specification-curve: the single-neuron verdict is stable across the multiverse of
defensible analytic choices.** The amygdala affect(best)−object(best) contrast, sorted, across 136 defensible
specifications (encoder set, best-of aggregation, RDM metric, neural normalization) for the pooled and left-amygdala
samples. Every specification is ≤ 0 (vermillion): the negative does not depend on any single analytic choice. The
only large mover is hemisphere (the right amygdala is null; reported separately, not shown here).

## References
1. Naselaris T, Kay KN, Nishimoto S, Gallant JL. Encoding and decoding in fMRI. *NeuroImage* 2011.
2. Yamins DLK, DiCarlo JJ. Using goal-driven deep learning models to understand sensory cortex. *Nat Neurosci* 2016.
3. Kragel PA, Reddan MC, LaBar KS, Wager TD. Emotion schemas are embedded in the human visual system. *Sci Adv* 2019;5(7):eaaw4358.
4. Jang G, Kragel PA. Understanding human amygdala function with artificial neural networks. *J Neurosci* 2025;45(18):e1436242025.
5. Kriegeskorte N, Douglas PK. Interpreting encoding and decoding models. *Curr Opin Neurobiol* 2019;55:167–179.
6. Liu Y, et al. Emergence of emotion selectivity in deep networks trained to recognize visual objects. *PLoS Comput Biol* 2024;20(3):e1011943.
7. Gao C, Ajith S, Peelen MV. Object representations drive emotion schemas across a large and diverse set of daily-life scenes. *Commun Biol* 2025;8:697. doi:10.1038/s42003-025-08145-1.
8. Aliko S, Huang J, Gheorghiu F, Meliss S, Skipper JI. A naturalistic neuroimaging database (NNDb). *Sci Data* 2020;7:347 (ds002837; 1.5T).
9. Toisoul A, Kossaifi J, Bulat A, Tzimiropoulos G, Pantic M. Estimation of continuous valence and arousal levels from faces in naturalistic conditions. *Nat Mach Intell* 2021;3:42–50.
10. Sun S, Cao R, Rutishauser U, Yu R, Wang S. A uniform human multimodal dataset for emotion perception and judgment. *Sci Data* 2023;10:773. doi:10.1038/s41597-023-02693-z.
11. Wang S, Tudusciuc O, Mamelak AN, Ross IB, Adolphs R, Rutishauser U. The human amygdala parametrically encodes the intensity of specific facial emotions and their categorical ambiguity. *Nat Commun* 2017;8:14821.
12. O'Neill P-K, Posani L, Meszaros J, Warren P, Schoonover CE, Fink AJP, Fusi S, Salzman CD. The representational geometry of emotional states in basolateral amygdala. *Nat Neurosci* 2026;29(7):1654–1666. doi:10.1038/s41593-026-02315-y (mouse).
13. Wagner J, Triantafyllopoulos A, et al. Dawn of the transformer era in speech emotion recognition: closing the valence gap. *IEEE TPAMI* 2023;45(9):10745–10759. doi:10.1109/TPAMI.2023.3263585.
14. Sander D, Grafman J, Zalla T. The human amygdala: an evolved system for relevance detection. *Rev Neurosci* 2003;14(4):303–316.
15. Peter A, Kim G, DiCarlo JJ. An image-computable characterization of the non-conditioned linkage of visual drive and valence in the primate amygdala. *bioRxiv* 2026.05.22.726311. doi:10.1101/2026.05.22.726311 (macaque).
16. Dupré la Tour T, Eickenberg M, Nunez-Elizalde AO, Gallant JL. Feature-space selection with banded ridge regression. *NeuroImage* 2022.
17. Nili H, Wingfield C, Walther A, Su L, Marslen-Wilson W, Kriegeskorte N. A toolbox for representational similarity analysis. *PLoS Comput Biol* 2014;10(4):e1003553.
18. Soderberg C, Jang G, Kragel PA. Sensory encoding of emotion conveyed by the face and visual context. *SCAN* 2026;21(1):nsag028. (convergent: locates the image-computable facial-emotion code in cortical regions, not the amygdala.)
19. Morgenroth E, Moia S, Vilaclara L, Fournier R, Muszynski M, Ploumitsakou M, Almató-Bellavista M, Vuilleumier P, Van De Ville D. Emo-FilM: a multimodal dataset for affective neuroscience using naturalistic stimuli. *Sci Data* 2025;12:684. doi:10.1038/s41597-025-04803-5.
