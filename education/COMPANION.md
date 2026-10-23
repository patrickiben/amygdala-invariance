# Reading the amygdala invariance study: an educational companion

## Start here

A network trained to recognize emotions can predict some brain activity. Does that mean the brain region contains an emotion-specific code? This study asks what extra comparisons are needed before making that inference.

This guide is for readers new to computational neuroscience. It explains the question, the main measurements, the corrected results and their limits. It is a teaching interpretation of a preprint, not a new brain analysis or a clinical guide.

The narrow conclusion matters: the strongest evidence concerns the **human left-amygdala sample and the tested image-computable representations**. It is not a claim that the amygdala is uninvolved in emotion, that people lack emotions, or that every possible emotion representation has been excluded.

### Which paper this guide follows

The basis is Patrick Iben’s *No emotion-specific image-computable code in the human left amygdala*, **Zenodo record 22314186, version 3**, under concept record 21422754. Read the [exact source version](https://doi.org/10.5281/zenodo.22314186). The deposited PDF is named `MANUSCRIPT_amygdala-false-floor_v2_linenumbered.pdf`; the filename says v2, but the official Zenodo record version is 3. This guide follows the contents of that exact file. Page and line locators below refer to its 25 pages and printed line numbers.

## 1. Prediction is not yet interpretation

An **encoder** converts an image or sound into numerical features. An **affect encoder** is trained on an emotion-related objective, such as scene-emotion categories or facial valence and arousal. A **generic encoder** learns broader visual features, such as objects. **Affect** here is a research term covering emotion-related properties; it does not mean that every affect encoder captures the same psychological construct perfectly.

An **encoding model** learns a mapping from these features to recorded brain responses. If it predicts held-out responses, its features contain useful predictive information. That alone does not show which feature made the prediction work. Faces, luminance, motion and objects can covary with emotion labels.

**Image-computable** means that a model derives its representation from the stimulus image. The term does not cover every possible bodily, contextual, task-dependent or internally generated aspect of emotion.

The paper operationalizes two requirements:

- **Criterion A, comparative advantage:** a construct-trained encoder must beat an appropriate capacity-matched non-construct encoder.
- **Criterion B, convergence:** two independently trained, substantially different construct encoders must agree about the relevant signal. A suitable positive-control region helps show that the assay can detect structure where expected.

Here **capacity-matched** means a comparison designed to reduce differences in model flexibility; it does not mean that the encoders share weights, architecture or pretraining data.

These are the study’s operational tests, not a theorem that automatically proves a brain’s mechanism. The paper reserves **encoder-specific** for the weaker one-encoder result and **construct-invariant** for agreement across encoders. Its title uses “emotion-specific” in the latter sense.

**Where we are:** Introduction, pp. 1–2, lines 30–54; Discussion, pp. 8–9, lines 312–323.

**What the passage does:** it identifies the gap between prediction and an interpretation of what a brain region represents.

**In ordinary language:** two thermometers that both respond to direct sunlight do not establish that the object being measured is producing heat. Compare plausible alternative explanations and instruments.

**Evidence status:** the A/B test is the paper’s methodological proposal. The thermometer example is an analogy, not neuroscience evidence.

**Do not accidentally conclude:** prediction is useless, or passing A and B would by itself establish a causal mechanism.

**Try it:** a face detector and an emotion network predict the same brain region equally well. What could the emotion-network result alone fail to distinguish?

## 2. What is actually measured?

**fMRI** measures changes in the blood-oxygen-level-dependent (**BOLD**) signal, an indirect indicator of neural activity. A **region of interest**, or ROI, is a predefined brain region used to summarize or select measurements. The study also analyzes processed **single-neuron firing rates**, which are a different measurement scale.

**Held-out** predictions are scored on data excluded from that model’s fit. **Cross-validation** repeats this separation across folds. The movie analysis uses five contiguous time blocks, with additional gap-based checks for temporal dependence.

**R²** compares squared prediction error with a mean-response reference. Held-out R² can be negative: prediction can be worse than that reference. A **variance partition** compares two models on the same target and folds. The extra contribution is `R²(full) − R²(reduced)`. The sign can be interpretable even if both absolute values are negative, because the same total sum-of-squares denominator cancels. It does not identify a unique biological cause.

**Representational similarity analysis (RSA)** compares patterns of differences between conditions. For seven face-morph levels, a **representational dissimilarity matrix (RDM)** records how different each pair of levels is. The study compares the upper triangles of neural and encoder RDMs using Spearman rank correlation. This asks whether their relative geometry is alike, rather than whether the network classifies a patient’s emotion correctly.

A **confidence interval** describes uncertainty under a specified sampling and analysis scheme. A **p value** describes how unusual a test statistic would be under the tested null hypothesis; it is not the probability that the scientific interpretation is true. Multiple-comparison corrections account for testing several hypotheses. **BY** below refers to the Benjamini–Yekutieli false-discovery-rate procedure, which allows dependence among tests.

A **pseudo-population** pools neurons recorded across sessions or people into one analysis matrix. Its apparent sample size does not make all neurons independent. A **bootstrap** repeatedly resamples observed data to describe uncertainty under the resampling scheme. Resampling individual units can understate dependence within a session; that is why the paper makes session-based inference primary.

**Where we are:** Methods §§4.1–4.5, pp. 9–14, lines 349–576.

**What the passage does:** it distinguishes predictive fit, relative fit and pattern geometry, and identifies the correct inference unit.

**In ordinary language:** predicting a response, ranking seven response patterns and showing that an emotion model has a special advantage are three different tasks.

**Evidence status:** definitions and reported analysis procedures. This guide has not rerun the brain-data pipeline.

**Do not accidentally conclude:** a ΔR² of 0.0009 means 0.09% of neurons are emotion-specific, or that 442 neurons equal 442 independent participants.

**Try it:** hypothetical models have held-out R² of −0.020 and −0.018 on the same target. What is their difference, and why is it useful despite both being negative?

## 3. The corrected movie result: A passes, B does not

Twenty participants viewed a naturalistic film. The study predicts their regional BOLD signals from scene-affect, face-affect, generic visual and low-level features. It corrects a previous EmoNet preprocessing error: the scene-affect network expects raw 0–255 RGB values resized to 227 pixels, without dividing by 255 or applying ImageNet normalization. The earlier clean “no EmoNet advantage” movie result must not be repeated as the current finding.

| Comparison | Current reported result | What it supports |
|---|---|---|
| Scene-affect EmoNet minus ResNet, over a shared ViT baseline | Left +0.0009, p=.002; right +0.0006, p=.003 | A small significant single-encoder advantage |
| EmoNet minus stronger generic DINOv2 | Left +0.0021; right +0.0015; both p<.001 | The A result is not removed by this stronger generic baseline |
| Second, face-affect encoder minus ResNet | Left −0.0012, p=.29; right −0.0024, p<.001 | The second affect encoder does not reproduce the advantage |
| Agreement between the encoders’ unique contributions | Fusiform r=+0.79, 95% CI [+0.39,+0.97]; left amygdala r=−0.28, CI [−0.74,+0.10]; right r≈0, CI [−0.63,+0.59] | Reliable convergence in fusiform; no reliable convergence in amygdala |

A confidence interval spanning zero does not demonstrate a reliably negative relationship. The left-amygdala negative point estimate therefore cannot be taught as proof that the two emotion models are anti-correlated there.

The fusiform scene-affect advantage over ResNet is +0.0105, roughly ten times the left-amygdala value. That ratio alone is not decisive: regions differ in size and signal quality. The more relevant test asks whether the second affect encoder corroborates the result within the same region.

There is also an important mismatch: the movie’s second affect encoder is face-trained, whereas the first is scene-trained. A second scene-affect encoder would be a stronger test for naturalistic movies. Its absence limits the interpretation of criterion B’s failure.

**Where we are:** Results §2.1, pp. 3–4, lines 59–125; Fig. 3 and legend, p. 23; Methods §§4.2–4.3; Limitations, pp. 19–20.

**What the passage does:** it preserves the correction and distinguishes a positive A result from failure to establish B.

**In ordinary language:** one specialist contributes a little extra prediction, but another specialist does not independently recover the same claimed signal.

**Evidence status:** corrected reported movie results. The paper interprets them as encoder-specific, weak corroboration of its wider argument, not a clean movie-arm absence.

**Do not accidentally conclude:** the movie shows no emotion-related predictive signal, or the generic baseline always wins.

**Try it:** complete this sentence accurately: “In the movie arm, the amygdala ___ criterion A but ___ criterion B.”

## 4. Why the single-neuron result carries more weight

The study analyzes **442 amygdala units**, comprising 277 left and 165 right units across 22 recording sessions in epilepsy patients. It forms within-session pseudo-populations for the primary tests. Requiring at least three units leaves 20 eligible pooled sessions, 14 left-amygdala sessions and 12 right-amygdala sessions. These effective session counts, not 442 independent people, govern the session-based tests.

The strongest reported anchor is left-lateralized: face-affect minus ResNet is **−0.089, per-session p<.001**, and scene-affect minus ResNet is **−0.071, p=.001**. In these comparisons, generic object features match the sampled neural geometry better. The right-amygdala versions are about +0.007 with p=.83 and .78, respectively; neither direction is established there.

The paper also pools units and computes a best-affect-minus-best-object composite. Its point estimate is **−0.083** and bootstrap median **−0.086**, with unit-bootstrap interval **[−0.156,−0.0065]**. These two center values are different estimators of the same composite, not evidence that the authors swapped encoder sets. The bootstrap is descriptive because it resamples units across sessions. Its boot-p=.033 must not replace the primary session-based evidence; the pooled composite’s session-based upper limit is **+0.0063**, so that interval crosses zero.

A secondary bound compares the result with a **+0.100** margin based on a medial-frontal comparison effect. Both +0.0063 and −0.0065 fall below that margin. This excludes a margin-sized advantage under the stated analyses; it does not prove exact equality to zero. The benchmark itself is marginal: its unit-bootstrap p=.040 does not become a firm session-level positive (the reported encoder-pair p values are .108 and .054). The strong left-amygdala negative contrasts are therefore the anchor, rather than a story of uniformly strong controls and total absence everywhere.

The authors add stronger generic encoders and another facial-expression model. A specification curve examines 136 analytic choices for pooled and left samples, with negative contrast throughout. These checks reduce some pipeline concerns; they do not cover every possible encoder, stimulus or biological account. Confirmatory affect-object contrasts are interpreted directly, not all adjusted as one large family; the pooled bootstrap and medial-frontal results remain uncorrected marginals.

**Where we are:** Results §2.3, pp. 5–6, lines 148–222; Methods §§4.5, 4.9, 4.11; Fig. 2 legend, p. 22; Fig. S1 legend, p. 24.

**What the passage does:** it explains why a significant left-session contrast, a pooled bootstrap and a margin comparison are different kinds of evidence.

**In ordinary language:** several measurements from one recording session should not be counted as several independently recruited patients.

**Evidence status:** reported session-based tests carry the main inference; pooled unit-bootstrap and margin results are secondary/descriptive, with their stated limitations.

**Do not accidentally conclude:** the right amygdala has the same demonstrated deficit, the benchmark is a strong positive in every frame, or non-superiority to a margin proves literal zero function.

**Try it:** if an interval includes zero but has an upper limit below +0.100, what claim can it support, and what stronger claim can it not support?

## 5. The other arms and the modest positive account

The face-morph fMRI arm has 19 participants. No encoder reliably matches amygdala geometry there (all p>.2), so this is an **uninformative RSA null**, not the same as demonstrating that a generic encoder fits better. The fusiform scene-affect increment is +0.093, p=.002; the second affect encoder is positive in sign but weaker, p=.06. Reliable two-encoder convergence is established in the movie arm, not by declaring both face-morph effects significant. The fear and ambiguity univariate checks do not survive the reported BY multiple-comparison correction (both q≈.06).

The audio arm analyzes the movie soundtrack with one speech-emotion encoder. Superior temporal gyrus has an audio-affect-over-vision increment of +0.0189, p=.001. The amygdala increments are negative. Because only one audio-affect encoder is used, this is a cross-modal control, **not a complete test of criterion B**.

The movie, face-morph and neuron arms are not four independent modalities. There are **three datasets and two modalities**, organized into four analyses. Face-morph fMRI and neurons use the same seven-level stimulus, so they are one stimulus measured two ways.

The paper’s additional neural regression finds a modest population bias toward stimulus intensity/salience, with mean t about +0.20. The pooled individually intensity-tuned fraction is 6.3% and does not exceed chance by its test. Left decision-uncertainty signals and right valence/choice signals are also described, with exploratory qualifications. This is compatible with a relevance-detection interpretation; it does not prove that this is the amygdala’s complete mechanism. Single-trial predictive model comparisons were at the noise floor and were not interpreted.

**Where we are:** Results §§2.2, 2.4–2.5; Methods §§4.6–4.9; Limitations §5; Figs. 1 and 4 legends (pp. 4–7, 15–17, 19–24).

**What the passage does:** it distinguishes assay failure, encoder disagreement, a one-encoder control and a modest descriptive account.

**In ordinary language:** different experiments answer different questions; putting their results in one figure does not make their units or evidence strength interchangeable.

**Evidence status:** dataset-specific results and a limited interpretation. Fig. 1 uses different units across arms and approximate error bars reconstructed from p values, not directly measured SEMs; do not compare bar heights as a common biological scale.

**Do not accidentally conclude:** the study shows that emotions are absent, that all amygdala neurons encode only salience, or that all four arms independently pass the complete A/B test.

**Try it:** why should a positive result from one audio encoder not be described as two-encoder invariance?

## 6. Reproducibility, limitations and a transfer example

This preprint corrects a substantial preprocessing error. It also changes the primary movie estimator from the preregistered banded-ridge route to fixed-budget partial least squares (**PLS**). Banded ridge is reported as a robustness check but strongly shrinks even the fusiform effect. PLS finds a small set of directions linking input features to responses; ridge uses a penalty that shrinks fitted coefficients. The change must remain visible; the guide does not describe the executed pipeline as an unchanged preregistered analysis.

The paper reports regenerated feature caches after the originals were lost and qualitative agreement under its reference extractor. The current repository separately states that full-data agreement of the reconstructed extractor with archived results remains unverified, and that no preserved extraction script exists for the additional movie DINOv2/CLIP caches. These are distinct statements: a preprint’s reproduction report is not a fresh independent reproduction by this guide or a completion certificate for the current repository.

Processed neural data are public, but raw clinical intracranial recordings are request-only. The copyrighted film is not redistributed. The educational addition does not change these access boundaries. It adds no data, weights or substitute film.

Other scope limits include one-dimensional face morphs, pooling across patients, modest BOLD effects, untested interoception, and the possibility of a task-state population geometry that these stimulus-to-neural tests do not address. A negative result for the tested image-computable encoders is not an impossibility result for every way emotion could be represented.

**Hypothetical transfer example:** a navigation-trained network predicts a brain region. Before calling that region a navigation code, compare matched networks trained on scene recognition or motion, try a second independent navigation encoder, and inspect a relevant positive control. This is a proposed design exercise, not an empirical result. Even convergent prediction would still need causal and task-specific follow-up.

**Where we are:** Methods §§4.3, 4.10–4.11; Limitations §5; Data and Code Availability (pp. 12, 17–21); repository README and real-data reproduction contract.

## Exercises

1. State criterion A and criterion B without using the word “invariance.”
2. Why can an emotion-trained network predict brain activity through generic visual features?
3. Two models have R² −0.020 and −0.018 on the same target. Calculate the second-minus-first change. Does it establish emotion-specific coding?
4. In the corrected movie arm, which criterion passes and which is not established?
5. The left convergence estimate is −0.28 with interval [−0.74,+0.10]. Is this reliable divergence? Explain.
6. Why is a roughly tenfold cross-region effect ratio insufficient by itself?
7. Why are 442 units not 442 independent participants? What are the eligible left-session and right-session counts?
8. Distinguish −0.083 from −0.086. Which inferential frame is primary?
9. An interval’s upper endpoint is +0.0063, below a +0.100 margin, but the interval spans zero. What does this establish and what does it leave unresolved?
10. Identify two reasons not to describe the study as four independent full A/B replications.
11. Rewrite “the study proves the amygdala is not involved in emotion.”
12. Design a stronger criterion-B follow-up for the movie arm. Name one limitation that follow-up alone would not resolve.

## Answer key

1. A: the construct-trained encoder must outperform an appropriate matched non-construct comparison. B: independently trained construct encoders must recover agreeing signal, with a suitable detection/control context.
2. Emotion labels covary with faces, objects, luminance, motion and other features. A model’s training label does not make every learned feature specific to that label.
3. +0.002. It is a relative prediction improvement, not proof of the psychological or causal interpretation.
4. A passes, including against the tested DINOv2 baseline. B is not established: the second affect encoder does not reproduce the advantage and amygdala convergence is unreliable.
5. No. The interval includes zero; the data do not establish a negative relationship. They lack the reliable positive convergence seen in fusiform.
6. Regions differ in size and signal quality. Within-region encoder comparisons are more informative for the specific attribution.
7. Units share sessions/patients. The primary within-hemisphere analyses retain 14 eligible left and 12 right sessions after the at-least-three-units rule.
8. They are the point estimate and bootstrap median for the same best-affect-minus-best-object composite. Session-based encoder-pair inference is primary; the unit bootstrap is descriptive corroboration.
9. It supports being below that margin under the analysis. It does not establish exact equality to zero or, on that interval alone, a negative effect.
10. There are three datasets/two modalities; face-morph fMRI and neurons share stimuli; the audio arm has one affect encoder and cannot assess B; the face-morph amygdala RSA is an uninformative null. Any two suffice.
11. The study finds no reliable construct-invariant image-computable emotion code under its tested encoders and datasets; its strongest comparative evidence is in the sampled left-amygdala neurons. It does not deny emotional involvement.
12. Use an independently trained scene-affect encoder appropriate to naturalistic movies, with matched controls and the same held-out targets. It would not by itself establish a causal emotion mechanism, generalize to all stimuli or repair the independent full-data reproduction gap.

These are self-study answers, not results from administered exercises. No human learner, accessibility or transfer validation has been conducted for this guide.

## Source and provenance

All empirical numbers summarize the exact preprint. No new neural analysis or full scientific validation is claimed. The hypothetical examples above are explicitly teaching devices.

- Exact basis: [Zenodo record 22314186, version 3](https://doi.org/10.5281/zenodo.22314186), `MANUSCRIPT_amygdala-false-floor_v2_linenumbered.pdf`.
- PDF SHA-256: `d4ba2bb793b7e2068da17775ac45c3a39a756c90d3421a4c0dc936437f5c4ae2`.
- The [analysis-code archive](https://doi.org/10.5281/zenodo.21367398) is software, not the manuscript. The current repository’s README and `real_data/REPRODUCE.md` state the execution and input-access boundaries.
- Source checked against the official Zenodo latest-record relation on 23 October 2026. Section, page and line locators are bound to this PDF, not an unversioned future manuscript.

Prepared 23 October 2026 with AI assistance. Review was limited to source fidelity, explanatory structure and arithmetic/claim checks. This is not full G14 completion, independent empirical replication, a human signoff or demonstrated learner benefit.
