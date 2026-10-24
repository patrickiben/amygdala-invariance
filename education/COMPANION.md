# Complete reading companion to the amygdala invariance study

## How to use this complete companion

This is the full teaching route through Patrick Iben’s *No emotion-specific image-computable code in the human left amygdala*. It explains the substantive passages, methods, results, figures and limitations of **Zenodo record 22314186, version 3**. The official PDF’s filename still contains `v2`; the record version is 3. Its SHA-256 is `d4ba2bb793b7e2068da17775ac45c3a39a756c90d3421a4c0dc936437f5c4ae2`.

The [concise study guide](STUDY_GUIDE.md) provides the shorter route. This complete companion follows the source in smaller teaching units. Every numbered unit gives a source locator, a passage role, an explanation, an evidence class, reasoning, a boundary, and an answered check. Cross-references connect results to the methods that produce them. Units revisit a finding when the source changes its role: a results paragraph states an estimate; a methods paragraph defines it; a figure teaches how to read it.

The source is a preprint, not a textbook consensus or a clinical recommendation. Empirical figures below are **reported paper outputs**, not new analyses. Examples labeled hypothetical are teaching arithmetic. A source-level claim is not promoted into a stronger biological claim. These explanations do not establish scientific validity or reproduce the full-data analysis.

### Nonclaims that stay active throughout

- A predictive emotion network does not, by itself, establish an emotion-specific or causal neural mechanism.
- The corrected movie result includes a small significant one-encoder advantage. It is not a clean movie null.
- The strongest neuron comparisons are left-lateralized and session-based. They are not a demonstrated right-amygdala deficit.
- Failure to establish cross-encoder convergence is not reliable negative convergence.
- A bound below a chosen margin does not prove exact zero effect.
- The study tests particular image-computable encoders and datasets. It does not deny the amygdala’s involvement in emotion or exclude every task-state, bodily or contextual code.
- The face-morph and neuron arms share stimuli; the audio arm has one affect encoder. Four analyses are not four independent complete A/B replications.

## Framing and the movie analysis

### A01. What the title and abstract actually claim

**Source locator:** title and Abstract, PDF p. 1, printed lines 1–28; the terminology clarification in Discussion, pp. 8–9, lines 312–323.

**Passage role:** announces the proposed test and the scope of the negative conclusion before the detailed evidence appears.

**Plain-language explanation:** The paper is not asking whether emotional pictures make the amygdala respond. It asks whether an emotion-trained network gives a stable, special description of its responses that alternative visual networks cannot provide. An encoder makes a numerical description of a stimulus. An encoding analysis then relates that description to recorded brain activity. The abstract requires two things: a suitable emotion-trained encoder must outperform a matched generic comparison, and independent emotion encoders must agree about the relevant signal.

These are criteria A and B. The word “emotion-specific” in the title means the stronger, cross-encoder property. The paper calls a gain from only one model “encoder-specific.” That distinction is necessary because the corrected movie analysis does find a small gain from one emotion encoder. Its existence is compatible with the title only under this explicitly stated meaning. The title also says “left” because the strongest neuron contrast is there; the right sample does not show the same demonstrated advantage for generic encoders.

**Evidence type:** methodological definition plus a summary of reported results. The abstract is not an extra dataset, and repeating its claims does not independently corroborate them.

**Reasoning or worked illustration:** Imagine two image models trained to recognize emotions. One predicts a region slightly better than an object model, while the other does not. The first comparison establishes a property of that particular model. To attribute the result to the common construct “emotion,” the study asks for agreement across models. This hypothetical pattern describes the logic, not an additional measurement.

**Boundary/nonclaim:** the title does not mean “the amygdala does nothing emotional.” The study does not test all possible encoders or all forms of neural coding.

**Self-check:** Why does a significant one-encoder result not automatically contradict this title?

**Answer:** The title uses emotion-specific to mean construct-invariant. A single-encoder gain can pass A without establishing B.

### A02. Why training labels do not determine a model’s neural meaning

**Source locator:** Introduction, PDF pp. 1–2, lines 30–54; Discussion, pp. 8–9, lines 293–345.

**Passage role:** identifies the inference being challenged and places the work beside prior encoding and cortical-localization studies.

**Plain-language explanation:** A network trained on emotion labels learns whatever features help its training task. Some may concern faces, objects, color, spatial composition or motion rather than emotion as a distinct psychological variable. Those same features can predict brain activity. Consequently, “an emotion network predicts this region” has more than one explanation. The region might track emotion-related information, generic stimulus structure, or a mixture.

The paper asks for comparisons that can distinguish these accounts. A capacity-matched control is intended to reduce a flexibility advantage: the emotion model should not win simply because it provides a richer fitting opportunity. Matching does not make the networks identical in architecture, data or every learned property. The introduction also acknowledges recent work locating image-computable facial-emotion structure in cortex. The paper presents its cortical result as convergent with that literature. Its proposed contribution is the disciplined comparison across encoders, neurons and sound, rather than claiming that every earlier researcher attributed all emotional function to the amygdala.

**Evidence type:** interpretation of an inference problem and positioning within cited research. The companion does not independently review every cited article or endorse a field-wide consensus.

**Reasoning or worked illustration:** In a hypothetical image collection, frightened faces appear against dark backgrounds. A model can learn darkness because darkness predicts the label. If a brain region tracks brightness, the emotion-trained model may predict it without the region representing fear. Holding out images reduces fitting optimism but does not remove that conceptual confound. A brightness or generic-vision control addresses a different question from cross-validation.

**Boundary/nonclaim:** prediction remains useful. The concern is what the prediction licenses us to say, not whether every predictive correlation is meaningless. Nor does a failed control comparison prove the region uses the control network’s exact mechanism.

**Self-check:** What does held-out prediction fix, and what can it leave unresolved in the darkness example?

**Answer:** It tests generalization of the fitted mapping to held-out data. It can leave unresolved whether darkness or emotion-related information drives the predictive relationship.

### A03. Learning the encoder cast before comparing it

**Source locator:** Results “Encoders (cast),” PDF p. 2; Methods §4.2, pp. 10–12; Methods §4.11, p. 18.

**Passage role:** fixes the identities and roles of models whose similar names recur across analyses.

**Plain-language explanation:** The scene-affect network is Kragel EmoNet, an AlexNet-based network trained on scene-emotion categories. In code it can appear as `kragelEmoNet`. The second affect network is a facial valence/arousal model, commonly called face-EmoNet or `faceEmoNet`. Despite the shared name, these are different networks with different training tasks and input conventions. ResNet-50 and ViT-B/16 supply object-trained visual comparisons.

DINOv2 and CLIP are added as stronger generic alternatives in the robustness work. A low-level model supplies features such as luminance, motion or image gradients; its exact contents differ between the movie and face arms. The audio arm uses a speech-emotion wav2vec2 model, not an image model applied to sound. Keeping these identities straight prevents a false impression that the study tested the same model twice or that every “emotion” network is equally appropriate for every stimulus. It also makes source-code aliases interpretable when tracing a reported result to a log.

**Evidence type:** model definitions and reported implementation choices. The cast table assigns roles; it does not itself establish that controls are perfectly matched or that a model exhausts its psychological construct.

**Reasoning or worked illustration:** Consider a hypothetical replication that replaces scene EmoNet with face-EmoNet merely because both names contain EmoNet. Even before checking output dimensions, that substitution changes the training objective and often the preprocessing. It would therefore not reproduce the same comparison. An exact checkpoint, feature layer and input transform are more informative than a family nickname.

**Boundary/nonclaim:** “independent affect encoders” means distinct trained representations for the comparison. It does not guarantee statistical independence of every training image or complete agreement about what emotion means. The source later acknowledges the face-versus-scene mismatch in the movie test.

**Self-check:** Why should a methods record specify more than “we used EmoNet”?

**Answer:** That name can refer to different models. Reproduction needs the particular network, weights, feature layer, preprocessing and intended comparison role.

### A04. The movie dataset and the preprocessing correction

**Source locator:** Results §2.1, PDF p. 3, lines 59–90; Methods §§4.1–4.2, pp. 9–11, lines 349–422; Limitations, p. 19, lines 745–780.

**Passage role:** describes the naturalistic dataset and discloses a correction that changes a previously reported result.

**Plain-language explanation:** Twenty participants watched *500 Days of Summer* in the Naturalistic Neuroimaging Database. The analysis uses released denoised, unsmoothed MNI-space derivatives rather than collecting new scans. Movie frames become encoder features, which are aligned to the BOLD time series. The crucial correction concerns scene EmoNet’s inputs. Its reference implementation expects RGB values on the raw 0–255 scale at 227 by 227 pixels, without the usual division by 255 and without ImageNet mean/standard-deviation normalization.

An earlier version scaled the inputs incorrectly. Under the corrected input convention, EmoNet has a small significant advantage over the matched object comparison in the amygdala. The previous clean “no advantage” movie conclusion is therefore superseded. A model can accept a wrongly scaled tensor and still produce numbers; successful execution is not proof that the representation is the intended one. The corrected result must remain visible even though it weakens one route to the manuscript’s broader interpretation.

**Evidence type:** reported dataset, extraction procedure and substantive correction. The companion explains the correction but does not repeat feature extraction from the copyrighted film.

**Reasoning or worked illustration:** A raw pixel value of 128 becomes about 0.502 after dividing by 255. Those are not interchangeable inputs for a network whose weights expect the first scale. This is explanatory arithmetic. It does not tell us how much any neural result must change; that consequence requires the actual rerun reported in the paper.

**Boundary/nonclaim:** correcting EmoNet does not make every processing choice validated. The current repository states a separate full-data reconstruction limitation. Nor may the old p=.38/.17 values be quietly retained as present evidence.

**Self-check:** If a corrected run changes the sign or significance of a control comparison, should a teaching guide keep the cleaner old conclusion for simplicity?

**Answer:** No. It must explain the correction and teach the current result, including the interpretive complication the correction introduces.

### A05. Why a four-percent timing error matters

**Source locator:** Results §2.1, PDF p. 3, lines 70–74; Methods §4.2 “PAL frame-rate alignment,” pp. 11–12, lines 445–454.

**Passage role:** explains stimulus-to-brain alignment rather than treating the film and scan clocks as automatically synchronized.

**Plain-language explanation:** BOLD at a particular scan time must be compared with features from the corresponding part of the film. The analyzed rip and the presentation use different frame rates: 23.976 and 25 frames per second. A small proportional mismatch accumulates during a long movie. The implementation uses a fixed scale of 1.042, based on the ratio 25/23.976, and linearly samples features at that stretched index.

It clips indices at the available cache boundary. The paper reports a small scale sweep in which prediction of visual control regions peaked at this choice. This is a practical reason to inspect timing, not a license to choose any time warp that maximizes the target region’s effect. A correction based on known source/presentation rates and checks in appropriate control regions is more interpretable than a flexible alignment fitted separately to each desired result.

**Evidence type:** reported alignment procedure and empirical control check. The numerical ratio below is independently understandable arithmetic, not an independent validation of the original film timestamps.

**Reasoning or worked illustration:** The ratio is approximately 1.0427. At a nominal index of 1,000 seconds, a 4.2% scale difference corresponds to roughly 42 seconds. This hypothetical index calculation shows why apparently small rate discrepancies can matter. It does not imply a constant 42-second shift: proportional stretching creates a growing displacement rather than one fixed offset. The source’s actual interpolation convention is the authority for the analysis.

**Boundary/nonclaim:** the scale sweep is not a separate sample of participants. An alignment check cannot by itself establish construct specificity. Clipping or missing tail frames can also matter in a new dataset; copying the constant into a different film would be unjustified.

**Self-check:** Why would a single fixed offset fail to correct a true frame-rate mismatch over an entire movie?

**Answer:** A rate mismatch accumulates with elapsed time. A fixed offset can align one moment while leaving increasing errors elsewhere; proportional resampling addresses the clock-rate discrepancy.

### A06. What the movie model fits and what it holds out

**Source locator:** Methods §4.3, PDF p. 12, lines 455–474; §4.3 held-out R², pp. 12–13, lines 487–499; §4.8 purged/gap control, p. 16.

**Passage role:** specifies the feature preparation, predictive estimator and cross-validation units that turn encoder features into a score.

**Plain-language explanation:** Features are resampled to the movie clock, convolved with a hemodynamic response function, standardized, and reduced to at most 300 principal components before another standardization. The hemodynamic convolution represents the delayed and spread-out BOLD response; a frame does not produce a fully instantaneous fMRI measurement. Partial least squares, or PLS, then fits a fixed low-dimensional relationship between features and regional voxel responses, using at most ten components. Five contiguous time blocks are held out in turn. The remaining blocks train the mapping, and the held-out predictions are assembled into a full time series.

Contiguous blocks are more relevant than randomly mixing nearby seconds because neighboring measurements can be correlated. Additional controls remove an eight-TR gap around a test block to examine leakage from adjacent times. The stated hemodynamic kernel is a normalized `GammaPDF(t;6) − GammaPDF(t;16)/6` over the specified time grid. Its positive and delayed negative terms model a response shape, not emotion. PCA and standardization are not emotion models in themselves; they prepare a common fitting route through representations with different original dimensions.

**Evidence type:** reported analysis procedure. This is a description of the executed estimator, not a claim that every preprocessing operation was independently cross-fitted in the same way as PLS.

**Reasoning or worked illustration:** In a hypothetical five-block sequence A–B–C–D–E, predicting C after training on A, B, D and E tests an unseen interval. However, the last moment of B may resemble the first moment of C. A gap check removes nearby training moments and asks whether such proximity explains the apparent performance. This illustration explains why the control exists; it does not provide its numerical outcome.

**Boundary/nonclaim:** held-out fitting reduces one form of overfitting, but does not solve feature interpretation, stimulus confounding or all temporal dependence. Equal component budgets do not make pretraining corpora identical.

**Self-check:** Does “frozen encoder” mean no fitting happens anywhere in this analysis?

**Answer:** No. Encoder weights stay fixed, but the encoding mapping from features to each participant’s brain responses is fitted on training folds.

### A07. Reading a variance partition when absolute R² is negative

**Source locator:** Methods §4.3, PDF pp. 12–13, lines 487–499; Results §2.1, p. 4, lines 107–110.

**Passage role:** defines the predictive comparison and explains why a relative improvement remains interpretable near the single-participant noise floor.

**Plain-language explanation:** R² compares a model’s squared prediction errors with the variation around a mean-response reference. On held-out data, it can be negative: a fitted model can predict worse than that reference. The paper compares a full feature set with a reduced feature set on the same target responses and folds. An EmoNet-unique-over-ImageNet partition, for example, adds EmoNet features to ViT and ResNet features and subtracts the reduced model’s R².

The score is `R² = 1 − SS_res/SS_tot`; the difference is `(SS_res,reduced − SS_res,full)/SS_tot`. Both scores use the same total sum-of-squares denominator. Their difference therefore tracks the reduction in held-out residual error, even if neither absolute score is positive. The phrase “unique variance” describes this model-set comparison. It is not an anatomical partition of neurons and does not prove that every extra predictable feature is psychologically unique to emotion.

**Evidence type:** definition and algebra underlying the reported estimator. The source’s comparisons use actual fitted models; the small example below is hypothetical teaching arithmetic.

**Reasoning or worked illustration:** Suppose the reduced model has R²=−0.020 and the full model −0.018 on exactly the same target. The full-minus-reduced difference is +0.002. Both models perform worse than the mean reference, yet the full model has less residual error. Algebraically the shared denominator lets us compare them consistently. If the targets or folds differed, that cancellation argument would not automatically apply. A positive difference still needs uncertainty estimation and controls before a construct interpretation is earned.

**Boundary/nonclaim:** +0.002 is not “0.2% of neurons encode emotion.” It is a change in the specified predictive score. Comparing two region-level R² increments also does not automatically compare equal biological signal quality across regions.

**Self-check:** What two conditions make the residual-error cancellation explanation appropriate here?

**Answer:** The full and reduced scores use identical target data and the same scoring denominator, with the same held-out comparison structure. They are not unrelated scores from different tasks or samples.

### A08. Criterion A really does pass in the corrected movie arm

**Source locator:** Results §2.1, PDF pp. 3–4, lines 82–125; Methods §4.11 movie strong-baseline comparison, p. 18, lines 731–736.

**Passage role:** reports the corrected positive comparison and tests whether it disappears against a stronger generic representation.

**Plain-language explanation:** The scene-affect network adds prediction beyond the object-feature baseline. More specifically, EmoNet minus ResNet over a shared ViT baseline is +0.0009 on the left and +0.0006 on the right, with p=.002 and .003. These are small but statistically significant reported increments. The study then challenges the comparison with DINOv2.

EmoNet minus DINOv2 is +0.0021 left and +0.0015 right, both p<.001. A stronger generic alternative therefore does not erase this one-encoder result. This matters because a paper about weak baselines should apply its own criticism to its comparisons. The result prevents the guide from summarizing the movie evidence as “generic vision always does just as well.” Instead, the movie arm passes criterion A. Whether that gain supports a common emotion construct depends on the different question in criterion B, developed in the next unit.

**Evidence type:** corrected reported predictive results across 20 participants. The p values refer to the specified participant-level comparisons, not the chance that the biological theory is true.

**Reasoning or worked illustration:** The source’s A test asks about an incremental comparison within a region. Replacing ResNet with DINOv2 changes the comparison, so the size of the increment can change as well. It is not contradictory for EmoNet’s gain over one generic model to be larger than over another. “Stronger” is a general model characterization, not a guarantee of higher score in every regional encoding task.

**Boundary/nonclaim:** the increments do not establish a causal emotion mechanism. Their small scale is not sufficient reason to discard them, and their significance is not sufficient reason to promote them to construct invariance. The source reports PLS rather than the preregistered ridge estimator as primary; that choice remains a separate qualification.

**Self-check:** Which inaccurate summaries should be rejected: “no movie effect” and “the movie proves an invariant emotion code”?

**Answer:** Both. The corrected data show a small one-encoder gain, while the cross-encoder requirement remains unestablished.

### A09. Criterion B asks for convergence, not another small p value

**Source locator:** Results §2.1, PDF pp. 3–4, lines 98–125; Fig. 3 and legend, p. 23, lines 886–897; Limitations, pp. 19–20.

**Passage role:** asks whether the movie advantage belongs to a shared affect construct rather than one particular model.

**Plain-language explanation:** The second affect encoder does not reproduce the scene-affect advantage. Relative to ResNet its reported movie increment is −0.0012 on the left, p=.29, and −0.0024 on the right, p<.001. The study also correlates the two affect encoders’ unique contributions across participants. In fusiform, the correlation is +0.79 with bootstrap interval [+0.39,+0.97].

In the amygdala, it is −0.28 on the left and approximately zero on the right, with intervals [−0.74,+0.10] and [−0.63,+0.59]. Neither amygdala interval establishes a reliably negative relationship. The finding is a lack of reliable positive convergence, not demonstrated anti-correlation. This is a different statistical question from whether one encoder has a positive average increment. Two models could both have positive means yet disagree about which participants show their strongest contributions.

**Evidence type:** reported cross-encoder comparison and bootstrap uncertainty. The result is weaker than proving that no shared affect representation exists.

**Reasoning or worked illustration:** In a hypothetical class, two examiners can assign the same average mark while ranking students differently. Agreement of averages does not establish agreement in individual patterns. Conversely, a noisy negative correlation cannot be treated as reliable disagreement without uncertainty. This analogy explains the convergence test; it is not evidence about participants. The face-versus-scene training difference also matters: a second scene-affect model would be a more directly matched movie partner than a face-trained model.

**Boundary/nonclaim:** failure of B can arise partly from model inadequacy, noise or construct mismatch. It is not a proof that every possible emotion representation is absent. The paper makes this limitation explicit and treats the movie as weaker corroboration than the left-neuron anchor.

**Self-check:** Does an amygdala correlation of −0.28 with an interval spanning zero establish reliable divergence?

**Answer:** No. It does not establish the sign. The justified claim is that reliable convergence was not demonstrated under this comparison.

### A10. The movie control battery checks several different alternatives

**Source locator:** Results §2.1, PDF p. 4, lines 107–110; Methods §4.8, pp. 16–17, lines 618–658.

**Passage role:** lists targeted attempts to explain the predictive increment without the intended stimulus-related signal.

**Plain-language explanation:** The controls are not interchangeable repetitions of one test. A random encoder asks whether feature dimensionality or arbitrary flexibility explains a gain. A circular time shift breaks stimulus-to-response alignment while retaining much of the feature sequence’s structure. Face-presence, face-count and face-area regressors ask whether simple face tracking explains the contribution. A second-object swap asks whether adding another generic representation gives the same advantage. A stricter, eroded amygdala mask and a hippocampus neighbor check address spatial mixing with nearby tissue.

Gap cross-validation examines dependence across neighboring train/test times. An empirical ensemble of random projections provides an artifact reference. Finally, the second-affect swap addresses criterion B. Each control narrows a particular alternative. None independently establishes the entire psychological interpretation. A guide that compresses them into “all confounds ruled out” loses what was actually tested.

**Evidence type:** reported robustness procedures and comparisons. These controls are largely applied to the same dataset; they do not recruit independent replication samples.

**Reasoning or worked illustration:** Suppose a hypothetical movie feature is high whenever a face fills the frame. A time shift may show that the predictor is genuinely synchronized to the film, yet face area could still explain its advantage. Adding a face-area baseline tests that explanation more directly. If the advantage survives, it narrows the face-area account but does not exclude every face property. Similarly, spatial-mask controls and timing controls attack different sources of error, so one cannot substitute for the other.

**Boundary/nonclaim:** the source sometimes describes controls with strong “rules out” language. Teach the tested alternative at the stated operationalization rather than an unlimited guarantee. Multiple controls can increase confidence in an encoder-specific relationship while B remains unestablished.

**Self-check:** Why is a temporally aligned effect not automatically emotion-specific?

**Answer:** It can be aligned because it tracks any film-linked feature, including generic visual structure. Temporal specificity and construct specificity are separate properties.

## Face-morph fMRI and single-neuron evidence

### A11. What the face-morph fMRI sample contains

**Source locator:** Results §2.2, PDF pp. 4–5, lines 126–147; Methods §4.1, pp. 10–11, lines 363–392.

**Passage role:** introduces a task-evoked comparison intended to complement passive naturalistic viewing.

**Plain-language explanation:** The face experiment presents a fear-to-happy morph continuum at seven levels. It is not a broad collection of all emotional situations. The analysis retains 19 participants with the required first-level design information. It uses supplied normalized and smoothed volumes and refits a first-level model rather than simply reading precomputed contrasts. Each expression level gets a beta map: an estimated spatial response pattern for that condition.

Motion parameters enter as nuisance covariates. The study also checks amygdala coverage and temporal signal-to-noise ratio so an absent encoder match is not casually attributed to an entirely missing region. The reported tSNR range is approximately 140–190, computed at runtime and not separately logged. That approximate range should not become a new precise verified dataset in a teaching document. This arm differs from the movie in task, stimulus range, acquisition and analysis; comparisons across arms need those distinctions.

**Evidence type:** reported sample selection, preprocessing and quality checks. These establish what was analyzed, not a successful construct-specific result.

**Reasoning or worked illustration:** A hypothetical participant can have usable whole-brain scans but a cropped amygdala field of view. A regional null from those scans would be difficult to interpret. Coverage checks address that obvious failure. However, retaining the region does not guarantee that seven closely related expressions provide enough representational variation for every encoder comparison. Physical coverage and an informative psychological stimulus space are different requirements.

**Boundary/nonclaim:** this task does not independently vary all visual features from all emotional properties. A fear-to-happy continuum can align emotional and low-level image changes. Good tSNR does not prove that every null is informative, and the source later describes this RSA null as uninformative.

**Self-check:** What does the coverage gate establish, and what does it not establish?

**Answer:** It checks that relevant amygdala voxels and usable signal are present. It does not establish construct specificity, adequate stimulus diversity or an informative fit for every encoder.

### A12. Turning seven response patterns into an RSA comparison

**Source locator:** Methods §4.4, PDF pp. 13–14, lines 509–532; §4.2 face-morph features, p. 11, lines 432–444.

**Passage role:** defines exactly what “matches the geometry” means in the face analysis.

**Plain-language explanation:** For each participant and region, the analysis forms a matrix with seven condition rows and voxel columns. It standardizes each condition pattern across voxels. A representational dissimilarity matrix then compares each pair of conditions using correlation distance, one minus their Pearson correlation. The same construction is applied to the encoder features averaged within expression levels.

Only the upper triangle is compared, because the lower triangle repeats the same pairwise relationships and the diagonal compares each condition with itself. With seven conditions there are 7×6/2=21 distinct pairs. Spearman correlation asks whether the ordering of neural dissimilarities resembles the ordering of encoder dissimilarities. It does not train a new emotion classifier on the participants. Correctly mapping the experimental expression codes to feature rows is essential; the paper explicitly gives the ordering rather than assuming file order matches the psychological continuum.

**Evidence type:** estimator definition and reported implementation. The 21-pair count is elementary combinatorial arithmetic, not a claim of 21 independent biological observations.

**Reasoning or worked illustration:** Imagine conditions A, B and C. If the brain treats A and B as similar but both as unlike C, an encoder matches that small geometry if it produces a comparable relative pattern. The actual firing magnitude need not be identical to the feature magnitude. Rank-based comparison also differs from testing whether each pairwise distance has the same absolute value. This miniature example illustrates the estimator without reproducing the real seven-level RDM.

**Boundary/nonclaim:** the 21 distances share conditions and are dependent. They are not 21 independent participants. A high RSA correlation supports a particular pattern relationship, not the model’s exact neural mechanism or a clinical ability to recognize emotion.

**Self-check:** Why compare only the upper triangle and exclude the diagonal?

**Answer:** It contains each distinct condition pair once. The diagonal is self-comparison and the lower triangle duplicates the upper one.

### A13. A responsive amygdala can still give an uninformative RSA null

**Source locator:** Results §2.2, PDF pp. 4–5, lines 126–147; Methods §§4.4 and 4.9; Fig. 1 legend, p. 22.

**Passage role:** distinguishes univariate responsiveness, a failed representation match and a positive-control result.

**Plain-language explanation:** A univariate analysis summarizes the region’s average response. RSA asks about the spatial pattern of differences across conditions. These are different measurements. The paper reports raw univariate p=.042 for left fear intensity and p=.028 for right ambiguity. Both become q≈.06 under the stated dependence-robust correction, so neither survives the corrected threshold. More importantly, neither proves encoder specificity.

In RSA, no tested encoder reliably matches amygdala geometry, with all p>.2. There is then little stable encoder fit on which to base an affect-versus-object distinction. The paper appropriately calls this an uninformative null. The fusiform scene-affect-minus-ResNet increment is +0.093, p=.002; the face-affect encoder points the same way but is weaker, p=.06. Do not turn that into two significant face-morph controls. Reliable two-encoder convergence is established in the movie arm, not supplied automatically by this figure.

The secondary univariate checks use a fear-linear regressor and ambiguity `−(fe−50)^2`, an inverted U highest at the morph midpoint. Separately, the methods compute leave-one-expression-level-out Ridge prediction with penalty α=1 after reducing encoder features to three principal components. These are specified secondary procedures; their mere computation is not a new successful prediction result.

**Evidence type:** reported fMRI findings with different inferential status. The null is not an equivalence test that excludes all meaningful regional structure.

**Reasoning or worked illustration:** A hypothetical region can become more active for several expressions while its pattern differences remain too noisy to rank reliably. Averaging across voxels and comparing spatial geometry answer different questions. Conversely, a structured pattern might exist without a large mean shift. Treating either measurement as a substitute for the other confuses the task each estimator performs.

**Boundary/nonclaim:** failure of all encoders is not proof that the region contains no emotional information. It may indicate that these models, these seven conditions or this measurement do not support the needed comparison. Corrected nonsignificance also does not prove exact zero univariate response.

**Self-check:** Is this face-morph result the same evidence as finding that an object encoder significantly outperforms an affect encoder?

**Answer:** No. Here no encoder reliably matches the regional geometry, so the comparison lacks a stable matching signal. The neuron arm supplies a different, stronger comparison.

### A14. Counting neurons, sessions and people honestly

**Source locator:** Results §2.3, PDF p. 5, lines 148–166; Methods §§4.1 and 4.5, pp. 10 and 14, lines 386–392, 541–555.

**Passage role:** identifies the recording sample and the unit used for primary uncertainty estimates.

**Plain-language explanation:** The source reports 442 amygdala units: 277 left and 165 right, recorded across 22 sessions in epilepsy patients. These are not 442 independently recruited people. Units recorded in the same session share stimuli, recording context and patient characteristics. The paper therefore makes within-session comparisons and tests their distribution across eligible sessions.

A session must contain at least three units in the region under analysis. That rule leaves 20 sessions for the pooled amygdala, 14 for the left, 12 for the right and 15 for the medial-frontal comparison. These differing counts matter for degrees of freedom and uncertainty. A headline sample total can be useful for describing the amount of neural recording, but it cannot replace the denominator of the primary group test. Calling the analysis “single-neuron” describes the recording resolution, not a guarantee of independent inference at the unit level.

**Evidence type:** reported sample composition and inference design. The sum 277+165=442 checks internal arithmetic; it is not a separate validation of the recordings.

**Reasoning or worked illustration:** Suppose a hypothetical session records 30 neurons and another records three. Treating all 33 as independent human replications gives the first session ten times the weight in a simple unit analysis and ignores shared conditions. A session-based comparison asks whether the effect recurs across recording contexts. It may be less precise, but the precision is tied more honestly to the nesting structure. This is why the source distinguishes descriptive unit bootstrap from primary session tests.

**Boundary/nonclaim:** session-based inference does not erase every dependence across sessions or automatically generalize from epilepsy patients to all humans. Nor are the left and right effective sample counts interchangeable just because their unit totals sum to the same pooled number.

**Self-check:** Which number is most relevant to the degrees of freedom of the left per-session test: 277 units or 14 eligible sessions?

**Answer:** The 14 eligible sessions; a one-sample t test then has 13 degrees of freedom, as stated in the paper.

### A15. How a neural pseudo-population is constructed

**Source locator:** Methods §4.5, PDF p. 14, lines 533–550; §4.4 geometry definition, p. 13.

**Passage role:** describes the transformation from trial firing data into the seven-condition geometry used by the neuron analysis.

**Plain-language explanation:** For each unit, the response is the trial count after subtracting a baseline count. The analysis restricts attention to fear–happy trials and averages responses within the seven expression levels. Units with a missing level are dropped from the tuning-vector calculation. Each remaining seven-level vector is standardized, emphasizing the shape of tuning across conditions rather than its overall firing magnitude.

Combining vectors creates a units-by-seven matrix. The correlation-distance RDM then compares expression conditions using units as features. A pooled pseudo-population combines units not all recorded simultaneously. Within-session pseudo-populations make the primary session comparisons. These transformations give a specific, reproducible meaning to “neural geometry.” They also delimit what is discarded: absolute response levels, missing-level units and aspects of timing not retained in the supplied summaries do not enter this representation in the same way.

**Evidence type:** reported analysis construction, not a claim of simultaneous population dynamics. The source uses processed public firing-rate data; it does not require access to the raw clinical recordings for this analysis.

**Reasoning or worked illustration:** A hypothetical neuron responding [1,2,3] and another responding [10,20,30] have the same rising tuning shape after standardization. If the question is about pattern geometry, that can be sensible. If the question were about absolute firing or energy use, the same normalization would discard relevant information. A transformation is useful relative to a question; it is not a neutral window onto every biological property.

**Boundary/nonclaim:** a pseudo-population match does not show that those exact neurons jointly implement an encoder in real time. Missing-level exclusion and normalization are analytic choices; the multiverse checks some alternatives but not every possible population description.

**Self-check:** Why can standardized tuning support a shape comparison while failing to preserve absolute firing differences?

**Answer:** Centering and scaling remove each unit’s baseline and overall scale. They retain relative variation across expression levels rather than the original response magnitude.

### A16. Where the left-neuron anchor is strong, and the right is unresolved

**Source locator:** Results §2.3, PDF p. 5, lines 153–177; Discussion, pp. 7–8, lines 263–275; Methods §4.9.

**Passage role:** reports the primary session-level encoder comparisons and narrows the strongest claim to the supported hemisphere.

**Plain-language explanation:** Across eligible pooled sessions, three of four affect-versus-object encoder-pair contrasts are negative and significant at the directly interpreted thresholds. The clearest comparisons are in the left amygdala: face-affect minus ResNet is −0.089, p<.001, and scene-affect minus ResNet is −0.071, p=.001. Negative means the generic object representation matches the sampled neural geometry better under this estimator. In the right amygdala, the two corresponding contrasts are approximately +0.007, with p=.83 and .78.

Those right-side tests do not establish either an affect advantage or a generic advantage. The source therefore treats the left result as the correction-robust anchor. This differs from using the pooled unit-bootstrap composite as the primary test. It also differs from concluding “both hemispheres are the same” because one side is nonsignificant. A difference in significance is not itself a formal test of a difference between hemispheres.

**Evidence type:** reported primary per-session contrasts. Some pooled p values are modest; the source explicitly emphasizes the strongest left comparisons rather than describing every contrast as equally decisive.

**Reasoning or worked illustration:** If a hypothetical session has object RSA 0.40 and affect RSA 0.31, its affect-minus-object contrast is −0.09. The primary inference asks about such paired contrasts across eligible sessions. That hypothetical subtraction illustrates the sign convention; it is not the source’s raw session data. Paired comparisons reduce some nuisance differences because both encoders face the same session’s conditions.

**Boundary/nonclaim:** these results concern particular encoders, stimuli and recorded samples. They do not prove a universal generic-vision mechanism, deny emotional responsiveness or establish a right-amygdala deficit.

**Self-check:** What is wrong with “the right amygdala also prefers objects because its affect comparison is not significant”?

**Answer:** Nonsignificance does not establish an object advantage. The reported right contrasts are small positive estimates with broad uncertainty, so neither direction is established.

### A17. One composite, two center estimates and two inferential frames

**Source locator:** Results §2.3, PDF p. 5, lines 167–188; Methods §4.5, pp. 14–15, lines 541–576; Fig. 2 legend, p. 22.

**Passage role:** reports descriptive pooled evidence while warning that its uncertainty is not the primary session-based inference.

**Plain-language explanation:** The composite selects the best matching affect encoder and subtracts the best matching object encoder. Low-level features are excluded from these two sets. On the pooled sample its direct point estimate is −0.083. Resampling units and recomputing the composite gives a bootstrap median of −0.086 and a percentile interval [−0.156,−0.0065], with boot-p=.033. The two center values are not evidence that one calculation secretly changed the encoders.

They are different summaries of the same statistic: the value in the observed sample and the median over resampled datasets. Selecting maxima makes a composite a nonlinear statistic, so these summaries need not coincide. The paper treats the unit bootstrap as descriptive because units share sessions. Its session-based upper bound is +0.0063, which crosses zero. The strongest primary support therefore remains the session-based encoder-pair contrasts, particularly the left result in A16.

The methods define the two-sided bootstrap p as `2 × min(fraction of ao≤0, fraction of ao≥0)`, where ao is the resampled best-affect-minus-best-object composite. It summarizes how often that bootstrap statistic falls on each side of zero. These fractions count resamples, not independent sessions, so the formula does not remove the nesting limitation.

**Evidence type:** reported descriptive bootstrap and a secondary bound. The marginal bootstrap significance is not interchangeable with the primary session-level anchor.

**Reasoning or worked illustration:** If hypothetical bootstrap values are [−0.11,−0.09,−0.08], their median is −0.09 even if the original sample statistic was −0.083. Resampling changes which observations carry weight, and a maximum can select a different model in different samples. This toy illustration explains why distinct estimates are possible; it is not a reconstruction of the paper’s bootstrap distribution.

**Boundary/nonclaim:** boot-p=.033 does not license “the pooled composite is significantly negative under every valid inference scheme.” The source expressly says otherwise. Nor does choosing the best model in each class exhaust all possible affect or object representations.

**Self-check:** Which statement is accurate: “−0.083 and −0.086 use different model sets” or “they use different center estimators for the same model-set composite”?

**Answer:** The latter. The original-sample point and bootstrap median differ while the affect and object sets remain the same.

### A18. A margin-sized effect can be excluded without proving exact zero

**Source locator:** Results §2.3, PDF pp. 5–6, lines 178–196; Methods §4.5, pp. 14–15, lines 556–576.

**Passage role:** supplements significance tests with a comparison against a stated positive reference effect.

**Plain-language explanation:** A nonsignificant test against zero is not automatically evidence that an effect is negligibly small. The paper therefore compares the affect-object composite with a +0.100 margin based on the effect detected in medial-frontal cortex. The amygdala upper limits are −0.0065 in the unit bootstrap and +0.0063 in the session-based frame. Both lie well below +0.100. The methods compute the session limit as `mean + 1.96 × standard error` and the descriptive bootstrap limit as its 97.5th percentile.

These are the source’s stated bound conventions, not interchangeable choices of a new cutoff. This supports non-superiority to that reference-sized advantage under the specified calculations. It does not prove exact equality to zero. The comparison region also needs honest treatment: its boot-p=.040 is marginal and its session-based encoder-pair tests are p=.108 and .054. Calling it a uniformly strong positive control would overstate the source. The left-amygdala negative contrasts provide the clearer primary evidence, independent of treating the medial-frontal benchmark as definitive.

**Evidence type:** secondary margin comparison with a data-derived reference, plus a descriptive sensitivity calculation. This is not a prospectively fixed universal threshold for biologically meaningful emotion coding.

**Reasoning or worked illustration:** An interval from −0.05 to +0.0063 includes zero but excludes +0.100. That hypothetical interval supports “not as large as +0.100 under this analysis” while leaving open whether the true value is slightly positive, zero or negative. The source’s unit-level sensitivity figure uses a bootstrap standard error of 0.039 and reports about 72% power for the margin. Because the standard error is data-derived and the bootstrap ignores session nesting, that figure is not an independent sample-size justification or a primary test.

**Boundary/nonclaim:** “bounded null” should not be taught as proof of no function. The margin’s empirical origin and the different resampling frames remain part of its meaning.

**Self-check:** Can an interval that crosses zero nevertheless exclude an advantage at least as large as a specified positive margin?

**Answer:** Yes. It can lie entirely below the margin while spanning zero. That establishes a bounded comparison, not exact absence or a reliably negative effect.

### A19. Stronger encoders and the 136-choice specification curve

**Source locator:** Results §2.3, PDF p. 6, lines 197–222; Methods §4.11, p. 18, lines 712–736; Fig. S1 legend, p. 24.

**Passage role:** applies the paper’s baseline criticism to its own neuron result and tests sensitivity to several defensible analytic choices.

**Plain-language explanation:** A negative claim can inherit the weakness of the models used to challenge it. The study therefore adds DINOv2 and CLIP as generic encoders and a more recent facial-expression ViT as an affect alternative. In the reported single-neuron geometry, DINOv2 matches the amygdala about as well as ViT (+0.592), while the modern affect encoder is lower (+0.140). Four encoder panels keep a negative affect-best-minus-object-best contrast.

A separate specification curve varies encoder panel, maximum-versus-mean aggregation, RDM metric and neural normalization across pooled and left samples. The source reports 136 defensible specifications with negative contrasts throughout; Fig. S1 sorts them for display. This reduces concern that one arbitrary analytic choice produces the sign. It does not make 136 specifications into 136 independent datasets or cover every possible encoder and model of emotion.

**Evidence type:** reported robustness analysis on the existing data. The study classifies analytic choices as principled-equivalent or uncertain; the companion does not certify that these exhaust the scientific choice space.

**Reasoning or worked illustration:** Suppose three reasonable distance definitions all yield a negative comparison. That is more reassuring than selecting only the favorable metric. However, if every definition uses the same seven stimuli and recorded patients, all share those limitations. A specification curve examines analytic sensitivity; new stimuli and participants examine sampling generalization. The two checks complement each other rather than substitute.

**Boundary/nonclaim:** the newer facial model is not proved to be the strongest possible affect probe. The source explicitly says its load-bearing point is the generic-side comparison and cross-encoder robustness, not a universal ranking of emotion models. The right hemisphere is reported separately; its null is not hidden among the pooled/left specifications.

**Self-check:** Why is “all 136 specifications agree” weaker than “the conclusion holds for every possible analysis and stimulus”?

**Answer:** The tested choices form a finite, specified set using the same data and restricted stimulus space. Agreement within that set does not quantify over untested models, tasks or samples.

## Audio and the descriptive positive account

### A20. Building audio features at the brain’s time resolution

**Source locator:** Methods §4.7, PDF pp. 15–16, lines 592–617; Results §2.4, pp. 6–7.

**Passage role:** extends the stimulus-computable comparison from images to the movie soundtrack with explicit acoustic controls.

**Plain-language explanation:** The audio-affect encoder is a speech-emotion-tuned wav2vec2 network. It processes the soundtrack at 16 kHz in 20-second chunks. Frame-level hidden states are pooled into one feature value per second, then put through the same broad alignment, hemodynamic convolution and PLS prediction route as the visual features. The acoustic comparison includes RMS level, spectral summaries, zero-crossing rate and eight log-mel energies.

These features describe basic sound properties that could covary with emotion labels. The model therefore does not compare “emotion” with silence; it asks what the learned audio representation adds beyond specified acoustic or visual features. Positive-control regions include auditory cortex and superior temporal gyrus. The exact soundtrack is not redistributed, so a reader cannot run the complete extraction merely by downloading this code repository.

The acoustic filterbank uses the HTK conversion `mel(f)=2595 × log10(1+f/700)`, where f is frequency in hertz. It changes the frequency coordinate used to space eight log-mel energy features between the stated frequency limits. This is an acoustic representation choice, not an emotion score or a neural result.

**Evidence type:** reported audio feature extraction and model specification. The term speech-emotion-tuned describes training; it does not automatically make every hidden feature specific to felt emotion in a film.

**Reasoning or worked illustration:** Loudness can vary with dramatic dialogue, music and noise. If an audio-affect model predicts a response, a basic loudness control asks whether that simple property accounts for the gain. Spectral features broaden that comparison. A model might still encode language, speaker identity or other uncontrolled properties, so an incremental prediction remains a model-set comparison. This reasoning explains the role of the acoustic baseline without inventing an extra result.

**Boundary/nonclaim:** the audio feature stream is downsampled and summarized; it is not a direct recording of participants’ emotional experiences. The prosody-trained model may be imperfect for music and non-speech sound. It is also the only affect encoder in this arm.

**Self-check:** Why include a low-level acoustic baseline when the research question concerns emotion-related sound?

**Answer:** To test whether basic properties such as energy or spectral structure can account for the prediction, rather than attributing every audio-model gain to emotion by its training label alone.

### A21. A useful audio control is not a full two-encoder replication

**Source locator:** Results §2.4, PDF pp. 6–7, lines 223–236; Methods §4.7, lines 606–617; Fig. 1 legend.

**Passage role:** reports the audio outcome while explicitly limiting the criterion it can address.

**Plain-language explanation:** Audio-affect features add predictive information over vision in superior temporal gyrus: +0.0189, p=.001. The primary auditory-cortex increment is +0.0040, p=.12, so not every auditory comparison is a significant positive. In the amygdala the reported increments are −0.0028 left and −0.0023 right, both p<.001 in the negative direction. The corrected visual baseline sharpens the STG result compared with the earlier run, while the audio features themselves are unaffected by the EmoNet image-input correction.

These comparisons support a modality-appropriate control and do not support a special audio-affect increment in the tested amygdala model. Nevertheless, this arm uses only one affect encoder. It cannot directly evaluate agreement between independent affect encoders, criterion B. Describing it as a fourth complete application of the two-part test would therefore be inaccurate.

**Evidence type:** reported single-encoder cross-modal control. It adds a distinct stimulus modality but uses the movie participants rather than an entirely independent new cohort.

**Reasoning or worked illustration:** A two-instrument agreement claim requires two instruments. One instrument can outperform a baseline or fail to do so; neither outcome directly measures agreement with an unobserved second instrument. This logical limit remains even if the one-model p value is very small. A future audio test would need another appropriate affect representation and corresponding comparisons, not merely more significant results from the first model.

**Boundary/nonclaim:** negative increments do not prove the amygdala cannot process emotionally relevant sound. They concern incremental prediction by this model over these controls in the analyzed soundtrack and recording regime. The manuscript’s broad prose about being “no more auditory than visual” should be read within that tested scope.

**Self-check:** What would be missing from a claim that this arm passed or failed criterion B?

**Answer:** A second independently trained, suitable audio-affect encoder and a defined agreement comparison. With one affect encoder, B is not tested.

### A22. The behavioral regression asks a different question from encoder RSA

**Source locator:** Results §2.5, PDF p. 7, lines 237–257; Methods §4.6, p. 15, lines 577–591.

**Passage role:** looks for a limited positive description of neuronal responses instead of ending with an encoder comparison alone.

**Plain-language explanation:** The analysis relates trial responses to six recorded or stimulus-derived predictors: valence, intensity, choice, reaction time, confidence and correctness. Valence is centered along the fear–happy scale; intensity is distance from the middle. These are separate mathematical descriptions even though the stimulus continuum constrains how independently they vary. Baseline-corrected firing is regressed on standardized predictors, retaining trials with usable values and units meeting the minimum of 40 valid trials. The fit uses an intercept plus six predictors, with residual degrees of freedom n−7; units with fewer than five residual degrees of freedom are dropped.

The stated unit threshold is absolute t>1.96. A unit is called tuned at the stated absolute-t threshold, and the fraction tuned is compared with a nominal 5% rate. The paper also tests the population mean of the unit-level t values. Thus “population bias” and “fraction of individually tuned neurons” are different summaries. The regression uses behavioral fields already recorded with the task. It does not infer participants’ full emotional experience from the network labels.

The unit statistic is `t=β/SE`: the fitted predictor coefficient divided by its estimated standard error. The methods obtain coefficient uncertainty from the residual-variance estimate σ² times `pinv(X′X)`, where X is the design matrix and pinv is a generalized inverse. Standard errors use the covariance diagonal’s square roots. These quantities govern the stated tuning threshold.

**Evidence type:** reported exploratory characterization. It is distinct from the primary affect-versus-object RSA and does not retrospectively make that RSA a causal test of salience.

**Reasoning or worked illustration:** On a seven-level continuum, the extremes can both produce stronger responses than the middle. A signed linear predictor can miss that U-shaped pattern because one extreme is negative and the other positive. An absolute-distance intensity predictor gives both extremes high values. This illustrates why valence and intensity coefficients can differ. It does not imply that all real emotional intensity has this one-dimensional form.

**Boundary/nonclaim:** correlated predictors, threshold choices and nesting limit interpretation. A statistically detected coefficient is not proof that a neuron exclusively represents one psychological variable. The source does not interpret the single-trial predictive comparison because its cross-validated performance is at the noise floor.

**Self-check:** Can a population show a small systematic mean-t shift while few individual units pass a tuning threshold?

**Answer:** Yes. A weak, broadly distributed tendency can shift a population summary without producing a large fraction of individually strong effects.

### A23. A modest intensity bias is not strong single-neuron decoding

**Source locator:** Results §2.5, PDF p. 7, lines 239–257; Fig. 4 and legend, pp. 23–24, lines 899–903; Limitations, pp. 18–20.

**Passage role:** summarizes the descriptive positive findings and restrains their mechanistic interpretation.

**Plain-language explanation:** The paper reports a mean-t intensity bias of about +0.20 across the amygdala population, with 75% of tuned units positive. The mean-t bias is bilateral and numerically larger on the right, but the fraction of individually intensity-tuned units is modest. Pooled, 6.3% does not exceed the chance reference by its reported test; the left fraction is 7.9%, while the right is 3.6%. The paper also describes a left confidence/uncertainty signal and right valence and choice signals.

Medial-frontal responses show a different reaction-time pattern. These observations are compatible with a relevance/salience and decision-uncertainty account. They are not a completed model of how the amygdala works, and the exploratory families require the source’s multiple-comparison qualifications. A good teaching account should not replace an overly strong “emotion code” story with an equally strong “only salience” story.

**Evidence type:** reported descriptive and exploratory neural results with modest effects. The relevance-detector interpretation is an interpretation of these patterns, not an intervention identifying a unique computation.

**Reasoning or worked illustration:** A group can have an average height difference even if most pairs of individuals overlap. Likewise, a population bias need not make single-unit decoding reliable. The analogy concerns aggregate versus individual evidence, not a quantitative model of neurons. The source’s single-trial cross-validated prediction remains at the noise floor, so its overfitting-dominated model comparison is not used to claim a practical decoder.

**Boundary/nonclaim:** the 75% figure is among tuned units, not 75% of all 442 units demonstrating strong intensity selectivity. Nor does a mean-t coefficient directly measure bits of emotional information or a behaviorally useful decoding accuracy.

**Self-check:** Rewrite “the neurons strongly decode intensity” using the source’s actual level of support.

**Answer:** The sample shows a modest population-level intensity bias; the pooled individually tuned fraction does not exceed the chance reference, and useful single-trial prediction was not established.

### A24. The discussion combines different strengths without making them equal

**Source locator:** Discussion, PDF pp. 7–9, lines 259–345; Author note on scope, p. 21, lines 847–851.

**Passage role:** integrates the arms, states the reusable methodological recommendation and identifies the load-bearing evidence.

**Plain-language explanation:** The source’s synthesis is not “every experiment found nothing.” The movie has a small corrected positive A result but unestablished B; the face-morph fMRI arm lacks a reliable encoder match; the neuron arm provides the strongest left/session evidence for generic encoders outperforming affect encoders; the audio arm is a one-encoder control. The discussion emphasizes this uneven architecture rather than requiring each arm to carry the same conclusion. It recommends matched non-construct comparisons and convergence across independent construct encoders with appropriate positive controls.

It also distinguishes its contribution from cortical-localization work: the authors present the neuron anchor, multimodal extension and test protocol as the additional work. The positive salience/uncertainty account remains modest and exploratory. A paper can make a useful methodological argument while containing a mixture of strong, weak and uninformative empirical pieces.

**Evidence type:** synthesis and recommendation based on the reported analyses. It is not a formal meta-analysis combining four independent effects into one universal estimate.

**Reasoning or worked illustration:** In a hypothetical investigation, one assay precisely rejects a large effect, another is too noisy, and a third finds a small model-dependent effect. Averaging the three verbal labels into “three nulls” would erase information. Instead, identify which test supports the narrow conclusion and ask whether the others contradict it, corroborate it or simply cannot decide. The source adopts that more qualified structure after the preprocessing correction.

**Boundary/nonclaim:** the recommendation is a guard against overinterpretation, not a sufficient checklist for causal neuroscience. The paper does not establish that every construct needs exactly these encoders or that no other evidence can support a coding claim.

**Self-check:** Why is “all four arms independently refute emotion coding” an unsuitable summary?

**Answer:** They differ in stimulus independence, methods and informative power; one has a positive A effect, one an uninformative RSA null and one no second affect encoder. The strongest comparative evidence is specifically left/session-based.

## Statistical choices, robustness and the limits ledger

### A25. The primary estimator differs from the preregistered route

**Source locator:** Results §2.1, PDF p. 3, lines 75–81; Methods §§4.3 and 4.10, pp. 12 and 17–18; Limitations, p. 19, lines 761–771.

**Passage role:** discloses the executed model choice and explains the authors’ reason for preferring it.

**Plain-language explanation:** The frozen design specified banded ridge as the production encoding path. Banded ridge fits coefficients with a separate shrinkage penalty for each feature block, choosing penalties by an inner validation procedure. The executed primary analysis instead uses PLS with a fixed ten-component budget. PLS builds directions linking features to responses, limiting the model through a low-dimensional representation.

The authors report that ridge strongly shrinks effects near the single-participant noise floor: the fusiform EmoNet-minus-ResNet increment becomes about +0.0001 instead of +0.0105, while the amygdala increment becomes effectively zero. Ridge therefore produces a similar broad interpretation but very different effect magnitudes. Reporting PLS as primary may be defensible, but it is a change from the preregistered route and must remain visible. A teaching account cannot call the entire executed analysis unchanged or entirely confirmatory merely because a frozen specification exists.

**Evidence type:** reported analytic deviation and robustness result. The author’s explanation for choosing an estimator is an explanation, not a separate preregistered prediction of that estimator’s result.

**Reasoning or worked illustration:** A model with severe shrinkage can predict nearly constant responses, making every incremental effect tiny. The example is conceptual: it shows why zero-like estimates can arise from regularization rather than biological absence. Conversely, a less-shrunk model can overfit. Held-out scoring and controls help compare the alternatives, but they do not remove the need to disclose that the primary route changed after observing behavior.

**Boundary/nonclaim:** “same qualitative verdict” does not mean identical estimates, statistical status or pipeline. The neuron RSA does not use this encoding estimator, so the PLS-versus-ridge choice does not directly generate its left-neuron anchor.

**Self-check:** What two facts should a reader retain together about the estimator choice?

**Answer:** PLS is the executed primary route because the authors report ridge over-shrinkage, and this differs from the preregistered production specification. Both facts matter for interpretation.

### A26. Multiple comparisons and the distinction between raw p and corrected q

**Source locator:** Methods §4.9, PDF p. 17, lines 659–681; Results §§2.2–2.3.

**Passage role:** identifies which tests are interpreted directly, which are corrected and which marginal findings remain exploratory.

**Plain-language explanation:** Trying several hypotheses creates more opportunities for small p values by chance. A correction’s meaning depends on its family: which tests are counted together. The paper partitions confirmatory and exploratory contrasts. It explicitly states that the confirmatory amygdala affect-object contrasts are interpreted without a collective BY correction. Thus the pooled bootstrap p=.033 and medial-frontal p=.040 remain uncorrected marginals.

The strongest left/session results are emphasized as more robust. The positive-control detection family includes movie fusiform, face fusiform and audio STG, with Benjamini–Yekutieli false-discovery-rate adjustment. BY includes a harmonic multiplier to accommodate dependence. Separately, the two univariate face-morph fear/ambiguity p values are corrected within their own small exploratory family, giving q≈.063 for both. Neither clears q<.05. A reader needs this ledger rather than one phrase claiming “all results corrected.”

**Evidence type:** reported testing policy and correction calculations. Family definitions are analytic choices, not properties fixed by the mere existence of multiple p values in a paper.

**Reasoning or worked illustration:** If two raw p values are .028 and .042, applying the source’s m=2 BY rule uses harmonic factor 1+1/2=1.5 and monotonic adjustment. The resulting adjusted values are about .063 for both. This illustrates the reported computation, not a new analysis. It shows why a raw p below .05 can fail the declared corrected criterion. It also shows why the correct label is q or adjusted p rather than silently keeping the raw threshold.

**Boundary/nonclaim:** a corrected nonsignificant result is not proof of zero; a tiny raw p is not a universal guarantee against all analytic flexibility. The strongest claim must still be bounded to the relevant data and comparison.

**Self-check:** Is “p=.042 is below .05” sufficient to say the univariate fear effect survives the paper’s stated correction?

**Answer:** No. Its adjusted value is about .063, which does not meet the stated q<.05 threshold.

### A27. Reproducibility has several levels, not one badge

**Source locator:** Methods §4.10, PDF pp. 17–18, lines 682–711; Limitations, p. 19, lines 772–780; current repository README and `real_data/REPRODUCE.md`.

**Passage role:** states software, result provenance and the boundary between archived claims and a new reconstruction.

**Plain-language explanation:** The paper lists a compatible real-data software stack and supplies analysis code and logs. It reports that original feature caches were lost and regenerated with a reference extractor, with qualitative verdicts reproduced. The repository maintained alongside this companion adds an important implementation boundary: full-data agreement of its reconstructed movie extractor with archived results remains unverified, and the extra movie DINOv2/CLIP feature caches lack a preserved extraction script. These statements need not be hidden or collapsed.

One is the preprint’s report about regeneration; the other is the current repository’s verified execution contract. Static log checks can show that manuscript numbers trace to committed outputs. Synthetic tests can check formulas and input shapes. Neither alone establishes a fresh end-to-end reproduction from film and brain data. A learner should ask which layer of the chain has actually been run.

**Evidence type:** author-reported reproduction plus current software limitations.

**Reasoning or worked illustration:** Consider a hypothetical repository containing a correct output log and a tested function that expects 4,096 features. That can support traceability and a shape contract. It cannot establish that the correct frame preprocessing produced those features in a new environment. The input transformation, caches, estimator and output comparisons are distinct links. A failure at one does not automatically falsify the paper, but an untested link cannot be described as validated.

**Boundary/nonclaim:** qualitative agreement is not bit-exact agreement, and the source explicitly anticipates BLAS/GPU differences. Conversely, “some differences are expected” cannot excuse arbitrary changes in conclusions. Data access and reconstruction requirements remain real.

**Self-check:** What should a learner say after synthetic tests and number-to-log checks pass, but full-data extraction has not been run?

**Answer:** The tested contracts and traceability pass; full-data reproduction remains unverified. Do not convert those narrower checks into an end-to-end certificate.

### A28. The movie limitations weaken a particular interpretation

**Source locator:** Limitations, PDF pp. 19–20, lines 745–789; Results §2.1; Fig. 3.

**Passage role:** collects the reasons the movie arm is supportive but not load-bearing.

**Plain-language explanation:** Four limitations belong beside the movie result. First, correcting preprocessing creates a genuine small positive A result; the old clean null cannot support the current argument. Second, negative convergence point estimates are unreliable because their intervals include zero. Third, PLS and banded ridge produce markedly different effect magnitudes near the noise floor. Fourth, criterion B compares a scene-affect encoder with a face-affect encoder.

A mismatch in stimulus appropriateness can contribute to failure of convergence. Together these points do not show that the movie result is useless. They tell us precisely what it does establish: an encoder-specific small advantage and an absence of demonstrated stable cross-encoder agreement under the tested models. The left-neuron comparisons carry more of the paper’s claim. This is an example of a useful limitation changing the weight assigned to evidence rather than merely adding a ceremonial disclaimer.

**Evidence type:** source limitations interpreting reported analyses. The companion preserves these qualifications rather than treating them as problems already solved by the source’s controls.

**Reasoning or worked illustration:** Suppose two measuring devices target “temperature,” but one works best in air and one on contact surfaces. Disagreement in one environment could reflect suitability rather than absence of temperature. That hypothetical analogy explains why an independent scene-affect partner would be informative. It does not mean the current finding must be wrong or that every comparison needs identical models; the point is to distinguish independence from construct appropriateness.

**Boundary/nonclaim:** neither a small effect nor a model mismatch proves an invariant emotion code. The correct response is to specify a stronger comparison and retain uncertainty, not replace a negative headline with an unsupported positive one.

**Self-check:** What targeted follow-up most directly addresses the movie criterion-B partner limitation?

**Answer:** Add an independently trained scene-affect encoder suited to naturalistic films, with the same targets, controls and convergence analysis. That would still not by itself establish a causal neural mechanism.

### A29. Stimulus-to-neural prediction is not every possible emotion code

**Source locator:** Results §2.3, PDF p. 6, lines 217–222; Limitations, p. 20, lines 797–818.

**Passage role:** identifies alternative coding questions that the negative result does not resolve.

**Plain-language explanation:** The paper tests a mapping from stimulus-derived features to neural responses. It does not exhaust population representations of internally generated emotional states, task context or bodily signals. A cited mouse study examines emotional-state population geometry, which is a different question from matching the geometry generated by human visual stimuli through frozen image encoders. The face-morph task also varies mainly along one fear–happy dimension.

It can reveal orderly tuning but provides limited leverage for separating a rich emotional construct from low-level image differences. Pooling neurons across patients adds another scope constraint. Interoception (the sensing of internal bodily state) is not tested and is not naturally represented by the image encoders in this analysis. The source points to richer movie annotations and more diverse single-trial stimuli as future opportunities. These are open alternatives, not findings that should be filled in by an educational story.

**Evidence type:** explicit scope limits and future research directions. Mentioning an alternative is not evidence that the alternative mechanism has been demonstrated in this human dataset.

**Reasoning or worked illustration:** A hypothetical region might respond differently to the same face depending on an instruction, memory or heartbeat state. An image-only encoder receives identical pixels in those cases. Failing to predict the context-dependent difference would not show that the region has no meaningful code; it would show that the chosen input omits relevant variables. This example makes the input-scope problem concrete without asserting a measured mechanism here.

**Boundary/nonclaim:** the negative result is not universal impossibility, nor does mouse evidence automatically transfer to the human left amygdala. The appropriate claim remains tied to the tested image-computable representations and recorded conditions.

**Self-check:** What new input would be needed before this study could directly test an interoceptive explanation?

**Answer:** Relevant bodily-state measurements and a design/model linking them to responses. Image features alone do not supply those missing observations.

### A30. Data independence, sample scope and realistic replication targets

**Source locator:** Introduction, p. 2; Limitations, PDF p. 20, lines 806–818; Methods §4.1.

**Passage role:** prevents the number of analyses from being mistaken for the number of independent replications.

**Plain-language explanation:** The study has three datasets, two stimulus modalities and four analyses. Movie visual encoding and soundtrack audio encoding use the same naturalistic viewing cohort. Face-morph fMRI and neurons use the same seven-level stimulus, measured in different recording regimes. Thus each arm adds a kind of evidence, but the combination does not equal four independently sampled stimulus worlds.

The neuron sample comes from epilepsy-patient recordings; the movie fMRI sample has 20 participants; the face fMRI analysis retains 19. These sample characteristics matter when generalizing to people and tasks. The source identifies a richer emotion-movie resource as a useful replication target, while noting that diverse emotional single-trial data with good amygdala coverage remain important. A replication can be valuable by changing one weakness at a time: new stimuli, new participants, a new affect encoder or a more appropriate task. It should declare which type of independence it adds.

**Evidence type:** sample and design scope, plus a proposed replication direction. The named future resource is not analyzed in this paper.

**Reasoning or worked illustration:** If two analysts process the same experiment with different software, that checks analytic robustness. If two scanners measure the same seven faces in different participants, that adds measurement and participant variation. If a new experiment uses unrelated emotional scenes, it adds stimulus generalization. These hypothetical distinctions explain why “replication” needs a qualifier rather than one undifferentiated count.

**Boundary/nonclaim:** sharing stimuli does not invalidate an arm; it changes what independence can be claimed. Nor does a larger sample automatically repair an inappropriate construct comparison or missing input modality.

**Self-check:** Name one independence dimension added by comparing fMRI with neurons and one not added when both use the same morph continuum.

**Answer:** Recording resolution and participant/measurement context differ. The stimulus family and its one-dimensional structure do not become independent merely because the recording method changes.

## Reading the figures and the publication record

### A31. Figure 1 does not use a common biological effect scale

**Source locator:** Fig. 1 and legend, PDF pp. 21–22, lines 855–871.

**Passage role:** gives a compact visual overview while declaring important plotting approximations.

**Plain-language explanation:** Figure 1 places four comparisons side by side. Some bars show a change in held-out R² from encoding models; others show a change in Spearman RSA correlation. Those are different units and cannot be read as a common quantity of “emotion in the brain.” Each arm’s amygdala comparison is contrasted with its own regional control.

The legend states that the error bars are approximate standard errors reconstructed from reported p values, not directly measured SEMs, and that significance stars use the actual reported p values. The figure is therefore a summary illustration of the result architecture rather than a substitute for the underlying uncertainty calculations. It includes the corrected tiny positive movie increment, the uninformative face-morph null, the negative neuron comparison and the negative audio increment. The neuron comparison’s medial-frontal benchmark is not simply another auditory or visual cortex point.

**Evidence type:** visualization of reported results with explicit approximations. A plotted error bar does not supply new uncertainty evidence beyond the calculation used to construct it.

**Reasoning or worked illustration:** Comparing 0.01 R² with 0.10 RSA correlation is like comparing centimeters and kilograms because both are decimal numbers. Each score has its own interpretation and range. Even when two quantities share a numerical range, they need not measure the same property. Within-arm comparisons and the underlying methods are the meaningful reading route. A figure title’s compact phrasing should not override qualifications in the legend.

**Boundary/nonclaim:** the figure does not demonstrate that one region contains ten times as much emotional information as another. It does not make the face null informative or change the primary inference unit of the neuron arm.

**Self-check:** Before interpreting the relative bar heights across two panels, what should you inspect first?

**Answer:** Their units, estimator definitions and error-bar construction. Different panels summarize different metrics and cannot automatically be compared on one biological scale.

### A32. Figure 2 separates the observed center, bootstrap interval and margin

**Source locator:** Fig. 2 and legend, PDF p. 22, lines 873–884; Methods §4.5.

**Passage role:** visualizes the bounded comparison without making the descriptive unit bootstrap the sole primary result.

**Plain-language explanation:** The orange center at −0.083 is the observed pooled composite point estimate. The plotted interval [−0.156,−0.0065] belongs to the unit-bootstrap calculation. The bootstrap median −0.086 is a related but distinct summary. The blue comparison marker at +0.100 sets the stated non-superiority margin.

A separate annotation identifies the session-based upper limit +0.0063. These visual elements answer different questions. The left-neuron session contrasts provide the strongest primary evidence; the graph’s conspicuous pooled interval should not quietly replace them. The caption also notes that the pooled composite is significantly negative only in the descriptive bootstrap frame. This is a case where a learner should read the caption and method before deciding what an apparently clean confidence interval proves.

**Evidence type:** display of reported estimates and bounds with differing inferential roles. The +0.100 margin is tied to a marginal observed benchmark, not a universal neuroscience threshold.

**Reasoning or worked illustration:** Draw a number line with zero, +0.0063 and +0.100. The second point lies to the right of zero but far left of the margin. That geometry makes two claims logically compatible: the primary interval can include zero, and its upper end can exclude a margin-sized positive advantage. The plotted bootstrap interval lies left of zero, but it uses a different sampling frame. A learner must not merge the two into one stronger interval.

**Boundary/nonclaim:** the picture does not establish exact equality, absolute absence or a strong medial-frontal positive in every inference frame. The approximate 72% sensitivity figure is descriptive and data-derived, not fresh experimental power evidence.

**Self-check:** Which part of the plot could mislead a reader who ignores session nesting?

**Answer:** The unit-bootstrap interval wholly left of zero could be mistaken for the primary pooled inference. The source explicitly distinguishes it from the session-based frame and emphasizes the left encoder-pair contrasts.

### A33. Figures 3, 4 and S1 answer three different robustness questions

**Source locator:** Figs. 3–4 and their legends, PDF pp. 23–24, lines 886–909; Methods §§4.6 and 4.11. The repository contains the Fig. S1 specification-curve image; the exact source PDF supplies its legend.

**Passage role:** separates encoder agreement, descriptive tuning and analytic-choice stability.

**Plain-language explanation:** Figure 3 asks whether two affect encoders’ unique contributions covary across participants. Fusiform shows reliable positive agreement, while amygdala intervals cross zero. The source reports raw agreement because correcting for measurement reliability becomes unstable: low or modest reliabilities can yield a disattenuated correlation outside the permissible range or make it undefined. Figure 4 instead shows mean-t coefficients for behavioral/stimulus predictors.

Its intensity bar is a population bias, not a percentage of strongly tuned neurons or a decoding accuracy. Figure S1 sorts 136 affect-minus-object contrasts across analytic choices for pooled and left samples. Sorting makes the sign pattern visible but removes any natural sequence of the choices; the figure is not a time series or a set of independent experiments. Reading each plot as its own question prevents visually similar bars from becoming one undifferentiated “null result.”

**Evidence type:** reported descriptive visualizations and sensitivity analysis. The S1 image inspected with the repository is an additional display; the claim’s exact-version anchor remains the source text and legend.

**Reasoning or worked illustration:** A reliability correction divides an observed relation by a function of estimated reliabilities. If a denominator is near zero, small estimation errors can produce a huge or undefined answer. That mathematical sensitivity explains why an apparently sophisticated correction can be less interpretable than the raw estimate and interval. This reasoning does not recalculate the paper’s reliability estimates.

**Boundary/nonclaim:** raw convergence is not a measurement-error-free construct estimate. Population mean tuning is not widespread strong single-unit selectivity. Specification stability is not independent population replication. Each figure retains the limitations of the estimator and sample that produced it.

**Self-check:** Which figure would you consult for encoder agreement, which for intensity bias and which for analytic-choice sign stability?

**Answer:** Figure 3, Figure 4 and Figure S1, respectively. None substitutes for the other’s question.

### A34. Ethics, AI assistance, availability and references are part of interpretation

**Source locator:** Ethics and Data & Code Availability, PDF pp. 20–21, lines 824–837; Use of AI and Author note, p. 21, lines 842–851; References, pp. 24–25.

**Passage role:** declares input access, secondary-analysis context, assistance and responsibility, and identifies the cited record.

**Plain-language explanation:** The analysis uses public de-identified derivatives and processed firing data. Raw intracranial recordings are request-only for clinical-privacy reasons; the copyrighted movie is not redistributed. Open code therefore does not mean every input is freely bundled or that a reader may redistribute the film. The manuscript reports substantial AI assistance in analysis and drafting under the author’s direction, and assigns responsibility to the author.

The source’s disclosure does not establish independent human validation. The source’s ethics statement describes the authors’ secondary-analysis position and the original studies’ approvals; this guide does not perform a fresh legal or institutional determination. Funding and competing-interest declarations are publication metadata, not experimental evidence for or against a neural result. The reference list supplies the paper’s scholarly context; the guide teaches the uses made of those references without claiming to have rerun or independently validated every cited study.

**Evidence type:** author disclosures, access statements and bibliographic metadata. These are distinct from measured neural effects, and their truth is not established merely by putting them in an educational narrative.

**Reasoning or worked illustration:** A hypothetical student can inspect a source log and rerun a synthetic function without possessing the original stimulus. That is useful access, but it is not the same as a full-data rerun. A responsible reproduction plan records the missing or restricted link rather than obtaining a substitute film and silently assuming equivalence. Similarly, a model-generated answer key is a teaching artifact, not evidence that actual learners understood the study.

**Boundary/nonclaim:** an explanation of the declarations does not establish an ethics approval, dataset permission or scientific validity. Consult the original source and data terms.

**Self-check:** What should “openly available code” mean to a reader planning reproduction here?

**Answer:** The code and logs can be inspected under their terms; separate public or restricted inputs, weights, stimulus access and reconstruction limitations must still be addressed explicitly.

## Implementation reading ledger

These source parameters make the methods traceable. They are definitions/settings, not new empirical findings. Units A04–A07, A11–A15, A20–A22 and A25–A27 explain why they matter.

| Source method | Declared setting or operation | Interpretation boundary |
|---|---|---|
| §4.1 movie | Twenty participants; approximately 5,703–5,704 BOLD rows; released no_blur_no_censor MNI derivatives, no new registration | The exact dataset and derivative matter, not just “movie fMRI.” |
| §4.1 faces | Nineteen retained designs; seven expression levels; motion confounds; refitted SPM-HRF first-level model at TR 2 s, cosine drift/high-pass 1/128 Hz | Beta patterns depend on a defined first-level model; coverage is not construct validity. |
| §4.2 frame features | Six frames/second, 256 px extraction; TR averages; scene EmoNet 227 px/raw RGB/4,096 features; ViT 768 and ResNet 2,048 features | Models have their own canonical transforms. Do not apply scene EmoNet’s raw scale to every encoder. |
| §4.2 other features | Face encoder 256 features at 256 px and /255; movie low-level 10 features; face low-level 4 features | “Low-level” is not an identical feature set across arms. |
| §4.2 alignment | Target feature index 1.042 × TR index, linear interpolation, clipped at cache end | Source-clock correction, not a universal film constant. |
| §4.3 preparation | HRF followed by column standardization, at most 300 PCA directions, standardization again; PLS at most 10 components; five contiguous folds | The source gives the precise preparation order. |
| §4.3 score | Mean voxelwise held-out R², near-constant target sums of squares set to NaN | Negative held-out scores are permitted; missing variation is not zero fit. |
| §§4.3–4.4 ROIs | Harvard–Oxford 25% masks for main ROIs; stricter 50% amygdala sensitivity; nearest-neighbor mask resampling | Atlas thresholds and spatial controls define the measured region. |
| §4.4 RSA | Seven ordered levels FE = [0,30,40,50,60,70,100]; distance 1 − Pearson; Spearman on 21 upper-triangle entries | Distances are dependent, not 21 participants. |
| §4.5 neuron | Baseline-corrected fear–happy responses; drop missing-level vectors; per-unit tuning shape standardized | Pooled geometry is not simultaneous population dynamics. |
| §4.5 uncertainty | Unit resampling 2,000 RSA/4,000 equivalence iterations; within-session minimum 3 units; observed max-affect − max-object | Unit bootstrap is descriptive, session contrasts primary. |
| §4.5 bound | Margin .100; mean + 1.96 SE session limit; 97.5th bootstrap percentile; descriptive sensitivity Φ(.100/.039 − 1.96) | Conventions and data-derived margin/SE must remain visible. |
| §4.6 regression | Intercept plus 6 z-scored predictors; minimum 40 valid trials; residual dof = n − 7; absolute t > 1.96 unit threshold | Thresholded unit fractions differ from population mean coefficients. |
| §4.7 audio | 16 kHz, 20 s chunks; 1 Hz pooled 1,024-feature vectors; 13 acoustic features; same PLS route | One audio-affect encoder cannot establish criterion B. |
| §4.8 controls | Quarter/half/three-quarter time shifts; face-presence/count/area; 8 TR gap; 20 random projections | Each targets a declared alternative, not every confound. |
| §4.9 multiplicity | Direct confirmatory amygdala interpretation; BY positive-control family; separate 2-test univariate family | Raw p and corrected q are not interchangeable. |
| §4.10 software | Real-data stack Python 3.9–3.12, numpy 1.26.4 (<2), torch 2.2.2, torchvision .17.2 and listed analysis libraries | A current synthetic CI environment is not a full-data rerun of this stack. |
| §4.11 robustness | Four encoder panels; 136 declared pooled/left specifications; stronger movie generic comparison | Finite analytic sensitivity does not quantify over all models or stimuli. |

## Glossary

- **Affect:** emotion-related properties, including the particular valence/arousal or scene categories used to train a model; not automatically one universal construct.
- **Baseline/control:** a comparison selected to test an alternative explanation. A generic encoder and a positive-control region play different roles.
- **BOLD/fMRI:** a blood-oxygen-dependent imaging signal and the method that records it; an indirect neural measurement.
- **Bootstrap:** resampling observed data to describe uncertainty under a stated resampling unit. Unit and session bootstraps are not interchangeable.
- **Capacity matching:** reducing differences in model flexibility; it does not make all training data or architectures identical.
- **Confidence interval:** a range constructed with a stated repeated-sampling interpretation under assumptions, not a personal probability that a theory is true.
- **Construct:** the theoretical property, such as emotion, whose interpretation is at issue.
- **Criterion A/B:** comparative advantage over a non-construct model, and convergence across independent construct models, respectively.
- **Encoder:** a mapping from stimulus to numerical features. Frozen weights do not imply an unfitted downstream encoding model.
- **Encoding/decoding:** predicting neural responses from features versus predicting stimulus/behavior labels from responses; success in either needs careful interpretation.
- **Hemodynamic response:** the delayed BOLD response modeled by convolving stimulus features with a response function.
- **Interoception:** sensing internal bodily conditions; not measured by the image encoders here.
- **PCA/PLS:** dimensional reduction based on input variation / a low-dimensional feature-response relationship.
- **Pseudo-population:** units pooled across recordings to form a representation; not necessarily a simultaneously recorded functional population.
- **p value/q value:** a null-model tail probability / an adjusted significance quantity under a declared multiple-testing rule. Neither is the probability that a biological explanation is true.
- **R²/ΔR²:** prediction score relative to a mean reference / a difference of scores on matched targets.
- **RDM/RSA:** a matrix of condition-pair dissimilarities / a comparison of neural and model dissimilarity patterns.
- **Region of interest:** a predefined brain region selected for analysis.
- **Ridge:** a penalized regression whose shrinkage depends on the chosen penalty.
- **Salience/intensity:** the source’s modest descriptive account of response tendencies, not a complete demonstrated mechanism.
- **Specification curve:** a display of results across declared analytic choices; not a set of independent datasets.

## Twelve answered transfer questions

Try these questions before reading the answers. They ask you to apply the source’s distinctions to other studies.

1. **A speech model predicts a language region. What comes before calling it a speech-specific code?** Compare suitable non-speech features at a comparable fitting budget, test an independent speech representation and a useful control, and identify what prediction leaves causally unresolved.
2. **A model’s held-out R² changes from −.020 to −.018. What can you say?** The matched score improves by .002. Neither absolute score beats the mean reference, and the difference does not name the psychological cause.
3. **Two models have equal average increments. Does B pass?** Not automatically. They might disagree across participants or conditions. Define and estimate convergence with uncertainty.
4. **An agreement estimate is −.28 with an interval containing zero. Is negative convergence established?** No. The sign is unresolved under that interval; absence of reliable positive convergence is the narrower description.
5. **A pooled unit bootstrap is significant but session-based uncertainty crosses zero. Which is primary in this paper?** Session-based inference. Report the bootstrap as descriptive and preserve the nesting explanation.
6. **An upper bound is .0063 against a margin .100. Does this prove zero?** No. It excludes the margin-sized advantage under the analysis while allowing values near zero, including small positive ones.
7. **A second affect model is poorly suited to the stimulus domain. What follows?** Failure to agree is harder to attribute solely to absence of the construct. Test a suitable independent partner without assuming the present result must be false.
8. **An analysis changes from preregistered ridge to PLS. Can it be described as unchanged confirmatory execution?** No. State the deviation, motivation, both results and which inference does not depend on that estimator.
9. **A face fMRI ROI responds on average but no encoder matches its geometry. What is missing?** An informative representational comparison. Mean activation does not guarantee reliable pattern geometry.
10. **A specification curve has 136 negative entries. How many independent participant samples does it add?** None. It examines analytic sensitivity within the existing data; a new sample is a different robustness dimension.
11. **A movie result and its soundtrack result use the same participants. Are they useless because they are not independent?** No. They test different feature/modality questions, but their shared sample must be retained in any replication claim.
12. **What evidence would improve a bodily-state account of the same region?** Collect relevant bodily/contextual inputs and define a new task/model with independent testing. The present image-computable result neither measures nor rules out that alternative.

## Exact source

Exact source: [Zenodo record 22314186, version 3](https://doi.org/10.5281/zenodo.22314186), `MANUSCRIPT_amygdala-false-floor_v2_linenumbered.pdf`, SHA-256 `d4ba2bb793b7e2068da17775ac45c3a39a756c90d3421a4c0dc936437f5c4ae2`. The [code archive](https://doi.org/10.5281/zenodo.21367398) is software, not the manuscript. Source locators use this exact PDF’s pages and printed lines. [COMPLETE_COVERAGE.md](COMPLETE_COVERAGE.md) maps the full source to teaching units and explicitly accounts for references and administrative matter.
