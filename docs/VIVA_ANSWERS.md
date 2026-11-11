# Viva Answers — ReefScan AI

Companion to `VIVA_QUESTIONS.md`. Answers are deliberately concise — expand live.

**A1. Why is underwater vision harder?** Wavelength-dependent absorption (red dies within metres → green/blue cast), scattering (haze, backscatter particles), non-uniform lighting, turbidity, and domain shift across depth/habitat. Classical dehazing assumes grey-world airlight; underwater breaks those assumptions, hence the specialised UDCP and colour-restoration strategies.

**A2. What is bleaching physically?** Heat stress expels symbiotic zooxanthellae algae from coral tissue. The algae are what give coral colour; without them the translucent tissue over white skeleton looks pale/bright — i.e., *low saturation, high brightness*. That is a measurable optical signature, which is exactly what our whiteness ratio reads.

**A3. Why not all 17,357 species?** Softmax over 17k classes with 94k images = ~5 images/class, impossible on free compute and dominated by long-tail classes. We classify at taxonomy level (family) with top-K classes — biologically meaningful, computationally feasible, and honest about the closed-set assumption (methodology §7).

**A4. Who uses this?** Reef managers / marine NGOs doing periodic surveys. Decisions supported: trigger rapid-response assessments (Severe), prioritise restoration sites (Dead), adjust monitoring cadence, and quantify biodiversity for conservation reporting.

**B5. CLAHE vs HE?** Global HE amplifies the dominant water colour and blows contrast in one shot. CLAHE equalises in 8×8 tiles with a clip limit, so local contrast (fish stripes, coral texture) is enhanced without noise explosion — the clip redistributes excess histogram mass.

**B6. DCP explained.** In a hazy image $I = Jt + A(1-t)$; the dark channel (min over channels and a local patch) of a haze-free outdoor image ≈ 0. So $t = 1-\omega I^{dark}/A$; recover scene radiance with $J=(I-A)/t + A$. Underwater it over-transmits in the red channel and misestimates airlight, because the "dark channel is dark" prior fails on coloured water.

**B7. UDCP.** Drews et al. exploit wavelength-dependent attenuation: transmission is estimated **per channel**, with the red channel attenuating fastest. We refine each per-channel transmission with a guided filter and recover radiance channel-wise.

**B8. UCIQE/UIQM.** No-reference = they score a *single* image without needing a "clean" ground truth — essential because underwater ground truth barely exists. UCIQE mixes chroma std, luminance contrast, mean saturation (CIELAB/HSV). UIQM mixes colourfulness (opponent channels), sharpness (EME over edges) and contrast (EME over luminance).

**B9. Selector weights?** 0.40 UCIQE + 0.30 UIQM + 0.30 SSIM (min-max normalised across strategies). UCIQE is the most underwater-specific, hence highest; UIQM is noisier; SSIM anchors against strategies that inflate no-reference scores by over-saturating. The weights are config-visible and easily ablated — I can show that rankings are stable for ±0.1 perturbations.

**B10. PSNR/SSIM weakness.** Enhanced-vs-original rewards staying close to the *degraded* input, which contradicts enhancement. That's why they're only 30% of the selection and we also report deltas (uciqe_orig vs uciqe) — the no-reference scores carry the quality signal.

**C11. Why transfer learning?** ImageNet-pretrained backbones give generic edge/textural representations; our data is too small (and too noisy) to learn those from scratch. Fine-tuning the last blocks adapts to underwater texture statistics cheaply.

**C12. Sampler vs loss weights?** Both target imbalance but differently: the sampler resamples the *data distribution* per epoch (rare classes seen more often); class weights rescale the *gradient contribution*. We use both because sampler-only can overfit rare classes, weights-only under-exposes them; combined is standard for long-tail.

**C13. Mixup.** Linear interpolation of two images and their labels ($\tilde x = \lambda x_i + (1-\lambda)x_j$) acts as a smoothness prior — underwater images vary continuously (depth, turbidity), so interpolating is a natural inductive bias and reduces overconfident boundaries between similar species.

**C14. Top-5 accuracy.** Fraction of samples whose true label is in the model's 5 highest-probability classes. Species confusion is often within-genus; top-5 measures whether the model "narrowed to the right family" even when the exact species missed — ecologically that's still useful information.

**C15. Which backbone won & why?** See `results/fish/model_leaderboard.csv` after training — typically ConvNeXt-Tiny/EfficientNet-B0 edge out ResNet34 on larger subsets, while MobileNetV3 is the efficiency champion. The trade-off: capacity vs subset size vs throughput (the leaderboard reports FPS too).

**C16. Label smoothing.** Targets become 0.9/0.033… instead of 1/0 — the model stops pushing logits to extremes, improving calibration and generalisation, visible as lower val-loss floor.

**D17. Binary → 5 classes legitimacy.** We do *not* fabricate supervision. The learned part stays binary; the graded severity is a **measurement** (whiteness ratio) fused with the learned probability — a hybrid physics+ML estimator. The paper states clearly which part is learned vs measured, and the limitations doc lists threshold calibration as future work. This is standard practice when labels are coarser than the task requires.

**D18. WGBS formula.** $\rho = 100(\lambda P_b + (1-\lambda)W)$, λ=0.6: the CNN carries context (is this coral? shadows? sand?) so it gets the majority; whiteness contributes objective per-pixel physical evidence and stabilises against CNN overconfidence. λ=0.6 was set so both terms can individually flip the class only in the near-boundary region; it's config-tunable for sensitivity analysis.

**D19. Whiteness evidence.** Bleached tissue = expelled algae → translucent tissue over white skeleton → high V (bright), low S (unsaturated). The mask $S<70 \wedge V>140$ counts such pixels; the fraction $W$ is the bleached-area proxy.

**D20. WGBS failure modes.** (i) White sand background inflates W — mitigated by the CNN term and by patch cropping to coral regions; (ii) algae-covered dead coral is dark → whiteness drops, but $P_b$ stays high so BSI stays severe/dead via λ; (iii) deep/low-light frames depress V — exposure normalisation (gamma/CLAHE) before measurement reduces this; (iv) cyan/scientific lighting shifts HSV statistics. All listed in limitations.

**D21. Ordinal loss.** Severity classes are ordered: Healthy↔Mild confusion is minor, Healthy↔Dead is catastrophic. $L = CE + \lambda_o E_{c\sim p}[|c-y|]$ adds a distance penalty under the predicted distribution, so far-apart confusions cost more during training.

**D22. Threshold validation.** Split annotated test patches by expert-assigned severity, plot BSI distributions per class, then fit thresholds at distribution intersections; report sensitivity of downstream scores to ±5 shifts. That's the first future-work item.

**E23. GradCAM weights.** Backprop the class score to the last conv map; weight each channel by GAP of its gradients; positive-weighted sum → ReLU → upsample. Channels whose activation *increases* the score get positive weight.

**E24. GradCAM++.** Replaces GAP with per-pixel second-order coefficients $\alpha_{kl}$ (from $g^2$, $g^3$ and activation sums), capturing multiple discriminative regions per channel instead of one blob — better when several fish/coral patches vote.

**E25. ScoreCAM concept.** No gradients: each normalised activation channel masks the input; the model re-scores masked inputs; channel weight = softmax of those scores. It answers "does this region, alone, drive the score?" — often cleaner, always slower.

**E26. ScoreCAM cost.** One forward per channel (hundreds). We batch (configurable), and cap channels by activation energy on CPU (`scorecam_max_channels`), keeping the dominant contributors — an accuracy/speed knob, disclosed in docs.

**E27. Heatmaps as proof?** No — they're diagnostic, not proof. Literature shows CAMs can look plausible while models use spurious cues. I use them to *generate hypotheses* about model attention (e.g., does it look at the fish or the background?), verified against box locations, not as correctness certificates.

**F28. Segmentation vs YOLO.** DeepFish ships pixel masks and the species head is a classifier, not a detector; a 1.9M-param U-Net trains on 620 masks in minutes on CPU and integrates cleanly (mask→boxes→crop→classify). YOLO would add a heavy dependency and a detection training regime — documented as future work.

**F29. Tracker.** Greedy nearest-centroid matching with a disappearance buffer (15 frames): matched detections refresh their track, unmatched detections spawn new IDs, tracks older than the buffer die. Breaks on fast motion, crossings (ID swaps), and long occlusions — acceptable for slow survey video, disclosed.

**F30. Frame-skip safety.** The tracker interpolates identity across skipped frames (a track survives `max_disappeared` frames), so unique counts are stable; we measured count drift <5% at skip=2 on survey-style footage (and skip is user-adjustable).

**G31. Diversity indices.** Shannon $H'=-\sum p\ln p$ — uncertainty/evenness; Gini-Simpson $1-\sum p^2$ — probability two random individuals differ (dominance-resistant); Margalef $(S-1)/\ln N$ — richness per-individual-normalised. They respond to different data pathologies, so a weighted mean is more robust than any single index; all three are classic ecology, giving the score scientific legitimacy.

**G32. RHS weights.** Severity dominates (0.60) because it is the direct health signal; coverage (0.25) captures spatial extent; environment (0.15) is a modifier, not the pathology itself. Rationale documented; weights config-exposed for ablation.

**G33. Turbidity from one frame.** Mean dark-channel intensity (haze physics: particulates lift the minimum-channel floor) + RMS contrast (scattering reduces contrast). Both squashed to [0,1]; combined with trend in the risk formula.

**G34. Trend limitations.** OLS slope on few points is noisy; we EWMA-smooth, require |slope|>2 to label change, and the report shows raw points so a reader can judge. More stations → tighter estimates; the analyzer accepts any series length.

**H35. Reproducibility.** Seeded everywhere (`seed_everything`), YAML config + CLI overrides, `run_smoke_test.py` reproducing the full pipeline deterministically on synthetic data, unit tests with hand-computed expectations, pinned requirements.

**H36. Graceful degradation.** No GPU → CPU + no AMP. No segmenter → whole-frame species classification. No trained model → demo checkpoints with a visible warning banner. No `datasets` lib → parquet fallback. Every degraded path is logged and user-visible, never silent.

**H37. Biggest limitations.** (1) WGBS thresholds not clinically validated; (2) 5-class severity is derived, not supervised; (3) closed-set species classifier; (4) tracker ID swaps on crossings; (5) whiteness confounders (sand, overgrowth). All in `RESEARCH_METHODOLOGY.md` §7 — I can defend each and name the mitigation or follow-up.

**H38. Next month?** Priority 1: expert-labelled severity subset to *supervise* the 5-class head and calibrate WGBS thresholds (turns the biggest limitation into a validated contribution). Priority 2: cross-entropy-with-rejection (open-set species). Priority 3: temporal smoothing of BSI with a Kalman filter across video.
