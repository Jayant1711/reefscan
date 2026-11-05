# Interview Questions — ReefScan AI (ML/CV engineering angle)

*How industry interviewers would probe this project — with answer pointers.*

## Computer vision fundamentals
1. Walk me through the full underwater pipeline you built. Where would it break first on real footage? *(backscatter-heavy frames → enhancement selection; see B-answers)*
2. CLAHE vs histogram equalization — when does each fail? *(tiled, clipped vs global; noise amplification)*
3. Explain the dark channel prior and two reasons it fails underwater. *(airlight estimate, red channel)*
4. What no-reference metrics exist for image quality and what are their blind spots? *(UCIQE/UIQM inflate under saturation hacks — hence SSIM anchor)*

## Deep learning
5. Why did you pick those four backbones? What axis do they span? *(accuracy/latency/params frontier)*
6. Your dataset is imbalanced. Walk me through every balancing mechanism in your trainer and why both sampler and loss weights. *(WeightedRandomSampler + inverse-√ weights)*
7. Explain mixup mathematically. When would you disable it? *(when labels are not linearly interpolable — e.g., counting)*
8. What does label smoothing do to calibration? How would you verify? *(ECE before/after)*
9. Why cosine LR with warmup instead of step decay?
10. Mixed precision: what actually happens on GPU, and why is it pointless on CPU?

## Evaluation & honesty
11. Your coral dataset is binary but you claim 5-class severity. Defend that to a skeptical reviewer. *(measurement+learning hybrid; limitations doc; validation plan)*
12. How would you detect overfitting in your history.csv and what knobs would you turn first? *(aug strength, mixup, early stop, backbone size)*
13. How do you report results you don't trust? *(flag demo vs trained models everywhere in UI/CLI)*
14. Design an ablation study for WGBS λ. *(grid λ∈{0.3..0.9}, measure class-boundary stability vs expert labels)*

## Production / MLOps
15. How does your dashboard avoid reloading models every request? *(st.cache_resource singleton)*
16. A video pipeline must process 30 FPS on CPU. What do you change? *(frame-skip, MobileNet, ONNX, 192px seg, worker pool)*
17. How would you version models and datasets? *(checkpoints already carry class_names/metrics; add DVC or MLflow)*
18. Your gdown download breaks weekly. How do you make data acquisition robust? *(checksums, mirrors, resume, manual-fallback docs — already implemented)*
19. How would you add streaming inference (RTSP) to your design? *(process_video already takes any cv2 source; add reconnect + ring buffer)*

## Software engineering
20. Explain your package layering and how you avoided circular imports. *(schema dataclasses; dependency direction doc)*
21. Why a registry pattern for enhancers? Show how you'd add a ninth strategy. *(subclass + decorator + import)*
22. What do your 25+ unit tests actually pin down? *(hand-computed metric values, contract tests per strategy, CAM shapes, tracker semantics)*
23. How is the config system designed and what prevents typos in keys? *(DotDict get_nested with defaults; tests pin key paths)*

## Research thinking
24. Propose the next paper from this codebase. *(calibrated ordinal severity with expert labels; open-set species recognition; cross-dataset generalisation of whiteness)*
25. ScoreCAM vs GradCAM — which would you ship to a marine biologist and why? *(GradCAM++ default, ScoreCAM optional: cost/latency vs interpretability)*
26. If biodiversity scores disagree with a human ecologist, how do you debug? *(component-wise breakdown, per-index values logged in SurveyAnalytics)*
