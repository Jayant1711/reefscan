# Viva Questions — ReefScan AI

*Questions your examiners are most likely to ask, grouped by topic. Model answers: `VIVA_ANSWERS.md`.*

## A. Problem & motivation
1. Why is underwater computer vision harder than ordinary vision?
2. What is coral bleaching, physically, and why can a camera detect it?
3. Why can't you just train a 17,357-class species classifier on FishNet directly?
4. What is the real-world user of your system and what decision do they make?

## B. Phase 1 — Enhancement
5. Why does CLAHE beat plain histogram equalization underwater?
6. Explain the Dark Channel Prior. Why does it need modification underwater?
7. What is UDCP and how does it differ from DCP?
8. What do UCIQE and UIQM measure? Why are they "no-reference"?
9. How exactly does your auto-selector choose the best strategy, and why those weights?
10. Why is PSNR/SSIM of *enhanced vs original* a weak metric here, and how do you compensate?

## C. Phase 2 — Classification
11. Why transfer learning instead of training from scratch?
12. Explain WeightedRandomSampler vs class-weighted loss — don't they do the same thing?
13. What is mixup and why does it help marine datasets?
14. How did you compute top-5 accuracy and why report it?
15. Which backbone won and why (capacity vs data size trade-off)?
16. What is label smoothing and what problem does it fix?

## D. Phase 3 — Coral / WGBS (your core innovation — expect deep probing)
17. The NOAA dataset is binary. How do you produce 5 severity classes and is that legitimate?
18. Walk through the WGBS formula. Why λ=0.6?
19. What physical evidence does the whiteness ratio capture?
20. When will WGBS fail? (white sand? dead algae-covered coral? low light?)
21. What is the ordinal penalty loss and why is severity "ordinal"?
22. How would you validate the 15/35/60/85 thresholds scientifically?

## E. Phase 4 — Explainability
23. How does GradCAM compute its weights?
24. What does GradCAM++ add mathematically over GradCAM?
25. How does ScoreCAM differ conceptually (gradients vs perturbation)?
26. Why is ScoreCAM slow on CPU and what did you do about it?
27. Can heatmaps be used as *proof* of correctness? (careful!)

## F. Phase 5 — Real-time
28. Why segmentation instead of a YOLO-style detector?
29. How does your centroid tracker keep unique IDs? What breaks it?
30. What is frame-skip and why doesn't it ruin fish counting?

## G. Phase 6 — Analytics
31. Derive or explain Shannon entropy, Gini-Simpson, Margalef — and why combine all three?
32. Justify the RHS weights (0.60/0.25/0.15).
33. How do you measure turbidity from a single frame?
34. What are the limitations of your trend slope with only a few observations?

## H. Engineering & evaluation
35. Why is your project reproducible? (seeds, configs, smoke tests)
36. How does the system degrade gracefully on a machine without GPU / without trained models?
37. What are the biggest scientific limitations of this project?
38. If given one more month, what would you add first and why?
