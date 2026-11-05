# Research Methodology — ReefScan AI

This document is the authoritative reference for every formula, threshold and design
decision in the system. Cite it as the methodology appendix of the paper.

---

## 1. Problem formulation

Given an underwater image/frame $I$, produce:
1. fish detections $\{(b_j, s_j, c_j)\}$ (box, species, confidence),
2. coral condition $(k, \rho)$ — ordinal class $k \in \{\text{Healthy},\text{Mild},\text{Moderate},\text{Severe},\text{Dead}\}$ and continuous severity $\rho \in [0,100]$,
3. explanation maps $M_m$ for method $m \in \{\text{GradCAM}, \text{GradCAM++}, \text{ScoreCAM}\}$,
4. survey-level scores: MBS, RHS, trend, environmental risk.

## 2. Phase 1 — Enhancement and its automatic selection

### 2.1 Strategies
| Name | Core idea | Reference |
|---|---|---|
| `clahe` | Contrast-limited AHE on LAB L-channel, tiles 8×8, clip 2.0 | Zuiderveld 1994 |
| `dark_channel` | DCP dehaze: $t(x)=1-\omega\, I^{dark}(x)/A$, guided-filter refinement, $\omega=0.95$, $t_0=0.1$ | He et al. 2010/2011 |
| `udcp` | Per-channel transmission using underwater attenuation physics | Drews et al. 2013 |
| `white_balance` | Shades-of-gray, Minkowski $p=6$, 2% percentile clip | Bianco et al. 2017 |
| `color_restoration` | Red-deficit compensation $R' = R + 1.15\max(\bar{GB}-R, 0)$ + per-channel stretch | Galdran et al. 2015 (inspired) |
| `gamma` | $\gamma = \ln(0.5)/\ln(\bar I)$, clamped to $[0.3, 2.5]$ | Huang et al. 2012 |
| `histogram_eq` | Global HE on L (baseline) | — |
| `fusion_wcg` | WB → CLAHE → γ composite | this work |

### 2.2 Metrics
- **PSNR** $= 10\log_{10}(255^2/\mathrm{MSE})$ — restoration proximity.
- **SSIM** — 11×11 Gaussian window, $\sigma=1.5$, $C_1=(0.01\cdot255)^2$, $C_2=(0.03\cdot255)^2$, averaged over channels.
- **UCIQE** $= c_1\sigma_c + c_2\,con_l + c_3\mu_s$, $(c_1,c_2,c_3)=(0.4680,0.2745,0.2576)$, with chroma spread $\sigma_c=\sqrt{\mathrm{Var}(a^*)+\mathrm{Var}(b^*)}$ in CIELAB, luminance contrast $con_l=\mathrm{std}(L^*)$, mean HSV saturation $\mu_s$ (Yang & Sowmya 2015).
- **UIQM** $= c_1\mathrm{UICM} + c_2\mathrm{UISM} + c_3\mathrm{UIConM}$, $(c_1,c_2,c_3)=(0.0282,0.2953,3.5753)$ (Panetta et al. 2016): opponent-colour statistics with 3% two-sided alpha-trimming (UICM), EME over the Sobel edge map of HSV-V (UISM), EME over $\log(1+V)$ (UIConM).

### 2.3 Automatic selection
For each strategy, metrics are averaged over a sample of $n=8$ images, min-max normalised
across strategies, and fused:

$$\text{score}(e) = 0.40\,\widehat{\text{UCIQE}}_e + 0.30\,\widehat{\text{UIQM}}_e + 0.30\,\widehat{\text{SSIM}}_e$$

The argmax strategy is persisted (`results/enhancement/enhancement_report.json`) and reused
at inference. Rationale: UCIQE/UIQM are no-reference underwater-specific quality proxies;
SSIM anchors the selection against over-processing that inflates no-reference scores.

## 3. Phase 3 — Whiteness-Guided Bleaching Severity (WGBS)  ★ core innovation

**Motivation.** Public coral-bleaching labels are binary (NOAA: CORAL vs CORAL_BL), yet reef
management needs *graded* severity. Human annotators grade bleaching by how *pale and
bright* the coral tissue appears. We operationalise exactly that physical evidence.

**Whiteness ratio.** In HSV space:

$$W(I) = \frac{1}{|I|}\,\big|\{\,p:\ S(p) < S_{\max}\ \wedge\ V(p) > V_{\min}\,\}\big|,\qquad S_{\max}=70,\ V_{\min}=140$$

Bleached tissue is pigment-free ⇒ low saturation, high brightness. Healthy tissue is
pigmented ⇒ high saturation. Dead coral is typically overgrown/dark and is separated by the
CNN component below (documented limitation: pure whiteness cannot distinguish "bleached"
from "white sand/skeleton" without context — this is precisely why the CNN fusion helps).

**Fusion.** Let $P_b$ be the CNN's bleached probability:

$$\rho = 100\,\big(\lambda P_b + (1-\lambda) W\big),\qquad \lambda = 0.6$$

**Ordinal mapping.** $\rho<15$: Healthy; $[15,35)$: Mild; $[35,60)$: Moderate; $[60,85)$: Severe; $\ge 85$: Dead.

**Class index weights** $w=(0,25,50,75,100)$ give the expected severity of a patch distribution:
$\bar\rho_{\text{class}} = \sum_k w_k\, \pi_k$ where $\pi_k$ is the share of class $k$.

**Ordinal loss (training).** $L = \mathrm{CE}(y,\hat y) + \lambda_o\, \mathbb{E}_{c\sim p_{\hat y}}|c-y|$ penalises
severity-distance-weighted mistakes during binary/5-class fine-tuning.

## 4. Phase 4 — Explainability
All three CAMs are implemented from scratch (see `src/explainability/cam.py`):
- **GradCAM** (Selvaraju 2017): $\alpha_k = \mathrm{GAP}(\partial y_c/\partial A_k)$, $M = \mathrm{ReLU}(\sum_k \alpha_k A_k)$.
- **GradCAM++** (Chattopadhyay 2018): $\alpha_{kl} = \dfrac{g_{kl}^2}{2g_{kl}^2 + \sum_{ab} A_{kab}\, g_{kl}^3 + \epsilon}$, $w_k=\sum_l \mathrm{ReLU}(g_{kl})\alpha_{kl}$.
- **ScoreCAM** (Wang 2020): per-channel normalised activations mask the input; channel weights are softmax scores of the masked forwards (batched, channel-capped on CPU).

## 5. Phase 5 — Real-time pipeline
Frame loop: enhance → segment (SmallUNet) → boxes (`connectedComponentsWithStats`, area filter)
→ per-box species softmax → centroid tracking (greedy nearest-neighbour with disappearance
buffer 15 frames) → unique-ID counting → WGBS on the frame → optional CAM overlay → annotate.

## 6. Phase 6 — Research analytics (formulas)

### 6.1 Marine Biodiversity Score (MBS)
With species counts $n_i$, $N=\sum_i n_i$, $p_i=n_i/N$, $S$ = species observed:

$$H' = -\sum_i p_i \ln p_i \qquad 1\text{-}D = 1-\sum_i p_i^2 \qquad d = \frac{S-1}{\ln N}$$

$$\boxed{\ \mathrm{MBS} = 100\left(0.40\,\frac{H'}{\ln S_{\text{ref}}} + 0.30\,(1-D) + 0.30\,\min\!\Big(\frac{d}{d_{\text{ref}}},1\Big)\right)\ },\quad S_{\text{ref}}=25,\ d_{\text{ref}}=3$$

Shannon captures evenness, Gini-Simpson dominance-resistance, Margalef richness — three
orthodox facets of diversity; the weighted mean is robust to any single index gaming.

### 6.2 Reef Health Score (RHS)
$$\boxed{\ \mathrm{RHS} = 100 - \big(0.60\,\bar\rho + 0.25\,(100\cdot f_{\text{bleached}}) + 0.15\,R_{\text{env}}\big)\ }$$

Bands: ≥80 Excellent · ≥60 Good · ≥40 Fair · ≥20 Poor · <20 Critical.

### 6.3 Population statistics
Unique-individual counts via tracker IDs; relative abundance $100\,n_i/N$; dominant/rare
species flags (rare $<1\%$ share); activity profile = detections per frame.

### 6.4 Bleaching trend
Per-observation BSI series $\rho_1..\rho_T$; EWMA $\tilde\rho_t = \alpha\rho_t + (1-\alpha)\tilde\rho_{t-1}$
($\alpha=0.3$); OLS slope $\beta$; labels: Improving ($\beta<-2$) · Stable ($|\beta|\le2$) ·
Declining ($\beta>2$) · Critical ($\bar\rho\ge60 \wedge \beta\ge0$).

### 6.5 Environmental risk
Turbidity proxy = mean dark channel $/0.35$ (clipped); visibility proxy = RMS contrast $/0.22$ (clipped);

$$R_{\text{env}} = 0.45\,\text{turb} + 0.25\,(1-\text{vis}) + 0.30\,\tanh(\beta) \quad (\times 100)$$

Bands: <25 Low · <50 Moderate · <75 High · ≥75 Critical.

## 7. Honest limitations (state clearly in the viva)
1. WGBS thresholds (15/35/60/85) are calibrated defaults, not clinically validated — sensitivity analysis is future work.
2. Binary coral labels mean the 5-class head is *derived*, not directly supervised; only the binary probability is learned.
3. Dead-coral detection via whiteness is weak when algae overgrowth darkens skeletons; the CNN term mitigates but cannot eliminate this.
4. Synthetic smoke-test models are for pipeline verification only — never report their metrics as results.
5. Species classifier is trained on a curated FishNet subset; closed-set assumption (unknown species will be forced into a known class — a rejection head is future work).
