# 🌊 ReefScan AI

**Underwater Marine Species & Coral Reef Health Visual Assessment System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![CPU friendly](https://img.shields.io/badge/runs%20on-CPU%20%7C%20Colab%20%7C%20Kaggle-success)](#-quick-start)
[![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20smoke-informational)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ReefScan AI is an end-to-end, research-grade computer-vision pipeline that turns ordinary
underwater photos and survey videos into **actionable reef reports**:

| Phase | Capability | Where |
|---|---|---|
| 1 | **8 enhancement strategies** (CLAHE, DCP, UDCP, white balance, color restoration, gamma, HE, fusion) benchmarked with **PSNR / SSIM / UCIQE / UIQM** + automatic best-strategy selection | `src/enhancement/` |
| 2 | **Species classification benchmark**: ResNet34 · EfficientNet-B0 · MobileNetV3 · ConvNeXt-Tiny with class balancing, underwater augmentation, mixup, leaderboards | `src/classification/` |
| 3 | **Coral bleaching analysis**: binary NOAA-trained CNN + **WGBS** → 0–100 severity index and 5 classes (Healthy → Dead), risk score, health reports | `src/coral_analysis/` |
| 4 | **Explainable AI**: GradCAM, GradCAM++, ScoreCAM — implemented from scratch, downloadable heatmaps | `src/explainability/` |
| 5 | **Real-time**: image / video / webcam with segmentation-based fish boxes, unique-ID counting, annotated MP4 output | `src/realtime/` |
| 6 | **Research analytics**: Marine Biodiversity Score, Reef Health Score, population statistics, bleaching trends, environmental risk | `src/analytics/` |
| 7 | **Streamlit dashboard** (9 pages) | `app.py` |
| 8 | **PDF reports** (reportlab) with gauges, detections, heatmaps, recommendations | `src/reporting/` |
| 9 | **IEEE-format research paper** (LaTeX + Markdown) | `research_paper/` |
| 10 | **Full documentation** incl. setup, viva Q&A, interview prep | `docs/` |

Everything runs **100% free** — laptop CPU, Google Colab or Kaggle.

---

## 🚀 Quick start

```bash
# 1) install
pip install -r requirements.txt

# 2) prove the whole system works end-to-end (synthetic data, CPU, ~5-10 min)
python run_smoke_test.py

# 3) open the dashboard — demo models are already active
streamlit run app.py
```

Then, for real research results:

```bash
# datasets (~13.6 GB total; each can run independently — see docs/DATASETS.md)
python download_data.py --dataset coral      # NOAA coral, ~770 MB (easiest)
python download_data.py --dataset deepfish   # DeepFish, ~7.1 GB
python download_data.py --dataset fishnet    # fishnet.ai v0.1.2, ~5.7 GB S3 (resumable curl)

# training
python train.py --task fish --model all      # 4-backbone species benchmark
python train.py --task coral --model all     # coral health benchmark
python train.py --task seg                   # fish-box U-Net

# evaluation artefacts: metrics, leaderboards, confusion matrices, XAI grids
python evaluate.py --task fish
python evaluate.py --task coral
python evaluate.py --enhancement --data datasets/fishnet/curated/train
python evaluate.py --xai --image sample.jpg

# inference
python inference.py --mode image --input photo.jpg --cam --report
python inference.py --mode video --input survey.mp4
python inference.py --mode webcam
```

Cloud training: open the ready-to-run platform notebooks — `notebooks/platform/KAGGLE_training.ipynb`,
`COLAB_training.ipynb`, `LIGHTNING_backup.ipynb` — they auto-clone, gate on the test suite,
train the E-matrix, and zip artifacts for download. See `PLATFORMS.md` + `STEPS_TO_DO.md`.

---

## 🧠 The core innovation — WGBS severity estimation

Public coral datasets (including NOAA's) label bleaching **binary** (coral vs bleached),
but reef managers need **graded severity**. ReefScan fuses the CNN's learned bleached
probability with a physics-motivated whiteness measurement:

```
W   = fraction of pixels with HSV(S < 70, V > 140)      ← pale-bright tissue evidence
BSI = 100 · (0.60·P(bleached) + 0.40·W)                 ← 0–100 severity index
BSI → Healthy [0,15) · Mild [15,35) · Moderate [35,60) · Severe [60,85) · Dead [85,100]
```

Every learned component stays under honest binary supervision — the graded output is a
*measurement*, not fabricated labels. Full derivation and limitations:
[`docs/RESEARCH_METHODOLOGY.md`](docs/RESEARCH_METHODOLOGY.md) §3.

## 📊 Research analytics (Phase 6)

| Score | Formula (weights configurable) |
|---|---|
| **MBS** — Marine Biodiversity | `100·(0.40·H'/lnS_ref + 0.30·(1−D) + 0.30·min(d/d_ref,1))` with Shannon, Gini-Simpson, Margalef |
| **RHS** — Reef Health | `100 − (0.60·BSI + 0.25·bleached% + 0.15·envRisk)` |
| Population stats | unique-tracked counts, abundance %, dominant/rare species |
| Trends | EWMA + OLS slope of the severity series → Improving/Stable/Declining/Critical |
| Environmental risk | `0.45·turbidity + 0.25·(1−visibility) + 0.30·tanh(slope)` from image physics proxies |

## 🗂 Repository layout

```
├── app.py                  # Streamlit dashboard entry (9 pages)
├── train.py                # fish / coral / seg training CLI
├── evaluate.py             # leaderboards, enhancement benchmark, XAI grids
├── inference.py            # image / video / webcam pipeline
├── download_data.py        # all three datasets + FishNet subset builder
├── run_smoke_test.py       # end-to-end synthetic verification (CPU)
├── configs/default.yaml    # single source of truth for every knob
├── src/
│   ├── enhancement/  classification/  coral_analysis/  segmentation/
│   ├── explainability/  realtime/  analytics/  reporting/  dashboard/
│   ├── data/               # downloaders + prepare_fishnet
│   └── utils/  config/
├── notebooks/              # 7 Colab-ready notebooks (incl. all-in-one GPU run)
├── tests/                  # pytest: hand-computed metrics, contracts, CAMs, trackers
├── docs/                   # SETUP · DATASETS · TRAINING · DEPLOYMENT · API ·
│                           # USER_MANUAL · METHODOLOGY · VIVA Q&A · INTERVIEW · …
├── research_paper/         # reefscan_ieee.tex + markdown mirror + compile guide
├── datasets/ models/ results/ reports/   # artefacts (git-ignored, .gitkeep'd)
```

## 📈 What the smoke test proves

`python run_smoke_test.py` builds synthetic mini-datasets and executes **every phase**:
enhancement benchmark → fish training (2 backbones) → coral training + WGBS → U-Net
segmentation → GradCAM/++/ScoreCAM grids → full video pipeline → MBS/RHS analytics →
both PDF reports. It then promotes demo checkpoints so the dashboard runs instantly.
Any failure anywhere fails loudly with a traceback — no silent degradation in CI.

## 📚 Documentation

| Document | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | install (local/Colab/Kaggle), verification, troubleshooting |
| [docs/DATASETS.md](docs/DATASETS.md) | all 3 datasets, download + manual fallbacks, licensing |
| [docs/TRAINING.md](docs/TRAINING.md) | training recipes, hyperparameters, expected runtimes |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | dashboard, Colab, Docker, CPU optimisation |
| [docs/API.md](docs/API.md) | Python API for every module |
| [docs/RESEARCH_METHODOLOGY.md](docs/RESEARCH_METHODOLOGY.md) | every formula + honest limitations |
| [docs/VIVA_QUESTIONS.md](docs/VIVA_QUESTIONS.md) / [ANSWERS](docs/VIVA_ANSWERS.md) | 38 viva questions with model answers |
| [docs/INTERVIEW_QUESTIONS.md](docs/INTERVIEW_QUESTIONS.md) | industry-style probing questions |
| [research_paper/](research_paper/) | IEEE LaTeX paper + markdown + Overleaf guide |

## 🧪 Tests

```bash
python -m pytest tests/ -q   # 80 unit tests: metric identities, WGBS math, CAMs, tracker, cleaning, export parity
```

## 🙏 Datasets & citations

- **fishnet.ai v0.1.2** (The Nature Conservancy) — https://www.fishnet.ai/download (arXiv:2106.09178)
- **DeepFish** (Scientific Reports 2020) — Saleh et al. — https://alzayats.github.io/DeepFish/
- **NOAA PIFSC ESD Coral Bleaching** — https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset

Please cite the original papers (BibTeX in `research_paper/reefscan_ieee.tex`) when using ReefScan outputs.

## 📄 License

MIT — see [LICENSE](LICENSE). Datasets remain under their own licenses.
