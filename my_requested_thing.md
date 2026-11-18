# ReefScan AI — Experiment Ledger (my requested thing)

> **Purpose:** single source of truth for *which dataset, which cleaning, which model,
> which strategy (feature extraction vs fine-tuning), how many parameters, training time,
> final evaluation, and which model is finally chosen* after all GPU runs on Kaggle /
> Colab / Lightning finish. The **Experiment log** and **Final selection** sections are
> auto-generated from run manifests — never hand-edit numbers.

---

## 1. Datasets (what, why, where)

| Dataset | Task | Size | Splits | Chosen because |
|---|---|---|---|---|
| **NOAA PIFSC ESD Coral Bleaching** (`NMFS-OSI/…` on HuggingFace) | coral binary CORAL vs CORAL_BL (+ WGBS severity layer) | 10,419 × 224 px (~770 MB) | train 7,292 / validation 1,562 / test 1,565 | Only public bleaching dataset with honest labels; binary supervision is the basis of the WGBS severity innovation |
| **Fish4Knowledge v1 (F4K)** (University of Edinburgh, official GROUNDTRUTH mirror) | fish species classification from **verified crops** | single 487 MB tar + 0.6 MB class_id.csv; **verified layout**: `fish_image/fish_01..23/*.png` — 27,394 crops whose **directory IS the species label** (fish_NN == cluster N, verified count-for-count vs the official class_id.csv: 27,370 rows, 23 clusters); matching `mask_image/` masks included (future segmentation bonus) | seeded **video-group split** 80/10/10 (crops of one video segment share a split — prevents scene-level leakage) | Real reef-camera fish imagery with expert-verified species clusters; 13 of 23 clusters have ≥ 100 images (26,903 imgs kept); far cleaner than fishnet.ai (no bbox plumbing, no Human/No-fish exclusions, no missing-image gap) |
| **DeepFish** (`Alzayats/DeepFish` on HF) | fish segmentation → detection boxes | 620 pixel masks from 39,766 frames (~7.1 GB tar) — **verified live**: 1,244 seg files (620 images + 620 masks incl. 310 no-fish negative frames), SHA-256 match | 80/20 seeded split (seed 42) | Only free dataset with paired Full-HD frames + pixel masks; feeds the real-time box/counting pipeline |

Download: `python download_data.py --dataset {coral,f4k,deepfish}` — idempotent via completion markers; an *interrupted* download re-runs cleanly (partial archives resume, incomplete extractions are never mistaken for done).
F4K preparation (`src/data/prepare_f4k.py`) is written against the **verified** tar layout (labels carried by `fish_NN` directories); schema changes raise an explicit diagnostic.

## 2. Data cleaning (applied before every training run)

Command: `python clean_data.py --dataset {coral,f4k,seg} --apply` (report-only without `--apply`; nothing is deleted — flagged files move to `<root>/_quarantine/<reason>/`).

| Dataset | Recipe | Report |
|---|---|---|
| coral | corrupt/decode scan (PIL verify), min-size 32 px, exact-dup (MD5), near-dup (8×8 dHash), **train↔val↔test leakage** via dHash overlap | `results/cleaning/coral_cleaning_report.json` |
| f4k (species crops) | same as coral + class-count audit (warn < 10 crops) | `results/cleaning/f4k_cleaning_report.json` |
| seg (DeepFish) | image↔mask pairing validation, corrupt scan, **min mask-area** filter (foreground < 1% quarantined) | `results/cleaning/seg_cleaning_report.json` |

Cleaning report snapshot (F4K, measured locally on the real curated crops, 2026-11-15):
scanned 26,903 · corrupt 0 · too-small 5 · duplicate images 686 · leaked images 152 → **691 quarantined** (26,212 usable).
Report: `results/cleaning/f4k_cleaning_report.json`.

## 3. Models considered (screen → fine-tune protocol)

**Protocol decision:** screen all 4 backbones with cheap **feature extraction** (frozen backbone, head-only) → take the **top-2** by `val_weighted_f1` → **full fine-tune** those two → optionally `partial_finetune` the overall winner. Rationale: FE gives a reliable backbone-quality ranking at ~⅓ the GPU cost; fine-tuning only the winners saves ~50% total budget.

| Backbone | Approx params | Why it's in the screen |
|---|---|---|
| ResNet34 | ~21.3 M | Classic baseline, well-understood behaviour |
| EfficientNet-B0 | ~5.3 M | Best accuracy/FLOPs trade-off; likely overall winner |
| MobileNetV3-Small | ~2.5 M | Real-time candidate for the live pipeline |
| ConvNeXt-Tiny | ~28.6 M | Strongest modern convnet; accuracy ceiling |
| SmallUNet (seg) | ~1.9 M | Dedicated fish-mask U-Net (from scratch) |

## 4. Transfer-learning strategies (implemented, pickable per run)

| Strategy | What trains | LR scheme | When to use |
|---|---|---|---|
| `feature_extraction` | head only (backbone frozen) | head @ lr×10 | E1/E4 screening — fast, stable |
| `finetune` | everything | backbone @ lr, head @ lr×10 | E2/E5 finalisation — best accuracy |
| `partial_finetune` | head always; last stage after `freeze_epochs` | head @ lr×10, last stage @ lr×0.1 | Middle ground when FT overfits |

Run with `python train.py --task coral --model all --strategy feature_extraction --platform kaggle`.

## 5. Evaluation metrics (computed and stored per run)

| Task | Metrics | Artifacts |
|---|---|---|
| fish / coral | top-1 acc, top-5 acc, **macro AP (one-vs-rest)**, per-class AP, macro/weighted F1, precision/recall, confusion matrix, **PR curves**, throughput fps | `results/<task>/<model>/metrics.json`, `per_class_report.csv`, `pr_curves.png`, `confusion_matrix.png`, `model_leaderboard.csv` |
| seg | **IoU**, **Dice**, **pixel AP**, **box AP@0.5** (greedy IoU matching vs GT boxes), overlays | `results/seg/metrics.json`, `overlay_*.png` |
| every run | platform, GPU, git SHA, params (total/trainable), **train time**, config snapshot | `runs/<task>/<model>_<strategy>/manifest.json` |

## 6. Platform plan (no local CPU training)

| Platform | GPU | Role in matrix |
|---|---|---|
| **Kaggle** (T4×2, 30 h/wk, phone-verified ✓) | primary — can run the FULL matrix E1–E6 | coral screen → FT → seg → fish FE/FT |
| **Colab** (T4, free tier) | secondary | E4 fish FE screen → E5 fish FT → E6 XAI/enhancement |
| **Lightning AI** (T4, free credits) | backup | E7 re-run any failure; overflow fish fine-tune |
| **GPU workstation** (colleague's PC, HANDOFF.md) | local NVIDIA GPU | can run the full E1–E6 matrix in one session — `--platform gpu` |

Full runbook with copy-paste commands: [`PLATFORMS.md`](PLATFORMS.md).
**Before closing any session:** `python scripts/export_artifacts.py` → download the zip (Kaggle Output panel / Colab `files.download` / Lightning file browser).

<!-- EXPERIMENT-LOG:START -->

### Experiment log (auto-generated — do not edit by hand)

Refresh with: `python scripts/log_experiment.py` after every training run.

| run | platform | dataset | strategy | params (tot/train) | train time | acc | mAP | macro-F1 | wF1 | IoU | Dice | box-AP50 | git | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _(no runs recorded yet)_ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

<!-- EXPERIMENT-LOG:END -->

---

## 7. Final selection

<!-- FINAL-SELECTION:START -->
_Pending — run `python scripts/compare_runs.py` after all platform training finishes; the
winner per task (fish / coral / seg) with rationale will appear here automatically._
<!-- FINAL-SELECTION:END -->

## 8. What was finally deployed

*(fill after final selection: best coral checkpoint drives the WGBS severity layer in the
dashboard + reports; best fish checkpoint drives species pages; seg U-Net drives the
real-time box/counting pipeline — all shipped in the downloaded artifacts zip.)*
