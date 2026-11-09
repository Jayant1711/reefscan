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
| **fishnet.ai v0.1.2** (The Nature Conservancy, fishnet.ai/download, S3 mirror) | fish species classification from **bbox crops** | labels 2.1 MB CSV + images 5.7 GB; **verified schema**: `img_id, bbox_id, x_min, x_max, y_min, y_max, label_l1, label_l2` — 159,119 boxes / 34,987 frames / 28 L1 species classes (top: Albacore 33.9K, Yellowfin tuna 6.7K, Bigeye tuna 2.8K, Opah 1.6K, Mahi 1.1K…) / 13 L2 codes | derived **seeded image-level split** 80/10/10 (no official split column in v0.1.2; every crop of one frame stays in one split — no leakage) | Real EM-camera fish imagery with species labels; `Human`(104K)/`Unknown`/`No fish` boxes excluded; classes with <100 boxes dropped → ~10–12 species classes; crops mirror the real-time pipeline's per-box classification |
| **DeepFish** (`Alzayats/DeepFish` on HF) | fish segmentation → detection boxes | 620 pixel masks from 39,766 frames (~7.1 GB tar) | 80/20 seeded split (seed 42) | Only free dataset with paired Full-HD frames + pixel masks; feeds the real-time box/counting pipeline |

Download: `python download_data.py --dataset {coral,fishnet,deepfish}` — all idempotent (skip if present).
FishNet preparation (`src/data/prepare_fishnet.py`) is written against the **verified** v0.1.2 schema and crops every fish bbox with a 5% context margin; schema changes raise an explicit diagnostic.

## 2. Data cleaning (applied before every training run)

Command: `python clean_data.py --dataset {coral,fishnet,seg} --apply` (report-only without `--apply`; nothing is deleted — flagged files move to `<root>/_quarantine/<reason>/`).

| Dataset | Recipe | Report |
|---|---|---|
| coral | corrupt/decode scan (PIL verify), min-size 32 px, exact-dup (MD5), near-dup (8×8 dHash), **train↔val↔test leakage** via dHash overlap | `results/cleaning/coral_cleaning_report.json` |
| fishnet (bbox crops) | same as coral + class-count audit (warn < 10 crops) | `results/cleaning/fishnet_cleaning_report.json` |
| seg (DeepFish) | image↔mask pairing validation, corrupt scan, **min mask-area** filter (foreground < 1% quarantined) | `results/cleaning/seg_cleaning_report.json` |

Cleaning report snapshot: *(fill after first platform run)*

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
| **Kaggle** (T4×2, 30 h/wk, phone-verified ✓) | primary | E1 coral FE screen → E2 coral FT → E3 seg |
| **Colab** (T4, free tier) | secondary | E4 fish FE screen → E5 fish FT → E6 XAI/enhancement |
| **Lightning AI** (T4, free credits) | backup | E7 re-run any failure; overflow fish fine-tune |

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
