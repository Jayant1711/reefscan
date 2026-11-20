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
| **Kaggle** (T4×2, 30 h/wk, phone-verified ✓) | **the only training platform** (owner decision, 2026-11-18) | Full matrix E1–E6: coral screen → FT → seg → fish FE/FT → evals |
| ~~Colab / Lightning AI~~ | — | Removed from the plan (owner decision); kept in repo as optional backup lanes only |
| **GPU workstation** (colleague's PC, HANDOFF.md) | local NVIDIA GPU | can run the full E1–E6 matrix in one session — `--platform gpu` |

Full runbook with copy-paste commands: [`PLATFORMS.md`](PLATFORMS.md).
**Before closing any session:** `python scripts/export_artifacts.py` → download the zip (Kaggle Output panel / Colab `files.download` / Lightning file browser).

<!-- EXPERIMENT-LOG:START -->

### Experiment log (auto-generated — do not edit by hand)

Refresh with: `python scripts/log_experiment.py` after every training run.

| run | platform | dataset | strategy | params (tot/train) | train time | acc | mAP | macro-F1 | wF1 | IoU | Dice | box-AP50 | git | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coral::convnext_tiny::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | feature_extraction | 27.82M / 0.00M | 4.1 min | 0.8735 | 0.9215 | 0.8663 | 0.8739 | — | — | — | 280cb67 | runs\coral\convnext_tiny_feature_extraction\manifest.json |
| coral::efficientnet_b0::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | feature_extraction | 4.01M / 0.64M | 3.4 min | 0.8824 | 0.9349 | 0.8754 | 0.8827 | — | — | — | 280cb67 | runs\coral\efficientnet_b0_feature_extraction\manifest.json |
| coral::efficientnet_b0::finetune | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | finetune | 4.01M / 4.01M | 6.1 min | 0.8984 | 0.9458 | 0.8922 | 0.8986 | — | — | — | 280cb67 | runs\coral\efficientnet_b0_finetune\manifest.json |
| coral::mobilenet_v3::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | feature_extraction | 1.52M / 1.05M | 5.5 min | 0.8990 | 0.9513 | 0.8928 | 0.8991 | — | — | — | 280cb67 | runs\coral\mobilenet_v3_feature_extraction\manifest.json |
| coral::mobilenet_v3::finetune | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | finetune | 1.52M / 1.52M | 5.7 min | 0.8965 | 0.9518 | 0.8904 | 0.8967 | — | — | — | 280cb67 | runs\coral\mobilenet_v3_finetune\manifest.json |
| coral::mobilenet_v3::partial_finetune | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | partial_finetune | 1.52M / 1.11M | 5.2 min | 0.8888 | 0.9411 | 0.8815 | 0.8887 | — | — | — | 280cb67 | runs\coral\mobilenet_v3_partial_finetune\manifest.json |
| coral::resnet34::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/coral/imagefolders | feature_extraction | 21.29M / 0.00M | 5.8 min | 0.8160 | 0.8748 | 0.7987 | 0.8131 | — | — | — | 280cb67 | runs\coral\resnet34_feature_extraction\manifest.json |
| fish::convnext_tiny::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/f4k/curated | feature_extraction | 27.83M / 0.01M | 5.9 min | 0.9487 | 0.9852 | 0.9439 | 0.9505 | — | — | — | b98946c | runs\fish\convnext_tiny_feature_extraction\manifest.json |
| fish::efficientnet_b0::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/f4k/curated | feature_extraction | 4.02M / 0.65M | 5.9 min | 0.9879 | 0.9993 | 0.9859 | 0.9881 | — | — | — | b98946c | runs\fish\efficientnet_b0_feature_extraction\manifest.json |
| fish::efficientnet_b0::finetune | kaggle | /kaggle/tmp/reefscan_data/f4k/curated | finetune | 4.02M / 4.02M | 4.3 min | 0.9909 | 0.9980 | 0.9870 | 0.9910 | — | — | — | b98946c | runs\fish\efficientnet_b0_finetune\manifest.json |
| fish::mobilenet_v3::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/f4k/curated | feature_extraction | 1.53M / 1.07M | 5.8 min | 0.9804 | 0.9979 | 0.9727 | 0.9808 | — | — | — | b98946c | runs\fish\mobilenet_v3_feature_extraction\manifest.json |
| fish::resnet34::feature_extraction | kaggle | /kaggle/tmp/reefscan_data/f4k/curated | feature_extraction | 21.29M / 0.01M | 4.8 min | 0.8733 | 0.9404 | 0.8468 | 0.8781 | — | — | — | b98946c | runs\fish\resnet34_feature_extraction\manifest.json |
| seg::unet_small::from_scratch | kaggle | /kaggle/tmp/reefscan_data/deepfish | from_scratch | 1.94M / 1.94M | 1.4 min | — | — | — | — | 0.2791 | 0.4344 | — | 280cb67 | runs\seg\unet_small_from_scratch\manifest.json |

<!-- EXPERIMENT-LOG:END -->

---

## 7. Final selection

<!-- FINAL-SELECTION:START -->

### Final selection (auto-generated by scripts/compare_runs.py)

**coral** → `mobilenet_v3` (feature_extraction, trained on kaggle) — best weighted_f1 = **0.8991**, 1.52M params, 5.5 min training. Best feature-extraction screen: `mobilenet_v3`.

**fish** → `efficientnet_b0` (finetune, trained on kaggle) — best weighted_f1 = **0.9910**, 4.02M params, 4.3 min training. Best feature-extraction screen: `efficientnet_b0`.

**seg** → `unet_small` (from_scratch, trained on kaggle) — best dice = **0.4344**, 1.94M params, 1.4 min training. Best feature-extraction screen: `—`.

_Rationale: highest task metric wins; ties broken by fewer parameters (cheaper inference for the real-time pipeline) then training cost. Run `python scripts/compare_runs.py` after all platform runs to refresh._

<!-- FINAL-SELECTION:END -->

## 8. What was finally deployed

All 13 runs trained end-to-end on **Kaggle T4** in one session (2026-11-20, git b98946c),
artefacts zipped (416.8 MB) and downloaded to the owner's laptop —
`reefscan_artifacts_kaggle_b98946c_20261120_155318.zip`, integrity-verified (103 entries, 9 checkpoints):

- **coral (bleaching)** → `mobilenet_v3` feature_extraction — wF1 **0.8991**, mAP 0.951, 500 FPS, 1.52M params.
  Drives the WGBS severity estimator + risk bands in the dashboard and the PDF reports.
- **fish (species, F4K 13 classes)** → `efficientnet_b0` fine-tuned — acc **0.9910**, wF1 **0.9910**, mAP 0.998, 515 FPS, 4.02M params.
  Drives species identification pages and the biodiversity/analytics layer.
- **seg (DeepFish)** → `unet_small` from scratch — Dice **0.434**, boxAP@0.5 **0.892** (official 262-pair split).
  Drives real-time fish boxes + counting (masks_to_boxes → tracker).
- XAI (Grad-CAM) wired for the fish task; 105-test suite green on Kaggle (104 passed, 1 skipped);
  every run has a manifest (git SHA, platform, GPU, strategy, params, epochs, seconds, metrics).

Training fixes shipped during the live session: `bc134c9` (criterion device parity — T4 crash),
`280cb67` (DeepFish discovery scoped to the official Segmentation split), `8f4b447`/`b98946c`/`e0a1a66`
(F4K long-tail head-cap 1000/class + force-rebuild purge + coral Linux case-insensitive ready-check).
