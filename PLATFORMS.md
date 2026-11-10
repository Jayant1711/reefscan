# PLATFORMS.md — Multi-GPU training runbook (Kaggle · Colab · Lightning AI)

**Golden rules**

1. Never train on the local laptop CPU — the experiment matrix (E1–E7 in `my_requested_thing.md`) runs entirely on free cloud GPUs.
2. One repo, zero interference: every platform runs from its own fresh `git clone`; all artefacts stay **inside the clone** (models/, results/, runs/, reports/) except datasets, which platform profiles relocate to ephemeral dirs (`configs/platform/<name>.yaml`).
3. **Always run `python scripts/export_artifacts.py` and download the zip before a session ends.** Sessions die; your laptop doesn't.

---

## 0. Common first cell (every platform)

```bash
!git clone https://github.com/Jayant1711/reefscan.git
%cd reefscan
!pip install -q -r requirements.txt
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY — STOP")
!python -m pytest tests/ -q    # verify the full suite (69 tests) on the platform
```

GPU runtimes (enable BEFORE running any cell):

| Platform | How to enable GPU | Notes |
|---|---|---|
| Kaggle | right sidebar **Session options → Accelerator → GPU T4×2** (requires phone-verified account) + **Internet → On** | 30 h/week quota, sessions up to 12 h |
| Colab | Runtime → **Change runtime type → T4 GPU** | Free tier: retry if "Allocating runtime" hangs (peak-hour capacity), best 06:00–10:00 IST |
| Lightning AI | Studio → **switch machine → GPU (T4)** | ~15 free credits/hours monthly |

---

## 1. Kaggle (primary — experiments E1, E2, E3)

```bash
# 0) setup (common first cell above)
# 1) datasets -> /kaggle/tmp (ephemeral; the 20 GB Output carries only artefacts)
!python download_data.py --dataset coral --platform kaggle
!python download_data.py --dataset deepfish --platform kaggle
# FishNet (E4/E5 run on Colab; skip here unless re-assigned):
# !python download_data.py --dataset fishnet

# 2) clean (report -> quarantine flagged files)
!python clean_data.py --dataset coral --apply --platform kaggle
!python clean_data.py --dataset seg --apply --platform kaggle

# 3) E1 — coral FE screen (all 4 backbones, frozen backbone, ~35 min on T4×2)
!python train.py --task coral --model all --strategy feature_extraction --platform kaggle

# 4) E2 — coral fine-tune top-2 (pick from results/coral/model_leaderboard.csv)
!python train.py --task coral --model efficientnet_b0 --strategy finetune --platform kaggle
!python train.py --task coral --model convnext_tiny  --strategy finetune --platform kaggle

# 5) E3 — fish segmentation U-Net (~30 min)
!python train.py --task seg --platform kaggle
!python evaluate.py --task seg --platform kaggle   # IoU / Dice / pixel-AP / box-AP50

# 6) SAVE — zip + record (before session ends!)
!python evaluate.py --task coral --platform kaggle
!python scripts/log_experiment.py
!python scripts/export_artifacts.py
```

Download: Kaggle notebook **Output panel** → the `reefscan_artifacts_kaggle_*.zip`.

---

## 2. Colab (secondary — experiments E4, E5, E6)

```python
# 0) setup (common first cell above)
# 1) dataset (FishNet v0.1.2 from the fishnet.ai S3 mirror)
!python download_data.py --dataset fishnet --platform colab
!python download_data.py --prepare-fishnet --platform colab
!python clean_data.py --dataset fishnet --apply --platform colab

# 2) E4 — fish FE screen (all 4 backbones, ~1.5 h on T4)
!python train.py --task fish --model all --strategy feature_extraction --platform colab

# 3) E5 — fish fine-tune the E4 winner
!python train.py --task fish --model <winner> --strategy finetune --platform colab

# 4) E6 — XAI + enhancement benchmark
!python evaluate.py --task fish --platform colab
!python evaluate.py --enhancement --data /content/reefscan_data/fishnet/curated/train
!python evaluate.py --xai --image <any curated train image>

# 5) SAVE (before session ends!)
!python scripts/log_experiment.py
!python scripts/export_artifacts.py
from google.colab import files
import glob; files.download(sorted(glob.glob("/content/reefscan/reefscan_artifacts_colab_*.zip"))[-1])
```

Colab "Allocating runtime" hangs? → Disconnect & reconnect, or switch to Lightning for that phase (E7).

---

## 3. Lightning AI (backup — experiment E7)

Everything stays repo-relative (the studio persists); only bump workers/batch via the profile.

```bash
# inside the studio terminal
git clone https://github.com/Jayant1711/reefscan.git && cd reefscan
pip install -q -r requirements.txt

python download_data.py --dataset coral        # or fishnet, per the failed experiment
python clean_data.py --dataset coral --apply
python train.py --task coral --model <model> --strategy <strategy> --platform lightning
python scripts/log_experiment.py
python scripts/export_artifacts.py
```

Download: studio **file browser** → `reefscan_artifacts_lightning_*.zip`.

---

## 4. After all platforms finish (on your laptop)

Unzip every artefact bundle into the repo (models/, results/, runs/, reports/ merge cleanly — per-run dirs never collide), then:

```bash
python scripts/compare_runs.py     # -> results/comparison/final_comparison.csv + ledger Final selection
python scripts/log_experiment.py   # refresh the ledger tables
pytest tests/ -q                   # sanity: everything still green
streamlit run app.py               # dashboard now uses the REAL trained models
```

The **Final selection** section of `my_requested_thing.md` is written automatically:
best model per task, its strategy, params, training time, and the FE-screen runner-up.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Kaggle session fails to start | Known transient incident — retry later; move that phase to Colab/Lightning (E7). **Save & Run All re-downloads ~8 GB** (datasets live in `/kaggle/tmp`), so prefer interactive runs + the Output-panel zip. |
| Colab "Allocating runtime" forever | Free-tier capacity — retry off-peak or switch platform |
| `CUDA out of memory` | `--set <task>.batch_size=32` (or 16) |
| FishNet download slow/fails | fishnet.ai S3 mirror with resume (`curl -f -C -`): re-run the cell — partial zips resume. If extraction died mid-way, delete the dataset's `_EXTRACTED` marker to force a clean re-extract. |
| Zip too big for browser download | `python scripts/export_artifacts.py --no-models` for a metrics-only zip, download checkpoints individually from the Output panel |
| Lost a session before exporting | The experiment matrix rows for that platform are simply re-run — manifests under `runs/` on your laptop tell you which |
