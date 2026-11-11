# Setup Guide — ReefScan AI

## 1. Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.10–3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 3 GB (code + demo) | 30 GB (all datasets) |
| GPU | none (CPU works) | any CUDA GPU / Colab T4 |

## 2. Installation

### Option A — local (Windows / Linux / macOS)
```bash
git clone <your-repo-url> reefscan
cd reefscan
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> **CPU-only PyTorch:** on Windows/Linux the default `pip install torch` wheel is CPU. On Linux with NVIDIA GPUs use:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Option B — Google Colab (free GPU)
1. Open [colab.research.google.com](https://colab.research.google.com) → New notebook.
2. Runtime → Change runtime type → **T4 GPU**.
3. Follow `notebooks/00_ReefScan_Colab_AllInOne.ipynb`.

### Option C — Kaggle
Same as Colab; datasets can be attached as Kaggle Datasets to skip download time.

## 3. Verify the installation
```bash
python -c "import torch, torchvision, cv2, streamlit; print('ok')"
pytest tests/ -q            # unit tests (~1 min)
python run_smoke_test.py    # end-to-end demo on synthetic data (~5-10 min CPU)
```
After the smoke test, `streamlit run app.py` works immediately with **demo models**.

## 4. Configure
All settings live in [`configs/default.yaml`](../configs/default.yaml). Override any key from the CLI:
```bash
python train.py --task fish --model resnet34 --set fish.epochs=5 fish.batch_size=16
```

## 5. Folder contract
| Folder | Purpose |
|---|---|
| `datasets/` | downloaded + curated data (git-ignored) |
| `models/` | checkpoints (`models/<task>/<model>/best.pt`) + `models/demo/` |
| `results/` | metrics, leaderboards, plots, logs |
| `reports/` | generated PDF reports |

## 6. Troubleshooting
- **`torch.load` fails with weights_only** — our checkpoints are plain payload types and load with `weights_only=True` (the safe default since torch 2.4); if you load a foreign checkpoint and hit a weights_only error, do NOT flip it off blindly - only load checkpoints you trust.
- **Windows DataLoader hang** — keep `device.num_workers: 0` (default).
- **gdown quota error for FishNet** — download manually per `docs/DATASETS.md`.
- **OpenCV 5 + old numpy** — `pip install "numpy>=1.26"` resolves most ABI issues.
