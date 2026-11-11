#!/usr/bin/env python
"""Generate the 7 ReefScan notebooks (Colab-ready) into notebooks/.

Run from the project root:  python scripts/build_notebooks.py
No third-party deps — plain json construction.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def cell(kind: str, source: str) -> dict:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_COMMON = """\
# --- Environment check -------------------------------------------------------
import sys, pathlib
IN_COLAB = "google.colab" in sys.modules
PROJECT = pathlib.Path("/content/reefscan") if IN_COLAB else pathlib.Path.cwd()
print("Colab:", IN_COLAB, "| project:", PROJECT)
if IN_COLAB:
    import os
    from google.colab import drive
    # Mount Drive so datasets/checkpoints survive session resets (optional).
    # drive.mount("/content/drive")
"""

INSTALL_COLAB = """\
# --- (Colab only) clone the repository ---------------------------------------
# Clone the repo (uncomment and run ONCE if /content/reefscan does not exist):
# !git clone https://github.com/Jayant1711/reefscan.git /content/reefscan
import os
if os.path.isdir("/content/reefscan"):
    %cd /content/reefscan
else:
    print("NOTE: /content/reefscan missing — uncomment the git clone line above, run it, then re-run this cell.")
!pip install -q gdown pyarrow reportlab streamlit
"""

NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["00_ReefScan_Colab_AllInOne.ipynb"] = [
    ("markdown", "# ReefScan AI — All-in-One Colab Pipeline (GPU)\n\n"
                 "Runs every phase end-to-end on a **free Colab T4**: dataset download → enhancement "
                 "benchmark → fish + coral training (4 backbones) → segmentation → XAI → analytics → PDF reports.\n\n"
                 "**Runtime → Change runtime type → T4 GPU** before running."),
    ("code", INSTALL_COLAB.replace("{PROJECT_ROOT}", "/content/reefscan")),
    ("code", SETUP_COMMON),
    ("code", """\
import sys
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.config import load_config, ensure_output_dirs

cfg = load_config(PROJECT / "configs/default.yaml")
ensure_output_dirs(cfg)
import torch
print("CUDA available:", torch.cuda.is_available())
"""),
    ("markdown", "## 1. Datasets (~13.6 GB total — DeepFish is the big one)"),
    ("code", """\
# Coral (770 MB) and DeepFish (7.1 GB) are automatic.
# FishNet comes from the fishnet.ai S3 mirror (resumable curl, no quota);
# download manually (see docs/DATASETS.md) into datasets/fishnet/raw/.
!python download_data.py --dataset all
!python download_data.py --prepare-fishnet
"""),
    ("markdown", "## 2. Enhancement benchmark (Phase 1)"),
    ("code", """\
!python evaluate.py --enhancement --data datasets/fishnet/curated/train
import json, pathlib
print(json.dumps(json.load(open("results/enhancement/enhancement_report.json")), indent=2))
"""),
    ("markdown", "## 3. Species classification — 4 backbones (Phase 2)"),
    ("code", """\
# 15 epochs x 4 models ≈ 3-5 h on T4. Trim with --set fish.epochs=5 for a quicker pass.
!python train.py --task fish --model all
"""),
    ("code", """\
import pandas as pd
pd.read_csv("results/fish/model_leaderboard.csv")
"""),
    ("markdown", "## 4. Coral bleaching — 4 backbones + WGBS (Phase 3)"),
    ("code", """\
!python train.py --task coral --model all
pd.read_csv("results/coral/model_leaderboard.csv")
"""),
    ("markdown", "## 5. Fish segmentation for detection boxes (Phase 3.5)"),
    ("code", "!python train.py --task seg"),
    ("markdown", "## 6. Explainability (Phase 4) + inference demo (Phase 5)"),
    ("code", """\
import glob, cv2
sample = sorted(glob.glob("datasets/coral/imagefolders/test/*/*.jpg"))[0]
!python evaluate.py --xai --image {sample}
!python inference.py --mode image --input {sample} --cam --report
"""),
    ("markdown", "## 7. Launch the dashboard inside Colab"),
    ("code", """\
from pyngrok import ngrok  # or use the link printed by cloudflared below
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
"""),
    ("markdown", "## 8. Save checkpoints to Google Drive"),
    ("code", """\
# from google.colab import drive
# drive.mount("/content/drive")
# !cp -r models /content/drive/MyDrive/reefscan_models/
# !cp -r results reports /content/drive/MyDrive/reefscan_outputs/
"""),
]

NOTEBOOKS["01_data_download_and_preparation.ipynb"] = [
    ("markdown", "# 01 — Dataset Download & Preparation\n\nFishNet · DeepFish · NOAA coral. See `docs/DATASETS.md` for manual fallbacks."),
    ("code", SETUP_COMMON),
    ("code", "!python download_data.py --dataset coral"),
    ("code", """\
# Verify the coral ImageFolder splits
from pathlib import Path
for split in ("train", "validation", "test"):
    root = Path("datasets/coral/imagefolders") / split
    if root.exists():
        counts = {d.name: len(list(d.glob('*.jpg'))) for d in root.iterdir() if d.is_dir()}
        print(split, counts)
"""),
    ("code", """\
# DeepFish (~7.1 GB) — classification/localization/segmentation
!python download_data.py --dataset deepfish
"""),
    ("code", """\
# FishNet (~5.7 GB + 2.1 MB labels) + curated taxonomy-level subset
!python download_data.py --dataset fishnet
"""),
    ("code", """\
# Inspect the curated subset (class distribution)
import pandas as pd
from src.classification.class_balancing import count_class_images
counts = count_class_images("datasets/fishnet/curated/train")
pd.Series(counts).sort_values(ascending=False).plot(kind="bar", figsize=(12, 3));
"""),
]

NOTEBOOKS["02_enhancement_benchmark.ipynb"] = [
    ("markdown", "# 02 — Underwater Enhancement Benchmark (Phase 1)\n\n"
                 "8 strategies × 4 metrics (PSNR, SSIM, UCIQE, UIQM) with automatic best-strategy selection."),
    ("code", SETUP_COMMON),
    ("code", """\
import cv2, glob
from src.enhancement.selector import EnhancementSelector

images = [cv2.imread(p) for p in sorted(glob.glob("datasets/synthetic/enhance/*.jpg"))] or \\
         [cv2.imread(p) for p in sorted(glob.glob("datasets/fishnet/curated/train/*/*.jpg"))[:8]]
selector = EnhancementSelector(weights={"uciqe": 0.4, "uiqm": 0.3, "ssim": 0.3})
best, report, grid = selector.select_best(images, out_dir="results/enhancement")
selector.save_report(report, best, "results/enhancement")
print("Best strategy:", best)
report
"""),
    ("code", """\
from IPython.display import Image as IPyImage
IPyImage("results/enhancement/enhancement_comparison.png", width=1000)
"""),
]

NOTEBOOKS["03_fish_classification_training.ipynb"] = [
    ("markdown", "# 03 — Fish Species Classification (Phase 2)\n\n"
                 "ResNet34 · EfficientNet-B0 · MobileNetV3 · ConvNeXt-Tiny with class balancing, "
                 "underwater augmentation, mixup and cosine scheduling."),
    ("code", SETUP_COMMON),
    ("code", """\
# Single-model quick check, then the full sweep:
!python train.py --task fish --model resnet34 --set fish.epochs=3
"""),
    ("code", "!python train.py --task fish --model all"),
    ("code", """\
import pandas as pd, matplotlib.pyplot as plt
board = pd.read_csv("results/fish/model_leaderboard.csv")
display(board)
board.set_index("model")[["accuracy", "top5_accuracy", "macro_f1", "weighted_f1"]].plot.bar(figsize=(9, 4));
"""),
    ("code", """\
from IPython.display import Image as IPyImage
best = board.iloc[0]["model"]
IPyImage(f"results/fish/{best}/confusion_matrix.png", width=700)
"""),
]

NOTEBOOKS["04_coral_bleaching_analysis.ipynb"] = [
    ("markdown", "# 04 — Coral Bleaching Analysis (Phase 3)\n\n"
                 "Binary NOAA CNN + Whiteness-Guided Bleaching Severity (WGBS) → 0-100 index and "
                 "5-class condition (Healthy/Mild/Moderate/Severe/Dead)."),
    ("code", SETUP_COMMON),
    ("code", "!python train.py --task coral --model all"),
    ("code", """\
# WGBS on a few test patches
import glob, cv2, torch
from src.realtime.pipeline import load_best_classifier
from src.coral_analysis.severity import BleachingAssessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes, _, size = load_best_classifier("coral", "models", device)
assessor = BleachingAssessor()

for path in sorted(glob.glob("datasets/coral/imagefolders/test/*/*.jpg"))[:6]:
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    from torchvision import transforms as T
    t = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(
        Image.fromarray(rgb).resize((size, size)))[None].to(device)
    import torch.nn.functional as F
    with torch.no_grad():
        probs = F.softmax(model(t), 1)[0].cpu().numpy()
    p_bl = float(probs[classes.index("CORAL_BL")])
    verdict = assessor.assess(img, p_bl)
    print(f"{path.split('/')[-2]:>8} | P(bleached)={p_bl:.2f} | BSI={verdict.index:5.1f} | {verdict.label}")
"""),
]

NOTEBOOKS["05_fish_segmentation.ipynb"] = [
    ("markdown", "# 05 — Fish Segmentation for Detection Boxes (Phase 3.5)\n\n"
                 "Lightweight U-Net on DeepFish segmentation masks (~1.9 M params, CPU-friendly)."),
    ("code", SETUP_COMMON),
    ("code", "!python train.py --task seg"),
    ("code", """\
# Visualise predicted boxes on a DeepFish frame
import glob, cv2, matplotlib.pyplot as plt
from src.segmentation.unet import SmallUNet
from src.segmentation.postprocess import masks_to_boxes
import torch

ckpt = torch.load("models/seg/best.pt", map_location="cpu", weights_only=True)
model = SmallUNet(); model.load_state_dict(ckpt["model"]); model.eval()

img_path = sorted(glob.glob("datasets/deepfish/DeepFish/Segmentation/**/*.jpg", recursive=True))[0]
img = cv2.imread(img_path); rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
from src.segmentation.engine_seg import SegmentationTrainer
trainer = SegmentationTrainer(device=torch.device("cpu"))
prob = trainer.predict_mask(rgb)
boxes = masks_to_boxes(prob, threshold=0.5, min_area=400)
for b in boxes:
    cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off");
"""),
]

NOTEBOOKS["06_explainability_and_inference.ipynb"] = [
    ("markdown", "# 06 — Explainability + Real-time Inference (Phases 4-5)\n\n"
                 "GradCAM · GradCAM++ · ScoreCAM grids and the image/video pipeline."),
    ("code", SETUP_COMMON),
    ("code", """\
import glob
sample = sorted(glob.glob("datasets/fishnet/curated/validation/*/*.jpg"))
sample = sample[0] if sample else sorted(glob.glob("datasets/synthetic/enhance/*.jpg"))[0]
!python evaluate.py --xai --image {sample}
from IPython.display import Image as IPyImage
IPyImage("results/xai/xai_grid.png", width=1000)
"""),
    ("code", """\
# Single image through the full pipeline
!python inference.py --mode image --input {sample} --cam --report
"""),
    ("code", """\
# Video (upload your own survey clip to survey.mp4 first)
# !python inference.py --mode video --input survey.mp4 --output results/realtime
"""),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = OUT / name
        path.write_text(json.dumps(notebook([cell(k, s) for k, s in cells]), indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
