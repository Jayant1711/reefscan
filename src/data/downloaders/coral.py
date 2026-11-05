"""NOAA PIFSC ESD Coral Bleaching dataset downloader.

Repo: https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset
Binary labels: CORAL (healthy hard coral) vs CORAL_BL (bleached hard coral).
Splits: train 7,292 / validation 1,562 / test 1,565 — 224x224 patches.

Two access paths:
  1. ``datasets`` library (preferred; auto-converts to parquet).
  2. Direct parquet download via huggingface_hub + pyarrow (fallback).
Both materialise a local ImageFolder tree used by the training pipeline:

    datasets/coral/imagefolders/{train,validation,test}/{CORAL,CORAL_BL}/*.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CORAL_REPO = "NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
CLASS_NAMES = ["CORAL", "CORAL_BL"]
SPLIT_DIRS = {"train": "train", "validation": "validation", "test": "test"}


def _imagefolder_ready(root: Path, min_images: int = 100) -> bool:
    if not root.exists():
        return False
    total = sum(1 for _ in root.rglob("*.jpg")) + sum(1 for _ in root.rglob("*.png"))
    return total >= min_images


def _save_via_datasets(root: Path) -> Path:
    from datasets import load_dataset

    ds = load_dataset(CORAL_REPO)
    label_names = ds["train"].features["label"].names if "label" in ds["train"].features else CLASS_NAMES
    logger.info("HF coral label names: %s", label_names)

    import cv2
    import numpy as np

    for split, hf_split in SPLIT_DIRS.items():
        for idx, example in enumerate(ds[split]):
            img = np.array(example["image"].convert("RGB"))
            label = int(example["label"])
            out_dir = root / split / str(label_names[label])
            out_dir.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{split}_{idx:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return root


def _save_via_parquet(root: Path) -> Path:
    from huggingface_hub import snapshot_download
    import pandas as pd
    import pyarrow.parquet as pq

    local = Path(
        snapshot_download(repo_id=CORAL_REPO, repo_type="dataset", allow_patterns=["**/*.parquet"])
    )
    parquets = sorted(local.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet shards found in {local}")

    import cv2
    import numpy as np

    # Discover class names: ClassLabel features embedded in parquet metadata.
    names: list[str] | None = None
    counter: dict[str, int] = {}
    for pq_path in parquets:
        split = pq_path.stem.split("-")[0]
        split = SPLIT_DIRS.get(split, split)
        table = pq.read_table(pq_path, columns=["image", "label"]) if "image" in pq.read_schema(pq_path).names else None
        if table is None:
            continue
        if names is None:
            feat = pq.read_schema(pq_path).field("label").type
            hf_feat = None
            try:  # ClassLabel exposes names via the HF extension type metadata
                hf_feat = pq.read_schema(pq_path).field("label")
                names = list(hf_feat.metadata["hextras"]["names"].values()) if hf_feat.metadata and "hextras" in str(hf_feat.metadata) else None
            except Exception:
                names = None
            if names is None:
                names = CLASS_NAMES  # ClassLabel stores names alphabetically
            logger.info("Coral class names resolved to: %s", names)

        df = table.to_pandas()
        for idx, row in df.iterrows():
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            label = int(row["label"])
            out_dir = root / split / str(names[label])
            out_dir.mkdir(parents=True, exist_ok=True)
            arr = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            fname = f"{split}_{counter.get(split, 0):06d}.jpg"
            counter[split] = counter.get(split, 0) + 1
            cv2.imwrite(str(out_dir / fname), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return root


def download_coral(data_dir: str | Path, force: bool = False) -> Path:
    """Materialise the NOAA coral dataset as local ImageFolder splits."""
    data_dir = Path(data_dir)
    root = data_dir / "imagefolders"
    root.mkdir(parents=True, exist_ok=True)

    if not force and _imagefolder_ready(root):
        logger.info("Coral ImageFolders already present at %s — skipping.", root)
        return root

    try:
        logger.info("Loading NOAA coral dataset via `datasets` ...")
        _save_via_datasets(root)
    except ImportError:
        logger.info("`datasets` unavailable — falling back to parquet + pyarrow ...")
        _save_via_parquet(root)

    n = sum(1 for _ in root.rglob("*.jpg"))
    logger.info("Coral dataset ready: %d images under %s", n, root)
    return root
