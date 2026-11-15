"""NOAA PIFSC ESD Coral Bleaching dataset downloader.

Repo: https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset
Binary labels: CORAL (healthy hard coral) vs CORAL_BL (bleached hard coral).
Splits: train 7,292 / validation 1,562 / test 1,565 — 224x224 patches.

**Verified release layout (2026-11-15, HF tree API):** the repo is an
ImageFolder dataset — labels live in the **directory names** of the repo tree:

    train/CORAL/*.PNG   train/CORAL_BL/*.PNG
    val/CORAL/*.PNG     val/CORAL_BL/*.PNG
    test/CORAL/*.PNG    test/CORAL_BL/*.PNG

The datasets-server parquet conversion carries **no label column** (only
`image`) — which is what broke the original `load_dataset`-based downloader
(KeyError: 'label'). This downloader therefore bypasses conversions entirely:
it `snapshot_download`s the ImageFolder split/class directories and
materialises the training ImageFolder from the authoritative directory labels:

    datasets/coral/imagefolders/{train,validation,test}/{CORAL,CORAL_BL}/*.PNG
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CORAL_REPO = "NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
CLASS_NAMES = ("CORAL", "CORAL_BL")
# Verified split sizes (datasets-server + repo tree, 2026-11-15) — used as a
# download-time drift alarm (WARN only: upstream may legitimately grow).
EXPECTED_CORAL_SPLITS = {"train": 7292, "val": 1562, "test": 1565}
IMG_EXTS = {".png", ".jpg", ".jpeg"}
SNAPSHOT_PATTERNS = ["train/**", "val/**", "test/**"]


def _imagefolder_ready(root: Path, min_images: int = 100) -> bool:
    # A _COMPLETE marker is written after every split is materialised, so an
    # interrupted download is never mistaken for a ready dataset. Datasets
    # created before markers existed are healed in place.
    if not root.exists():
        return False
    total = sum(1 for _ in root.rglob("*.jpg")) + sum(1 for _ in root.rglob("*.png"))
    if total < min_images:
        return False
    # Healing a pre-marker dataset requires ALL splits populated - an
    # interrupted write that reached min_images total must NOT self-heal.
    splits = {
        d.name for d in root.iterdir() if d.is_dir()
        and (any(d.rglob("*.jpg")) or any(d.rglob("*.png")))
    }
    if not {"train", "test"} <= splits or not (splits & {"validation", "val"}):
        return False
    if not (root.parent / "_COMPLETE").exists():
        (root.parent / "_COMPLETE").write_text("ok\n", encoding="utf-8")
    return True


def _materialise_imagefolder(snapshot_dir: Path, root: Path) -> dict[str, int]:
    """Copy the snapshot's split/class dirs into the training ImageFolder layout.

    Repo split names: train / val / test. Training contract: train / validation / test.
    """
    import shutil

    rename = {"train": "train", "val": "validation", "test": "test"}
    counts: dict[str, int] = {}
    for src_split, dst_split in rename.items():
        for cls in CLASS_NAMES:
            src = snapshot_dir / src_split / cls
            if not src.is_dir():
                continue
            dst = root / dst_split / cls
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted(src.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                target = dst / f.name
                if not target.exists():
                    shutil.copy2(f, target)
                n += 1
            counts[f"{dst_split}/{cls}"] = counts.get(f"{dst_split}/{cls}", 0) + n
    return counts


def download_coral(data_dir: str | Path, force: bool = False) -> Path:
    """Materialise the NOAA coral dataset as local ImageFolder splits."""
    data_dir = Path(data_dir)
    root = data_dir / "imagefolders"
    root.mkdir(parents=True, exist_ok=True)

    if not force and _imagefolder_ready(root):
        logger.info("Coral ImageFolders already present at %s — skipping.", root)
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading NOAA coral ImageFolder splits via snapshot_download ...")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=CORAL_REPO,
            repo_type="dataset",
            allow_patterns=SNAPSHOT_PATTERNS,
        )
    )

    counts = _materialise_imagefolder(snapshot_dir, root)
    total = sum(counts.values())
    logger.info("Materialised: %s", counts)

    # Download-time drift alarm: compare against the documented split sizes.
    got = {"train": counts.get("train/CORAL", 0) + counts.get("train/CORAL_BL", 0),
           "val": counts.get("validation/CORAL", 0) + counts.get("validation/CORAL_BL", 0),
           "test": counts.get("test/CORAL", 0) + counts.get("test/CORAL_BL", 0)}
    for split, expected in EXPECTED_CORAL_SPLITS.items():
        if got[split] and abs(got[split] - expected) > expected * 0.05:
            logger.warning("Split '%s' has %d images (documented: %d) — upstream may have changed.",
                           split, got[split], expected)

    if not _imagefolder_ready(root):
        raise RuntimeError(
            f"Coral ImageFolder incomplete at {root} — got {counts}. "
            "Check the snapshot download and re-run."
        )

    (root.parent / "_COMPLETE").write_text("ok\n", encoding="utf-8")
    n = sum(1 for _ in root.rglob("*.jpg")) + sum(1 for _ in root.rglob("*.png"))
    logger.info("Coral dataset ready: %d images under %s", n, root)
    return root
