"""Dataset cleaning: corrupt scan, duplicate removal, split-leakage audit.

Nothing is deleted — flagged files are *moved* into ``<dataset>/_quarantine/``
only when ``apply=True``. Reports are always written so runs stay auditable
(and the recipe is recorded in ``my_requested_thing.md``).

Per-dataset recipes
-------------------
- Coral / FishNet ImageFolders: corrupt decode scan, min-size filter,
  exact-duplicate (MD5) + near-duplicate (8x8 dHash) detection, and a
  train/val/test dHash-overlap leakage check.
- DeepFish segmentation: image/mask pairing validation (reuses
  ``discover_pairs``), corrupt scan, min mask-area (foreground fraction) filter.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..utils.image_io import SUPPORTED_IMAGE_EXTS

logger = logging.getLogger(__name__)

MIN_IMAGE_SIZE = 32          # px per side
MIN_MASK_FRACTION = 0.01     # foreground fraction for segmentation masks
QUARANTINE_DIR = "_quarantine"


# ------------------------------------------------------------------ hashing
def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while data := fh.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def dhash_of(path: Path, hash_size: int = 8) -> str | None:
    """8x8 difference hash (dHash) — robust to recompression, catches near-dupes."""
    try:
        with Image.open(path) as im:
            gray = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.int16)
        diff = arr[:, 1:] > arr[:, :-1]          # horizontal gradient bits
        bits = diff.flatten()
        return "".join("1" if b else "0" for b in bits)
    except Exception:
        return None


# ------------------------------------------------------- imagefolder scans
def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
                  and QUARANTINE_DIR not in p.parts)


def _split_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def scan_imagefolder(root: str | Path, min_size: int = MIN_IMAGE_SIZE) -> dict:
    """Scan an ImageFolder tree: corrupt, too-small, duplicates, split leakage."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    report: dict = {
        "root": str(root),
        "scanned": 0,
        "per_split": {},
        "corrupt": [],
        "too_small": [],
        "duplicate_groups": [],      # [{"kind": "md5"|"dhash", "hash": ..., "paths": [...]}]
        "leakage_groups": [],        # same hash present in 2+ splits
        "counts": {},
    }
    md5_index: dict[str, list[Path]] = {}
    dhash_index: dict[str, list[Path]] = {}

    images = _iter_images(root)
    report["scanned"] = len(images)
    for path in images:
        split = _split_of(path, root)
        split_stats = report["per_split"].setdefault(split, {"scanned": 0, "corrupt": 0, "too_small": 0})
        split_stats["scanned"] += 1

        # Corrupt / too-small detection via a strict decode.
        try:
            with Image.open(path) as im:
                im.verify()  # structural check
            with Image.open(path) as im:
                width, height = im.size
            if width < min_size or height < min_size:
                report["too_small"].append(str(path))
                split_stats["too_small"] += 1
                continue
        except Exception:
            report["corrupt"].append(str(path))
            split_stats["corrupt"] += 1
            continue

        md5_index.setdefault(md5_of(path), []).append(path)
        dhash = dhash_of(path)
        if dhash:
            dhash_index.setdefault(dhash, []).append(path)

    split_rank = {"train": 0, "validation": 1, "val": 1, "test": 2}

    def _ordered(paths: list[Path]) -> list[Path]:
        """train copy first, then val, then test — quarantine never eats train data."""
        return sorted(paths, key=lambda p: split_rank.get(_split_of(p, root), 3))

    report["duplicate_groups"].extend(
        {"kind": "md5", "hash": h, "paths": [str(p) for p in _ordered(paths)]}
        for h, paths in md5_index.items() if len(paths) > 1
    )
    # Near-dup scan excludes files already grouped as exact duplicates, so one
    # redundant file is counted once, not twice.
    md5_dupes = {p for paths in md5_index.values() if len(paths) > 1 for p in paths}
    report["duplicate_groups"].extend(
        {"kind": "dhash", "hash": h, "paths": [str(p) for p in _ordered(paths)]}
        for h, paths in dhash_index.items() if len(paths) > 1
        and not set(paths) <= md5_dupes
    )

    # Leakage: identical dHash present in 2+ splits.
    hash_splits: dict[str, set[str]] = {}
    for h, paths in dhash_index.items():
        splits = {_split_of(p, root) for p in paths}
        if len(splits) > 1:
            hash_splits[h] = splits
    report["leakage_groups"] = [
        {"hash": h, "splits": sorted(s),
         "paths": [str(p) for p in _ordered(dhash_index[h])]}
        for h, s in hash_splits.items()
    ]

    report["counts"] = {
        "scanned": report["scanned"],
        "corrupt": len(report["corrupt"]),
        "too_small": len(report["too_small"]),
        "duplicate_images": len({p for g in report["duplicate_groups"] for p in g["paths"][1:]}),
        "leaked_images": sum(len(g["paths"]) - 1 for g in report["leakage_groups"]),
    }
    logger.info("Scan %s: %s", root.name, report["counts"])
    return report


# ------------------------------------------------------------ segmentation
def scan_segmentation_pairs(data_dir: str | Path, min_fraction: float = MIN_MASK_FRACTION) -> dict:
    """Validate DeepFish-style (image, mask) pairs: corrupt files + mask area."""
    from ..segmentation.engine_seg import discover_pairs

    data_dir = Path(data_dir)
    pairs = discover_pairs(data_dir)
    report: dict = {
        "root": str(data_dir),
        "pairs_found": len(pairs),
        "corrupt_images": [],
        "corrupt_masks": [],
        "empty_masks": [],       # mask file exists but ~no foreground
        "too_small_masks": [],   # foreground fraction < min_fraction
        "counts": {},
    }
    for img_path, mask_path in pairs:
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            report["corrupt_images"].append(str(img_path))
            continue
        try:
            with Image.open(mask_path) as im:
                mask = np.asarray(im.convert("L"))
        except Exception:
            report["corrupt_masks"].append(str(mask_path))
            continue
        fraction = float((mask > 127).mean())
        if fraction == 0.0:
            report["empty_masks"].append(str(mask_path))
        elif fraction < min_fraction:
            report["too_small_masks"].append(
                f"{mask_path} (foreground={fraction:.4f})"
            )

    report["counts"] = {
        "pairs_found": report["pairs_found"],
        "corrupt_images": len(report["corrupt_images"]),
        "corrupt_masks": len(report["corrupt_masks"]),
        "empty_masks": len(report["empty_masks"]),
        "too_small_masks": len(report["too_small_masks"]),
        "flagged_pairs": (len(report["corrupt_images"]) + len(report["corrupt_masks"])
                          + len(report["empty_masks"]) + len(report["too_small_masks"])),
    }
    logger.info("Seg scan %s: %s", data_dir.name, report["counts"])
    return report


# ---------------------------------------------------------------- quarantine
def apply_quarantine(root: str | Path, report: dict) -> list[str]:
    """Move flagged files into ``<root>/_quarantine/<reason>/`` (non-destructive)."""
    root = Path(root)
    moved: list[str] = []
    for group in report.get("duplicate_groups", []):
        for p in group["paths"][1:]:
            reason = f"duplicate_{group['kind']}"
            moved.append(_move_to_quarantine(root, Path(p), reason))
    for group in report.get("leakage_groups", []):
        for p in group["paths"][1:]:
            moved.append(_move_to_quarantine(root, Path(p), "leakage"))
    reasons = {
        "corrupt": report.get("corrupt", []) + report.get("corrupt_images", []) + report.get("corrupt_masks", []),
        "too_small": report.get("too_small", []),
        "empty_mask": report.get("empty_masks", []),
    }
    for reason, items in reasons.items():
        for item in items:
            moved.append(_move_to_quarantine(root, Path(item.split(" (foreground")[0]), reason))
    moved = [m for m in moved if m]
    logger.info("Quarantined %d files under %s/%s", len(moved), root, QUARANTINE_DIR)
    return moved


def _move_to_quarantine(root: Path, path: Path, reason: str) -> str:
    if not path.exists() or QUARANTINE_DIR in path.parts:
        return ""
    target_dir = root / QUARANTINE_DIR / reason
    target_dir.mkdir(parents=True, exist_ok=True)
    # Flatten but keep uniqueness: split dirs in the filename.
    safe_name = "_".join(path.relative_to(root).parts)
    target = target_dir / safe_name
    counter = 2
    while target.exists():
        target = target_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    shutil.move(str(path), str(target))
    return str(target)


def clean_imagefolder(root: str | Path, apply: bool = False, report_path: Path | None = None) -> dict:
    """Scan (and optionally quarantine) an ImageFolder dataset; write the report."""
    report = scan_imagefolder(root)
    if apply:
        report["quarantined"] = apply_quarantine(root, report)
        report["counts"]["quarantined"] = len(report["quarantined"])
    if report_path:
        from ..utils import save_json

        save_json(report, report_path)
    return report


def clean_segmentation(data_dir: str | Path, apply: bool = False,
                       report_path: Path | None = None,
                       min_fraction: float = MIN_MASK_FRACTION) -> dict:
    """Scan (and optionally quarantine) image/mask pairs; write the report."""
    report = scan_segmentation_pairs(data_dir, min_fraction=min_fraction)
    if apply:
        root = Path(data_dir)
        moved: list[str] = []
        for key in ("corrupt_images", "corrupt_masks", "empty_masks"):
            for item in report.get(key, []):
                moved.append(_move_to_quarantine(root, Path(item), key))
        for item in report.get("too_small_masks", []):
            moved.append(_move_to_quarantine(root, Path(item.split(" (foreground")[0]), "too_small_mask"))
        report["quarantined"] = [m for m in moved if m]
        report["counts"]["quarantined"] = len(report["quarantined"])
    if report_path:
        from ..utils import save_json

        save_json(report, report_path)
    return report
