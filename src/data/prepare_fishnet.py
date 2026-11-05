"""Build a curated FishNet ImageFolder subset for classification training.

FishNet ships 94,532 images with labels at four taxonomy levels
(order / family / genus / species). Training 17k species is infeasible on free
compute, so this script builds a *curated* subset:

    curated/<split>/<CLASS>/<image>.jpg

Selection rules
---------------
- ``level``   : which taxonomy column to use as the class label.
- ``min_images_per_class`` : drop rarer classes.
- ``top_k``   : keep the K largest classes after filtering.

The extractor is schema-adaptive: FishNet archives ship annotation CSVs
(train/val/test) whose column names may differ slightly; we match them
heuristically. If the archive already contains a plain ImageFolder tree, it is
used as-is.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LEVEL_ALIASES = {
    "order": ("order", "label_order", "Order"),
    "family": ("family", "label_family", "Family"),
    "genus": ("genus", "label_genus", "Genus"),
    "species": ("species", "label_species", "Species", "label", "class"),
}
SPLIT_KEYS = {"train": ("train", "training"), "validation": ("val", "validation", "valid"), "test": ("test",)}


def _find_annotation_csvs(raw_root: Path) -> dict[str, Path]:
    """Locate train/val/test CSVs anywhere under the extracted archive."""
    found: dict[str, Path] = {}
    for csv_path in raw_root.rglob("*.csv"):
        stem = csv_path.stem.lower()
        for split, aliases in SPLIT_KEYS.items():
            if split in found:
                continue
            if any(stem.startswith(a) or a in stem for a in aliases):
                found[split] = csv_path
    return found


def _pick_label_column(df: pd.DataFrame, level: str) -> str | None:
    aliases = LEVEL_ALIASES.get(level, (level,))
    for col in df.columns:
        if col.lower() in {a.lower() for a in aliases}:
            return col
    return None


def _pick_image_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(token in col.lower() for token in ("path", "file", "img", "image", "name")):
            return col
    return None


def _existing_image(img_root: Path, rel: str) -> Path | None:
    candidates = [img_root / rel, img_root / rel.replace("\\", "/")]
    if candidates[0].is_file():
        return candidates[0]
    hits = list(img_root.rglob(Path(rel).name)) if rel else []
    return hits[0] if hits else None


def prepare_fishnet_subset(
    raw_dir: str | Path,
    out_dir: str | Path,
    level: str = "family",
    top_k: int = 25,
    min_images_per_class: int = 60,
    copy_images: bool = True,
) -> Path:
    """Create the curated ImageFolder subset. Returns ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.rglob("*.jpg")) or any(out_dir.rglob("*.png")):
        logger.info("Curated FishNet subset already present at %s — skipping.", out_dir)
        return out_dir

    csvs = _find_annotation_csvs(raw_dir)
    if not csvs:
        # Maybe the archive is already an ImageFolder tree.
        class_dirs = [d for d in raw_dir.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]
        if class_dirs and any(d.parent != d for d in class_dirs):
            logger.info("FishNet archive looks like an ImageFolder tree; symlinking as-is.")
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            if not out_dir.exists():
                try:
                    out_dir.symlink_to(raw_dir.resolve(), target_is_directory=True)
                except OSError:
                    shutil.copytree(raw_dir, out_dir, dirs_exist_ok=True)
            return out_dir
        raise FileNotFoundError(
            f"No annotation CSVs or class folders found under {raw_dir}. "
            "See docs/DATASETS.md for the expected FishNet layout."
        )

    logger.info("Found annotation CSVs: %s", {k: str(v) for k, v in csvs.items()})
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count class frequencies at the requested level (train split).
    train_csv = csvs.get("train") or next(iter(csvs.values()))
    df_train = pd.read_csv(train_csv)
    label_col = _pick_label_column(df_train, level) or _pick_label_column(df_train, "species")
    if label_col is None:
        raise ValueError(f"No taxonomy column for level '{level}' in {train_csv} "
                         f"(columns: {list(df_train.columns)})")
    counts = df_train[label_col].value_counts()
    keep = counts[counts >= min_images_per_class].sort_values(ascending=False).head(top_k)
    keep_classes = set(keep.index.astype(str))
    logger.info("Keeping %d '%s' classes: %s", len(keep_classes), level, sorted(keep_classes))

    for split, csv_path in csvs.items():
        df = pd.read_csv(csv_path)
        img_col = _pick_image_column(df)
        if img_col is None:
            logger.warning("No image-path column in %s; skipping split.", csv_path)
            continue
        kept = df[df[label_col].astype(str).isin(keep_classes)]
        copied = 0
        for _, row in kept.iterrows():
            src = _existing_image(raw_dir, str(row[img_col]))
            if src is None:
                continue
            class_name = str(row[label_col]).replace(" ", "_").replace("/", "-")
            dest_dir = out_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():
                if copy_images:
                    shutil.copy2(src, dest)
                else:
                    try:
                        dest.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, dest)
            copied += 1
        logger.info("Split '%s': %d images curated.", split, copied)

    return out_dir
