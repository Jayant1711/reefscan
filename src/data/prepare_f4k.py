"""Build the fish species classification ImageFolder from the Fish4Knowledge (F4K) release.

Verified release layout (fishRecognition_GT.tar, 487 MB):

    fishRecognition_GT/fish_image/fish_01..fish_23/*.png   27,394 fish crops
    fishRecognition_GT/mask_image/mask_01..mask_23/*.png   matching masks (kept in raw)
    class_id.csv (separate download)                        official annotation

Species labels are carried by the tar **directory structure**: fish_NN == species
cluster N (verified count-for-count against the official class_id.csv: 27,370
rows, 23 clusters). The release provides no human-readable species names for the
clusters, so classes keep their cluster ids.

Curated output:

    curated/<split>/<species_NN>/<src name>.png

* Seeded **group-level** split: every crop from the same video segment (the
  leading `fish_<video-id>` prefix of the file stem) shares one split — crops of
  one video are near-duplicates, so this prevents scene-level leakage.
* Classes with fewer than ``min_images_per_class`` images are dropped
  (13 of 23 clusters have >= 100 images; 26,903 images total at the default).
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SPLIT = (0.8, 0.1, 0.1)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _video_group(image_path: Path) -> str:
    """Group key for the seeded split: the video/segment prefix of the stem.

    F4K stems look like ``fish_000000009598_05281`` — video/segment
    ``fish_000000009598`` + fish index. Crops from one video segment are
    near-duplicates, so they must share a split.
    """
    stem = image_path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2])  # e.g. fish_000000009598
    return stem


def _grouped_split(groups: list[str], fractions: tuple[float, float, float],
                   seed: int = 42) -> dict[str, str]:
    import random

    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    n_train = int(len(order) * fractions[0])
    n_val = int(len(order) * fractions[1])
    assignment: dict[str, str] = {}
    for rank, group in enumerate(order):
        split = "train" if rank < n_train else ("validation" if rank < n_train + n_val else "test")
        assignment[group] = split
    return assignment


def prepare_f4k_subset(
    raw_dir: str | Path,
    out_dir: str | Path,
    min_images_per_class: int = 100,
    top_k: int = 13,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = 42,
    copy_images: bool = True,  # unused; kept for CLI compatibility
) -> Path:
    """Build the curated F4K species ImageFolder. Returns ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    done_marker = out_dir / "_COMPLETE"
    if done_marker.exists():
        # NOTE: the marker records completion, not the config that produced it —
        # after changing prepare knobs, delete the marker to rebuild.
        logger.info("Curated F4K subset already complete at %s — skipping.", out_dir)
        return out_dir

    fish_root = raw_dir / "fishRecognition_GT" / "fish_image"
    if not fish_root.is_dir():
        # The downloader may also have extracted flat — accept either layout.
        fish_root = raw_dir / "fish_image"
    if not fish_root.is_dir():
        raise FileNotFoundError(
            f"F4K fish_image tree not found under {raw_dir}. "
            "Run `python download_data.py --dataset f4k` first."
        )

    # Classes = the fish_NN directories (verified == class_id.csv clusters).
    class_counts: dict[str, int] = {}
    for class_dir in sorted(d for d in fish_root.iterdir() if d.is_dir()):
        n = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)
        if n:
            class_counts[class_dir.name] = n
    keep = {
        c for c, n in class_counts.items()
        if n >= min_images_per_class
    }
    keep = set(sorted(keep, key=lambda c: -class_counts[c])[:top_k])
    logger.info("Keeping %d F4K classes (>= %d images): %s",
                len(keep), min_images_per_class, sorted(keep))
    if not keep:
        raise ValueError(
            f"No F4K class has >= {min_images_per_class} images under {fish_root} — "
            "check the extraction, or lower fish.prepare.min_images_per_class."
        )

    # Collect (class, image path) for kept classes + group by video segment.
    items: list[tuple[str, Path, str]] = []  # (class, src, group)
    for cls in sorted(keep):
        for src in sorted((fish_root / cls).rglob("*")):
            if src.suffix.lower() in IMG_EXTS:
                items.append((cls, src, _video_group(src)))
    if not items:
        raise ValueError(f"No images found for the kept classes under {fish_root}")

    groups = sorted({g for _, _, g in items})
    assignment = _grouped_split(groups, split_fractions, seed=seed)

    per_split: dict[str, int] = defaultdict(int)
    saved = 0
    for cls, src, group in items:
        split = assignment[group]
        dest_dir = out_dir / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            saved += 1
            per_split[split] += 1
            continue
        shutil.copy2(src, dest)
        saved += 1
        per_split[split] += 1

    for split in ("train", "validation", "test"):
        logger.info("Split '%s': %d images across %d classes.",
                    split,
                    sum(1 for p in (out_dir / split).rglob("*") if p.is_file()),
                    len({p.parent.name for p in (out_dir / split).rglob("*") if p.is_file()}))
    done_marker.write_text("ok\n", encoding="utf-8")
    logger.info("Curated F4K subset -> %s (%d images).", out_dir, saved)
    return out_dir
