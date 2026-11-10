"""Build the fish classification ImageFolder from the fishnet.ai archive.

fishnet.ai (The Nature Conservancy) ships **bounding-box** annotations, not
taxonomy CSVs — verified schema of ``foid_labels_bbox_v012.csv``:

    img_id, bbox_id, x_min, x_max, y_min, y_max, label_l1, label_l2

* one row per fish/creature bbox; ``img_id`` is the image file stem
* ``label_l1`` = 28 fine-grained classes (incl. `Human`, `Unknown`, `No fish`)
* ``label_l2`` = 13 coarse codes (ALB, YFT, BET, LAG, DOL, OIL, SKJ, BILL, …)
* no split column → we derive a seeded, **image-level** train/val/test split
  (every crop of one image lands in the same split — no leakage)

Preparation = *crop every fish bbox out of its frame* → classification crops:

    curated/<split>/<CLASS>/<img_id>_<bbox_id>.jpg

`Human` / `Unknown` / `No fish` rows are excluded by default (they are not
fish species); classes with fewer than ``min_images_per_class`` boxes are
dropped; ``label`` selects ``l1`` (species-ish) or ``l2`` (coarse codes).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_CSV_CANDIDATES = ("foid_labels_bbox", "labels_bbox", "labels")
EXCLUDE_CLASSES_DEFAULT = ("Human", "Unknown", "No fish")
BBOX_MARGIN_FRACTION = 0.05      # widen each bbox by 5% per side for context
DEFAULT_SPLIT = (0.8, 0.1, 0.1)  # image-level train/val/test fractions


def _find_labels_csv(raw_root: Path) -> Path | None:
    candidates = [p for p in raw_root.rglob("*.csv")
                  if "__MACOSX" not in p.parts]
    for hint in LABEL_CSV_CANDIDATES:
        matches = [p for p in candidates if hint in p.stem.lower()]
        if matches:
            # Prefer the non-"freq" variant (per-bbox rows, not frequency table).
            non_freq = [p for p in matches if "freq" not in p.stem.lower()]
            return (non_freq or matches)[0]
    return candidates[0] if candidates else None


def _index_images(raw_root: Path) -> dict[str, Path]:
    """Map image file stem (== img_id) -> path, anywhere under the archive."""
    index: dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png"}
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts and "__MACOSX" not in path.parts:
            index.setdefault(path.stem, path)
    logger.info("Indexed %d images under %s", len(index), raw_root)
    return index


def _image_level_split(img_ids: list[str], fractions: tuple[float, float, float],
                       seed: int = 42) -> dict[str, str]:
    """Seeded split at the IMAGE level: all crops of one image share a split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(img_ids))
    n_train = int(len(img_ids) * fractions[0])
    n_val = int(len(img_ids) * fractions[1])
    assignment: dict[str, str] = {}
    for rank, idx in enumerate(order):
        split = "train" if rank < n_train else ("validation" if rank < n_train + n_val else "test")
        assignment[img_ids[idx]] = split
    return assignment


def _crop_and_save_from(img: "np.ndarray", box: tuple[int, int, int, int],
                        dest: Path, margin: float = BBOX_MARGIN_FRACTION) -> bool:
    """Crop one bbox (with context margin) from an already-decoded frame."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return False  # degenerate crop — cleaned later by min-size scan anyway
    crop = img[y1:y2, x1:x2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def prepare_fishnet_subset(
    raw_dir: str | Path,
    out_dir: str | Path,
    level: str = "l1",
    top_k: int = 12,
    min_images_per_class: int = 100,
    exclude_classes: list[str] | tuple[str, ...] = EXCLUDE_CLASSES_DEFAULT,
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = 42,
    copy_images: bool = True,  # unused; kept for CLI compatibility
) -> Path:
    """Crop fish bboxes from the fishnet.ai archive into a classification ImageFolder."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    done_marker = out_dir / "_COMPLETE"
    if done_marker.exists():
        # NOTE: the marker records completion, not the config that produced it —
        # after changing fish.prepare.* (level/top_k/min_images) delete the
        # marker (or the curated dir) to rebuild with the new settings.
        logger.info("Curated fish subset already complete at %s — skipping.", out_dir)
        return out_dir
    # The crop loop itself is idempotent (per-crop dest.exists() skip), so an
    # interrupted run simply continues where it stopped — no partial-data trap.

    labels_csv = _find_labels_csv(raw_dir)
    if labels_csv is None:
        raise FileNotFoundError(
            f"No labels CSV (foid_labels_bbox*) found under {raw_dir}. "
            "Run `python download_data.py --dataset fishnet` first."
        )
    logger.info("Using labels CSV: %s", labels_csv)
    df = pd.read_csv(labels_csv)
    required = {"img_id", "label_l1", "label_l2", "x_min", "x_max", "y_min", "y_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Unexpected fishnet.ai CSV schema in {labels_csv}: missing {sorted(missing)}. "
            f"Actual columns: {list(df.columns)}. The archive layout may have changed — "
            "inspect the labels zip and update prepare_fishnet."
        )

    label_col = "label_l1" if level == "l1" else "label_l2"
    exclude = set(exclude_classes)
    if label_col == "label_l2":
        exclude |= {"HUMAN", "OTH", "NoF"}

    counts = df[~df[label_col].isin(exclude)][label_col].value_counts()
    keep = set(counts[counts >= min_images_per_class].sort_values(ascending=False).head(top_k).index)
    logger.info("Keeping %d '%s' classes (>= %d boxes): %s",
                len(keep), label_col, min_images_per_class, sorted(keep))

    work = df[df[label_col].isin(keep)].copy()
    coord_cols = ["x_min", "x_max", "y_min", "y_max"]
    for col in coord_cols + ["bbox_id"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    dropped_rows = int(work[coord_cols].isna().any(axis=1).sum())
    if dropped_rows:
        logger.warning("Dropping %d malformed bbox rows (non-numeric coordinates).", dropped_rows)
        work = work.dropna(subset=coord_cols + ["bbox_id"])
    img_index = _index_images(raw_dir)
    missing_images = sum(1 for iid in work["img_id"].unique() if str(iid) not in img_index)
    if missing_images:
        logger.warning("%d labelled images not found in the archive (may belong to other releases).",
                       missing_images)

    # Split by image, then materialise crops.
    present_ids = [str(i) for i in work["img_id"].unique() if str(i) in img_index]
    assignment = _image_level_split(present_ids, split_fractions, seed=seed)

    per_split_class = defaultdict(int)
    saved, skipped = 0, 0
    frame_cache: dict[str, "np.ndarray | None"] = {}

    def _frame(img_path: Path, img_id: str):
        """Decode each frame once (rows of one img_id share the decode)."""
        if img_id not in frame_cache:
            frame_cache[img_id] = cv2.imread(str(img_path))
        return frame_cache[img_id]

    for row in work.itertuples(index=False):
        img_id = str(row.img_id)
        img_path = img_index.get(img_id)
        if img_path is None:
            skipped += 1
            continue
        class_name = str(getattr(row, label_col)).replace(" ", "_").replace("/", "-")
        split = assignment[img_id]
        dest = out_dir / split / class_name / f"{img_id}_bbox{int(row.bbox_id)}.jpg"
        if dest.exists():
            saved += 1
            continue
        img = _frame(img_path, img_id)
        if img is None:
            skipped += 1
            continue
        box = (int(row.x_min), int(row.y_min), int(row.x_max), int(row.y_max))
        if _crop_and_save_from(img, box, dest):
            saved += 1
            per_split_class[(split, class_name)] += 1
        else:
            skipped += 1
        # keep the cache bounded (Albacore schools repeat frames rarely)
        if len(frame_cache) > 64:
            frame_cache.clear()

    logger.info("Cropped %d fish regions (%d skipped: missing frames/degenerate boxes).", saved, skipped)
    for split in ("train", "validation", "test"):
        total = sum(v for (s, _), v in per_split_class.items() if s == split)
        classes = len({c for (s, c), v in per_split_class.items() if s == split and v > 0})
        logger.info("Split '%s': %d crops across %d classes.", split, total, classes)
    done_marker.write_text("ok\n", encoding="utf-8")
    return out_dir
