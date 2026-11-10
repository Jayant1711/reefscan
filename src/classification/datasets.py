"""Dataset & dataloader construction for ImageFolder-style classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tvd

from .class_balancing import count_class_images, make_weighted_sampler
from .transforms import build_eval_transform, build_train_transform

logger = logging.getLogger(__name__)


@dataclass
class LoaderBundle:
    """Train/val/test loaders plus class metadata."""

    train: DataLoader | None
    val: DataLoader | None
    test: DataLoader | None
    class_names: list[str]
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def count_class_images_public(root: str | Path) -> dict[str, int]:
    """Re-exported helper (public alias)."""
    return count_class_images(root)


def _split_dir(data_dir: Path, split: str) -> Path | None:
    candidates = [data_dir / split, data_dir / {"val": "validation", "validation": "val"}[split] if split in {"val", "validation"} else data_dir / split]
    for cand in candidates:
        if cand.exists() and any(p.is_dir() for p in cand.iterdir()):
            return cand
    return None


def build_dataloaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    balance_train: bool = True,
    train_cfg: Mapping | None = None,
) -> LoaderBundle:
    """Build train/val/test loaders from an ImageFolder tree.

    Expected layout (built by prepare_fishnet.py / coral downloader)::

        data_dir/train/<class>/*.jpg
        data_dir/validation/<class>/*.jpg      (or val/)
        data_dir/test/<class>/*.jpg            (optional)

    With ``balance_train`` the train loader uses a WeightedRandomSampler so
    rare classes are seen equally often per epoch.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset root '{data_dir}' not found. Run download_data.py first "
            "or point the config at an existing ImageFolder tree."
        )

    train_dir = _split_dir(data_dir, "train")
    val_dir = _split_dir(data_dir, "val") or _split_dir(data_dir, "validation")
    test_dir = _split_dir(data_dir, "test")

    if train_dir is None:
        # Single-folder dataset: split by directory if val exists, else error.
        raise FileNotFoundError(
            f"'{data_dir}' must contain train/ and validation/ subfolders with class directories."
        )

    # Class metadata comes from the union of splits (sorted for determinism).
    class_names = sorted({d.name for split in (train_dir, val_dir, test_dir) if split for d in split.iterdir() if d.is_dir()})
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_tf = build_train_transform(train_cfg or {}, image_size)
    eval_tf = build_eval_transform(image_size)

    def _loader(split_dir: Path | None, train: bool, counts_root: Path | None) -> DataLoader | None:
        if split_dir is None:
            return None
        # Only keep classes present in the shared class list.
        ds = tvd.ImageFolder(str(split_dir), transform=train_tf if train else eval_tf)
        # Remap to shared class indices if a class is missing in this split.
        if ds.classes != class_names:
            ds.class_to_idx = class_to_idx
            remap = {old_idx: class_to_idx[name] for name, old_idx in
                     {c: i for i, c in enumerate(ds.classes)}.items()}
            ds.targets = [remap[t] for t in ds.targets]
            ds.samples = [(path, remap[label]) for (path, label) in ds.samples]
            ds.classes = class_names
        sampler = (make_weighted_sampler(ds.targets, count_class_images(split_dir),
                                         class_names=class_names)
                   if (train and balance_train) else None)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(train and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    logger.info("Dataset: %d classes from %s", len(class_names), data_dir)
    return LoaderBundle(
        train=_loader(train_dir, True, train_dir),
        val=_loader(val_dir, False, val_dir),
        test=_loader(test_dir, False, test_dir),
        class_names=class_names,
        class_counts=count_class_images(train_dir),
    )
