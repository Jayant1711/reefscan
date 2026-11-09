"""DeepFish segmentation dataset (schema-adaptive pair discovery) + trainer.

DeepFish ships a Segmentation task with pixel-level fish masks. The archive
layout has varied across releases, so the dataset scans for image/mask pairs
by matching file stems inside candidate folders (any directory whose name
hints at masks/labels/segmentation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..utils import ensure_dir
from ..utils.logger import get_logger
from .unet import SmallUNet

logger = get_logger("seg.dataset")

MASK_HINTS = ("mask", "label", "segment", "annotation", "gt")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    """Find (image, mask) pairs under a DeepFish extraction or any folder tree."""
    data_dir = Path(data_dir)
    images_by_stem: dict[str, Path] = {}
    masks_by_stem: dict[str, Path] = {}
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        parent = path.parent.name.lower()
        stem = path.stem.lower()
        if any(h in parent for h in MASK_HINTS) or stem.endswith(("_mask", "_seg", "_label")):
            masks_by_stem[stem.replace("_mask", "").replace("_seg", "").replace("_label", "")] = path
        else:
            images_by_stem[stem] = path

    pairs = [(img, masks_by_stem[stem]) for stem, img in images_by_stem.items() if stem in masks_by_stem]
    logger.info("Discovered %d image/mask pairs under %s", len(pairs), data_dir)
    return pairs


class DeepFishSegmentationDataset(Dataset):
    """Paired image/mask dataset with resize + light augmentation."""

    def __init__(self, pairs: Sequence[tuple[Path, Path]], image_size: int = 256,
                 augment: bool = True) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        size = self.image_size
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.3:
                img = img.astype(np.float32) * np.random.uniform(0.85, 1.15)
                img = np.clip(img, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
        mask_t = torch.from_numpy((mask > 127).astype(np.float32))[None]  # (1, H, W)
        return img_t, mask_t


def make_seg_loaders(data_dir: str | Path, image_size: int = 256, batch_size: int = 8,
                     val_fraction: float = 0.2, num_workers: int = 0) -> tuple[DataLoader, DataLoader | None]:
    """Build train/val loaders from a discovered pair set."""
    pairs = discover_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs under {data_dir}. Download DeepFish (see docs/DATASETS.md)."
        )
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_fraction)) if len(pairs) > 5 else max(1, len(pairs) // 5)
    val_pairs = [pairs[i] for i in order[:n_val]]
    train_pairs = [pairs[i] for i in order[n_val:]]

    train_loader = DataLoader(DeepFishSegmentationDataset(train_pairs, image_size, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(DeepFishSegmentationDataset(val_pairs, image_size, augment=False),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_pairs else None
    return train_loader, val_loader


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + soft-Dice (standard segmentation objective)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
    return bce + dice


@torch.no_grad()
def dice_iou_scores(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    """Batch dice + IoU at a fixed threshold."""
    pred = (torch.sigmoid(logits) >= threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    dice = (2 * inter) / torch.clamp(pred.sum() + target.sum(), min=1.0)
    iou = inter / torch.clamp(union, min=1.0)
    return float(dice.item()), float(iou.item())


class SegmentationTrainer:
    """Small dedicated trainer for the fish-presence U-Net."""

    def __init__(self, image_size: int = 256, batch_size: int = 8, epochs: int = 10,
                 lr: float = 3e-4, device: torch.device | None = None, out_dir: str | Path = "models/seg") -> None:
        from ..utils.device import resolve_device

        self.device = device or resolve_device()
        self.epochs = epochs
        self.out_dir = ensure_dir(out_dir)
        self.model = SmallUNet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.image_size = image_size
        self.batch_size = batch_size

    def train(self, data_dir: str | Path, num_workers: int = 0) -> dict:
        train_loader, val_loader = make_seg_loaders(data_dir, self.image_size, self.batch_size, num_workers=num_workers)
        best_dice, best_iou = -1.0, float("nan")
        history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0
            for imgs, masks in tqdm(train_loader, desc=f"seg epoch {epoch}", leave=False):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(imgs)
                loss = dice_bce_loss(logits, masks)
                loss.backward()
                self.optimizer.step()
                running += loss.item() * imgs.size(0)

            entry = {"epoch": epoch, "train_loss": running / max(1, len(train_loader.dataset))}
            if val_loader is not None:
                self.model.eval()
                dices, ious = [], []
                for imgs, masks in val_loader:
                    logits = self.model(imgs.to(self.device))
                    d, i = dice_iou_scores(logits, masks.to(self.device))
                    dices.append(d)
                    ious.append(i)
                entry["val_dice"], entry["val_iou"] = float(np.mean(dices)), float(np.mean(ious))
                if entry["val_dice"] > best_dice:
                    best_dice = entry["val_dice"]
                    best_iou = entry["val_iou"]
                    torch.save({"model": self.model.state_dict(), "image_size": self.image_size},
                               self.out_dir / "best.pt")
            history.append(entry)
            logger.info("seg epoch %d: %s", epoch, entry)

        import pandas as pd

        pd.DataFrame(history).to_csv(self.out_dir / "history.csv", index=False)
        if not (self.out_dir / "best.pt").exists():  # single-epoch / no-val fallback
            torch.save({"model": self.model.state_dict(), "image_size": self.image_size},
                       self.out_dir / "best.pt")
        params = {
            "params_total": sum(p.numel() for p in self.model.parameters()),
            "params_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        return {"best_val_dice": max(best_dice, 0.0), "best_val_iou": best_iou,
                "epochs_ran": len(history), "params": params, "history": history}

    @torch.no_grad()
    def predict_mask(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        """Probability mask [H, W] for an RGB uint8 input (delegates to the model)."""
        return self.model.predict_mask(image_rgb_uint8)
