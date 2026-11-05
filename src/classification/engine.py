"""Training engine: AMP, cosine schedule, mixup/cutmix, early stopping, checkpoints.

Designed to run unchanged on free Colab GPUs and on CPU laptops.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils import ensure_dir, save_json
from .losses import build_loss
from .models import build_model, param_groups

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    early_stopping: int = 5
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    ordinal_loss: bool = False
    ordinal_lam: float = 0.5
    metric_for_best: str = "val_weighted_f1"
    amp: bool = True
    grad_clip: float = 5.0

    @classmethod
    def from_cfg(cls, section: Mapping) -> "TrainConfig":
        section = dict(section)
        aug_keys = ("augmentation", "models", "severity", "prepare", "classes", "class_colors", "hf_repo",
                    "data_dir", "raw_dir", "task_name", "image_size", "batch_size", "pretrained",
                    "freeze_backbone", "models_dir")
        section = {k: v for k, v in section.items() if k not in aug_keys}
        return cls(**section)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float = float("nan")
    val_acc: float = float("nan")
    val_macro_f1: float = float("nan")
    val_weighted_f1: float = float("nan")
    lr: float = 0.0
    seconds: float = 0.0


def mixup_criterion(logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor,
                    lam: float, criterion: nn.Module) -> torch.Tensor:
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)


class Trainer:
    """Single-task trainer for classification models."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: TrainConfig,
        class_names: Sequence[str],
        device: torch.device,
        out_dir: str | Path,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cfg = cfg
        self.class_names = list(class_names)
        self.out_dir = ensure_dir(out_dir)

        self.model = build_model(model_name, num_classes).to(device)
        self.criterion = build_loss(
            num_classes,
            class_weight=class_weight.to(device) if class_weight is not None else None,
            label_smoothing=cfg.label_smoothing,
            ordinal=cfg.ordinal_loss,
            ordinal_lam=cfg.ordinal_lam,
        )
        groups = param_groups(self.model, cfg.lr)
        if cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

        steps_per_epoch = max(1, len(train_loader))
        if cfg.scheduler == "cosine":
            warmup = max(0, cfg.warmup_epochs) * steps_per_epoch
            total = cfg.epochs * steps_per_epoch
            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return (step + 1) / max(1, warmup)
                progress = (step - warmup) / max(1, total - warmup)
                return 0.5 * (1 + math.cos(math.pi * progress))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.epochs // 3 or 1, gamma=0.1)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: list[EpochLog] = []
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self._bad_epochs = 0
        self._stop = False

    # ------------------------------------------------------------- utilities
    def _lr_now(self) -> float:
        return float(self.optimizer.param_groups[-1]["lr"])

    # ------------------------------------------------------------- train loop
    def fit(self) -> list[EpochLog]:
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch()
            log = EpochLog(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                           lr=self._lr_now(), seconds=time.time() - t0)

            if self.val_loader is not None:
                val_metrics = self.validate()
                log.val_loss = val_metrics["loss"]
                log.val_acc = val_metrics["accuracy"]
                log.val_macro_f1 = val_metrics["macro_f1"]
                log.val_weighted_f1 = val_metrics["weighted_f1"]
                self._save_history_csv()
                self._checkpoint_if_best(log, val_metrics)
                logger.info("Epoch %d | loss %.4f | acc %.4f | val loss %.4f | val acc %.4f | wF1 %.4f | %.1fs",
                            epoch, train_loss, train_acc, log.val_loss, log.val_acc, log.val_weighted_f1, log.seconds)
                if self._check_early_stop(log):
                    logger.info("Early stopping at epoch %d (best epoch %d).", epoch, self.best_epoch)
                    break
            else:
                self._save_history_csv()
                self._checkpoint_if_best(log, {"accuracy": train_acc})
                logger.info("Epoch %d | loss %.4f | acc %.4f | %.1fs", epoch, train_loss, train_acc, log.seconds)

            self.history.append(log)

        # Restore best weights for downstream evaluation.
        best_path = self.out_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            logger.info("Restored best weights from epoch %d.", ckpt.get("epoch", -1))
        return self.history

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"train {self.model_name}", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if self.cfg.mixup_alpha > 0 and np.random.rand() <= self.cfg.mixup_prob:
                    lam = float(np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha))
                    idx = torch.randperm(inputs.size(0), device=inputs.device)
                    mixed = lam * inputs + (1 - lam) * inputs[idx]
                    outputs = self.model(mixed)
                    loss = mixup_criterion(outputs, targets, targets[idx], lam, self.criterion)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += inputs.size(0)
            pbar.set_postfix(loss=f"{total_loss / max(1, seen):.3f}", acc=f"{correct / max(1, seen):.3f}")
        return total_loss / max(1, seen), correct / max(1, seen)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        all_preds, all_targets = [], []
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            preds = outputs.argmax(1)
            total_loss += loss.item() * inputs.size(0)
            correct += (preds == targets).sum().item()
            seen += inputs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
        return {
            "loss": total_loss / max(1, seen),
            "accuracy": correct / max(1, seen),
            "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        }

    # ------------------------------------------------------------- persistence
    def _checkpoint_if_best(self, log: EpochLog, metrics: Mapping[str, float]) -> None:
        metric = self.cfg.metric_for_best.replace("val_", "")
        value = metrics.get(metric, log.val_weighted_f1 if "weighted_f1" in metrics else log.val_acc)
        if value > self.best_metric:
            self.best_metric = float(value)
            self.best_epoch = log.epoch
            self._bad_epochs = 0
            save_json(
                {
                    "model_name": self.model_name,
                    "epoch": log.epoch,
                    "metric_for_best": self.cfg.metric_for_best,
                    "value": float(value),
                    "class_names": self.class_names,
                },
                self.out_dir / "best_meta.json",
            )
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "model_name": self.model_name,
                    "num_classes": len(self.class_names),
                    "class_names": self.class_names,
                    "epoch": log.epoch,
                    "metric": float(value),
                },
                self.out_dir / "best.pt",
            )
            logger.info("  ^ new best (%s=%.4f) — checkpoint saved.", self.cfg.metric_for_best, value)
        else:
            self._bad_epochs += 1

    def _check_early_stop(self, log: EpochLog) -> bool:
        if self.cfg.early_stopping and self._bad_epochs >= self.cfg.early_stopping:
            return True
        return False

    def _save_history_csv(self) -> None:
        pd.DataFrame([asdict(h) for h in self.history]).to_csv(self.out_dir / "history.csv", index=False)


@torch.no_grad()
def evaluate_model_loader(model: nn.Module, loader: DataLoader, device: torch.device,
                          num_classes: int) -> dict:
    """Full evaluation: predictions, probabilities, targets, timings."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model.eval()
    preds, targets_all, probs_all = [], [], []
    n_samples = 0
    t_start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probs = F.softmax(outputs.float(), dim=1)
        preds.append(outputs.argmax(1).cpu())
        probs_all.append(probs.cpu())
        targets_all.append(targets)
        n_samples += inputs.size(0)
    elapsed = time.perf_counter() - t_start

    if not preds:
        return {"n": 0}
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets_all).numpy()
    y_prob = torch.cat(probs_all).numpy()

    top5 = min(5, num_classes)
    top5_acc = float(np.mean([1.0 if y_true[i] in np.argsort(-y_prob[i])[:top5] else 0.0
                              for i in range(len(y_true))])) if len(y_true) else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": top5_acc,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "throughput_fps": float(n_samples / max(elapsed, 1e-6)),
    }
