#!/usr/bin/env python
"""ReefScan AI — training CLI (fish species | coral bleaching | fish segmentation).

Examples
--------
    python train.py --task fish  --model resnet34
    python train.py --task fish  --model all
    python train.py --task coral --model efficientnet_b0 --set coral.epochs=5
    python train.py --task seg
    python train.py --task fish  --model resnet34 --quick     # tiny CPU smoke run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.classification.datasets import build_dataloaders
from src.classification.engine import Trainer, TrainConfig, evaluate_model_loader
from src.classification.evaluate import build_leaderboard, run_model_evaluation
from src.classification.models import SUPPORTED_MODELS
from src.config import ensure_output_dirs, load_config, override_from_args
from src.utils import ensure_dir
from src.utils.device import configure_threads, resolve_device
from src.utils.logger import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["fish", "coral", "seg"])
    parser.add_argument("--model", default="resnet34",
                        help=f"Backbone for fish/coral: {SUPPORTED_MODELS} or 'all'")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--set", nargs="*", default=[], help="Config overrides: section.key=value")
    parser.add_argument("--quick", action="store_true", help="Tiny fast run (CPU smoke test)")
    return parser.parse_args()


def apply_quick(cfg, task: str) -> None:
    """Shrink a task config to a fast CPU-feasible run."""
    cfg.set_nested(f"{task}.epochs", 2)
    cfg.set_nested(f"{task}.image_size", 64)
    cfg.set_nested(f"{task}.batch_size", 16)
    cfg.set_nested(f"{task}.warmup_epochs", 0)
    cfg.set_nested(f"{task}.early_stopping", 2)
    if task == "fish":
        cfg.set_nested("fish.prepare.top_k", 8)
        cfg.set_nested("fish.prepare.min_images_per_class", 10)


def train_classifier(task: str, model_name: str, cfg, device: torch.device) -> None:
    section = cfg[task]
    data_dir = Path(section.data_dir)
    if not data_dir.exists():
        hint = "python download_data.py" if task == "coral" else (
            "python download_data.py --dataset fishnet && python download_data.py --prepare-fishnet"
        )
        raise FileNotFoundError(
            f"{task} dataset not found at '{data_dir}'. Prepare it first ({hint}); "
            "see docs/DATASETS.md."
        )

    bundle = build_dataloaders(
        data_dir,
        image_size=section.image_size,
        batch_size=section.batch_size,
        num_workers=cfg.device.num_workers,
        balance_train=True,
        train_cfg=section.get("augmentation", {}),
    )
    logger.info("Classes (%d): %s", bundle.num_classes, bundle.class_names[:12])

    # Class balancing: inverse-sqrt weights for the loss + weighted sampler already applied.
    from src.classification.class_balancing import class_weights

    weights = torch.tensor(
        class_weights(bundle.class_counts, mode="inverse_sqrt"), dtype=torch.float32
    ).to(device)

    train_cfg = TrainConfig.from_cfg(section)
    if task == "coral" and getattr(section, "ordinal_loss", False):
        train_cfg.ordinal_loss = True

    out_dir = ensure_dir(Path(cfg.paths.models_dir) / task / model_name)
    trainer = Trainer(
        model_name=model_name,
        num_classes=bundle.num_classes,
        train_loader=bundle.train,
        val_loader=bundle.val or bundle.test,
        cfg=train_cfg,
        class_names=bundle.class_names,
        device=device,
        out_dir=out_dir,
        class_weight=weights,
    )
    trainer.fit()

    # Evaluate on the held-out split and refresh the leaderboard.
    results_dir = ensure_dir(Path(cfg.paths.results_dir))
    run_model_evaluation(
        task_name=task,
        model_name=model_name,
        data_dir=data_dir,
        models_dir=cfg.paths.models_dir,
        results_dir=results_dir,
        class_names=bundle.class_names,
        image_size=section.image_size,
        batch_size=section.batch_size,
        num_workers=cfg.device.num_workers,
        device=device,
    )
    build_leaderboard(task, cfg[task].models, results_dir, cfg.paths.models_dir,
                      metric_for_best=section.metric_for_best)


def train_segmentation(cfg, device: torch.device) -> None:
    from src.segmentation.engine_seg import SegmentationTrainer

    data_dir = Path(cfg.seg.data_dir)
    if not data_dir.exists() or not any(data_dir.rglob("*")):
        from src.data.downloaders import download_deepfish

        download_deepfish(cfg.paths.data_root / "deepfish")

    trainer = SegmentationTrainer(
        image_size=cfg.seg.image_size,
        batch_size=cfg.seg.batch_size,
        epochs=cfg.seg.epochs,
        lr=cfg.seg.lr,
        device=device,
        out_dir=Path(cfg.paths.models_dir) / "seg",
    )
    result = trainer.train(data_dir / "DeepFish" if (data_dir / "DeepFish").exists() else data_dir)
    logger.info("Segmentation done: best val dice %.4f", result["best_val_dice"])


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    override_from_args(cfg, args)
    if args.quick:
        apply_quick(cfg, args.task)

    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)
    ensure_output_dirs(cfg)
    seed_everything(cfg.project.seed)
    configure_threads(cfg.device.num_threads)
    device = resolve_device(cfg.device.prefer_gpu)

    if args.task == "seg":
        if args.quick:
            cfg.set_nested("seg.epochs", 1)
        train_segmentation(cfg, device)
        return 0

    models = SUPPORTED_MODELS if args.model == "all" else [args.model]
    for model_name in models:
        logger.info("=== Training %s / %s ===", args.task, model_name)
        train_classifier(args.task, model_name, cfg, device)

    logger.info("Training complete. Leaderboard: results/%s/model_leaderboard.csv", args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
