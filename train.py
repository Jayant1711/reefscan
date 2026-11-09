#!/usr/bin/env python
"""ReefScan AI — training CLI (fish species | coral bleaching | fish segmentation).

Examples
--------
    python train.py --task fish  --model resnet34
    python train.py --task fish  --model all --strategy feature_extraction
    python train.py --task coral --model efficientnet_b0 --set coral.epochs=5
    python train.py --task seg
    python train.py --task fish  --model resnet34 --quick     # tiny CPU smoke run
    python train.py --task coral --model all --platform kaggle  # platform profile
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.classification.datasets import build_dataloaders
from src.classification.engine import STRATEGIES, Trainer, TrainConfig, evaluate_model_loader
from src.classification.evaluate import build_leaderboard, run_model_evaluation
from src.classification.models import SUPPORTED_MODELS
from src.config import ensure_output_dirs, load_config, override_from_args
from src.utils import ensure_dir
from src.utils.device import configure_threads, resolve_device
from src.utils.logger import setup_logging
from src.utils.provenance import build_manifest, count_params, detect_platform, git_sha, gpu_name
from src.utils.seed import seed_everything

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["fish", "coral", "seg"])
    parser.add_argument("--model", default="resnet34",
                        help=f"Backbone for fish/coral: {SUPPORTED_MODELS} or 'all'")
    parser.add_argument("--strategy", default=None, choices=list(STRATEGIES),
                        help="Transfer-learning strategy (default: config '%s.strategy')" % "task")
    parser.add_argument("--platform", default=None, choices=["kaggle", "colab", "lightning", "local"],
                        help="Platform profile overlay from configs/platform/<name>.yaml")
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


def _write_manifest(cfg, task: str, model_name: str, strategy: str, train_seconds: float,
                    epochs_ran: int, final_metrics: dict, params: dict) -> Path:
    """Write runs/<task>/<model>_<strategy>/manifest.json and return its path."""
    section = cfg[task]
    manifest = build_manifest(
        task=task,
        model_name=model_name,
        strategy=strategy,
        dataset=str(section.get("data_dir", "")),
        config_snapshot={
            "image_size": section.get("image_size"),
            "batch_size": section.get("batch_size"),
            "epochs": section.get("epochs"),
            "lr": section.get("lr"),
            "optimizer": section.get("optimizer"),
            "scheduler": section.get("scheduler"),
            "early_stopping": section.get("early_stopping"),
            "metric_for_best": section.get("metric_for_best"),
        },
        params=params,
        train_seconds=train_seconds,
        epochs_ran=epochs_ran,
        final_metrics=final_metrics,
        artifacts={
            "checkpoint": str(Path(cfg.paths.models_dir) / task / model_name / "best.pt"),
            "metrics": str(Path(cfg.paths.results_dir) / task / model_name / "metrics.json"),
            "leaderboard": str(Path(cfg.paths.results_dir) / task / "model_leaderboard.csv"),
        },
        project_root=Path(__file__).resolve().parent,
    )
    manifest_dir = ensure_dir(Path(cfg.paths.results_dir).parent / "runs" / task / f"{model_name}_{strategy}")
    manifest_path = manifest_dir / "manifest.json"
    from src.utils import save_json

    save_json(manifest, manifest_path)
    logger.info("Run manifest -> %s", manifest_path)
    return manifest_path


def train_classifier(task: str, model_name: str, cfg, device: torch.device,
                     strategy_override: str | None = None) -> None:
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
    if strategy_override:
        train_cfg.strategy = strategy_override

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
    t_start = time.time()
    history = trainer.fit()
    train_seconds = sum(h.seconds for h in history)
    epochs_ran = len(history)

    # Evaluate on the held-out split and refresh the leaderboard.
    results_dir = ensure_dir(Path(cfg.paths.results_dir))
    metrics = run_model_evaluation(
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

    _write_manifest(
        cfg, task, model_name, train_cfg.strategy, train_seconds, epochs_ran,
        final_metrics={k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and k != "n"},
        params=trainer.params,
    )


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
    t_start = time.time()
    result = trainer.train(data_dir / "DeepFish" if (data_dir / "DeepFish").exists() else data_dir)
    train_seconds = time.time() - t_start
    logger.info("Segmentation done: best val dice %.4f", result["best_val_dice"])

    _write_manifest(
        cfg, "seg", "unet_small", "from_scratch", train_seconds,
        epochs_ran=result.get("epochs_ran", 0),
        final_metrics={"best_val_dice": result["best_val_dice"],
                       "best_val_iou": result.get("best_val_iou", float("nan"))},
        params=result.get("params", count_params(trainer.model)),
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.platform:
        overlay = Path(__file__).resolve().parent / "configs" / "platform" / f"{args.platform}.yaml"
        if overlay.exists():
            from src.config import merge_config_file

            merge_config_file(cfg, overlay)
            logger.info("Applied platform profile: %s", overlay.name)
        else:
            logger.warning("No platform profile at %s — continuing with defaults.", overlay)
    override_from_args(cfg, args)
    if args.strategy:
        cfg.set_nested(f"{args.task}.strategy", args.strategy)
    if args.quick:
        apply_quick(cfg, args.task)

    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)
    ensure_output_dirs(cfg)
    seed_everything(cfg.project.seed)
    configure_threads(cfg.device.num_threads)
    device = resolve_device(cfg.device.prefer_gpu)
    logger.info("Platform=%s | device=%s | git=%s",
                detect_platform(), gpu_name(device), git_sha())

    if args.task == "seg":
        if args.quick:
            cfg.set_nested("seg.epochs", 1)
        train_segmentation(cfg, device)
        return 0

    models = SUPPORTED_MODELS if args.model == "all" else [args.model]
    for model_name in models:
        logger.info("=== Training %s / %s (%s) ===",
                    args.task, model_name, cfg[args.task].get("strategy", "finetune"))
        train_classifier(args.task, model_name, cfg, device)

    logger.info("Training complete. Leaderboard: results/%s/model_leaderboard.csv", args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
