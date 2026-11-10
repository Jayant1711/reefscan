#!/usr/bin/env python
"""ReefScan AI — evaluation CLI: metrics, leaderboards, enhancement benchmark, XAI grids.

Examples
--------
    python evaluate.py --task fish
    python evaluate.py --task fish --model efficientnet_b0
    python evaluate.py --enhancement --data datasets/fishnet/curated/train
    python evaluate.py --xai --image sample.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ensure_output_dirs, load_config, override_from_args
from src.utils import read_image_bgr
from src.utils.device import resolve_device
from src.utils.logger import setup_logging

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["fish", "coral", "seg"], default=None)
    parser.add_argument("--model", default=None, help="One model or omit for all trained ones")
    parser.add_argument("--enhancement", action="store_true", help="Run Phase-1 benchmark")
    parser.add_argument("--data", default=None, help="Image folder for the enhancement benchmark")
    parser.add_argument("--xai", action="store_true", help="Generate XAI grid for --image")
    parser.add_argument("--image", default=None, help="Image for --xai")
    parser.add_argument("--xai-task", default="fish", choices=["fish", "coral"],
                        help="Which trained classifier the XAI grid uses (default fish)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def run_task_evaluation(cfg, task: str, model: str | None, device) -> None:
    if task == "seg":
        from src.segmentation.evaluate_seg import evaluate_segmentation

        data_dir = Path(cfg.seg.data_dir)
        deepfish_root = data_dir / "DeepFish" if (data_dir / "DeepFish").exists() else data_dir
        metrics = evaluate_segmentation(
            models_dir=cfg.paths.models_dir,
            results_dir=Path(cfg.paths.results_dir),
            device=device,
            data_dir=deepfish_root,
            image_size=cfg.seg.image_size,
            batch_size=cfg.seg.batch_size,
            min_box_area=cfg.seg.min_box_area,
        )
        logger.info("Seg metrics -> %s", Path(cfg.paths.results_dir) / "seg" / "metrics.json")
        return metrics

    from src.classification.evaluate import build_leaderboard, run_model_evaluation

    section = cfg[task]
    models_dir = Path(cfg.paths.models_dir) / task
    results_dir = Path(cfg.paths.results_dir)

    if model:
        names = [model]
    else:
        names = sorted({p.parent.name for p in models_dir.glob("*/best.pt")})
        if not names:
            raise FileNotFoundError(f"No trained checkpoints under {models_dir}. Run train.py first.")

    class_names: list[str] | None = None
    for name in names:
        import torch

        ckpt = torch.load(models_dir / name / "best.pt", map_location="cpu", weights_only=False)
        class_names = ckpt.get("class_names")
        run_model_evaluation(
            task_name=task, model_name=name, data_dir=Path(section.data_dir),
            models_dir=cfg.paths.models_dir, results_dir=results_dir,
            class_names=class_names or [f"class_{i}" for i in range(ckpt.get("num_classes", 2))],
            image_size=section.image_size, batch_size=section.batch_size,
            num_workers=cfg.device.num_workers, device=device,
        )
    build_leaderboard(task, section.models, results_dir, cfg.paths.models_dir,
                      metric_for_best=section.metric_for_best)
    logger.info("Leaderboard -> %s", results_dir / task / "model_leaderboard.csv")


def run_enhancement_benchmark_cli(cfg, data_dir: str | None) -> None:
    from src.enhancement.selector import EnhancementSelector
    from src.utils import list_images

    folder = Path(data_dir) if data_dir else _guess_sample_folder(cfg)
    images = [read_image_bgr(p) for p in list_images(folder)[: cfg.enhancement.sample_size]]
    if not images:
        raise FileNotFoundError(f"No images found under {folder} — pass --data <folder>.")
    selector = EnhancementSelector(weights=dict(cfg.enhancement.weights),
                                   strategies=cfg.enhancement.strategies,
                                   params=dict(cfg.enhancement.params))
    best, df, _ = selector.select_best(images, out_dir=Path(cfg.paths.results_dir) / "enhancement")
    selector.save_report(df, best, Path(cfg.paths.results_dir) / "enhancement")
    logger.info("Enhancement report -> %s (best: %s)",
                Path(cfg.paths.results_dir) / "enhancement", best)


def _guess_sample_folder(cfg) -> Path:
    for candidate in ("datasets/fishnet/curated/train", "datasets/coral/imagefolders/train",
                      "datasets/synthetic/enhance"):
        path = Path(cfg.paths.data_root).parent / candidate if not candidate.startswith("datasets") else Path(candidate)
        if path.exists() and any(path.rglob("*.jpg")):
            return path
    return Path("datasets/fishnet/curated/train")


def run_xai_cli(cfg, image_path: str, device, task: str = "fish") -> None:
    import torch

    from src.explainability.visualize import generate_xai_grid
    from src.realtime.pipeline import load_best_classifier

    model, class_names, _src, img_size = load_best_classifier(
        task, cfg.paths.models_dir, device, cfg.paths.demo_dir
    )
    image = read_image_bgr(image_path)
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from PIL import Image
    from torchvision import transforms as T

    tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(Image.fromarray(rgb).resize((img_size, img_size)))[None].to(device)

    paths = generate_xai_grid(
        model, tensor, image, Path(cfg.paths.results_dir) / "xai",
        methods=cfg.xai.methods, class_names=class_names,
        scorecam_batch=cfg.xai.scorecam_batch, scorecam_max_channels=cfg.xai.scorecam_max_channels,
    )
    logger.info("XAI grid -> %s", paths["grid"])


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    override_from_args(cfg, args)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)
    ensure_output_dirs(cfg)
    device = resolve_device(cfg.device.prefer_gpu)

    if args.task:
        run_task_evaluation(cfg, args.task, args.model, device)
    if args.enhancement:
        run_enhancement_benchmark_cli(cfg, args.data)
    if args.xai:
        if not args.image:
            raise ValueError("--xai requires --image <path>")
        run_xai_cli(cfg, args.image, device, task=args.xai_task)
    if not (args.task or args.enhancement or args.xai):
        logger.info("Nothing to do — pass --task, --enhancement or --xai. See --help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
