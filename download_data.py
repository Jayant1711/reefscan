#!/usr/bin/env python
"""ReefScan AI — dataset download & preparation CLI.

Examples
--------
    python download_data.py                       # everything (all 3 datasets + prepare)
    python download_data.py --dataset coral       # NOAA coral only (~770 MB)
    python download_data.py --dataset deepfish    # DeepFish tar (~7.1 GB)
    python download_data.py --dataset fishnet     # FishNet zip (~15 GB)
    python download_data.py --prepare-fishnet     # build curated subset from the raw archive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, override_from_args
from src.utils.logger import setup_logging

logger = logging.getLogger("download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["all", "fishnet", "deepfish", "coral"], default="all")
    parser.add_argument("--prepare-fishnet", action="store_true",
                        help="Only build the curated ImageFolder subset from the raw FishNet archive")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def prepare_fishnet(cfg) -> None:
    from src.data.prepare_fishnet import prepare_fishnet_subset

    section = cfg.fish.prepare
    try:
        out = prepare_fishnet_subset(
            raw_dir=Path(cfg.fish.raw_dir),
            out_dir=Path(cfg.fish.data_dir),
            level=section.level,
            top_k=int(section.top_k),
            min_images_per_class=int(section.min_images_per_class),
        )
        logger.info("Curated FishNet subset -> %s", out)
    except FileNotFoundError as exc:
        logger.warning("FishNet raw directory missing (%s).", exc)
        logger.warning("Skipping FishNet subset preparation. To use FishNet, download the raw dataset into datasets/fishnet/raw/. See docs/DATASETS.md")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    override_from_args(cfg, args)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)

    if args.prepare_fishnet:
        prepare_fishnet(cfg)
        return 0

    root = Path(cfg.paths.data_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "coral"):
        from src.data.downloaders import download_coral

        path = download_coral(root / "coral")
        logger.info("✅ Coral dataset ready -> %s", path)

    if args.dataset in ("all", "deepfish"):
        from src.data.downloaders import download_deepfish

        path = download_deepfish(root / "deepfish")
        logger.info("✅ DeepFish ready -> %s", path)

    if args.dataset in ("all", "fishnet"):
        from src.data.downloaders import download_fishnet

        path = download_fishnet(root / "fishnet" / "raw")
        logger.info("✅ FishNet ready -> %s", path)
        prepare_fishnet(cfg)

    logger.info("All requested datasets ready. Next: python train.py --task fish --model resnet34")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
