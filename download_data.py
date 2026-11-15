#!/usr/bin/env python
"""ReefScan AI — dataset download & preparation CLI.

Examples
--------
    python download_data.py                       # everything (all 3 datasets + prepare)
    python download_data.py --dataset coral       # NOAA coral only (~770 MB)
    python download_data.py --dataset deepfish    # DeepFish tar (~7.1 GB)
    python download_data.py --dataset f4k         # Fish4Knowledge ~487 MB (resumable curl)
    python download_data.py --prepare-f4k         # build curated species ImageFolder from the raw archive
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
    parser.add_argument("--dataset", choices=["all", "f4k", "deepfish", "coral"], default="all")
    parser.add_argument("--prepare-f4k", "--prepare-fishnet", dest="prepare_f4k", action="store_true",
                        help="Only build the curated species ImageFolder from the raw F4K archive")
    parser.add_argument("--platform", default=None,
                        choices=["kaggle", "colab", "lightning", "gpu", "local"],
                        help="Platform profile: aligns dataset paths with train.py --platform")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()





def apply_platform(cfg, platform: str | None):
    """Merge configs/platform/<name>.yaml (same profiles train.py uses)."""
    if not platform:
        return cfg
    overlay = Path(__file__).resolve().parent / "configs" / "platform" / f"{platform}.yaml"
    if overlay.exists():
        from src.config import merge_config_file

        merge_config_file(cfg, overlay)
        logger.info("Applied platform profile: %s", overlay.name)
    else:
        logger.warning("No platform profile at %s — using defaults.", overlay)
    return cfg

def prepare_f4k(cfg) -> None:
    from src.data.prepare_f4k import prepare_f4k_subset

    section = cfg.fish.prepare
    try:
        out = prepare_f4k_subset(
            raw_dir=Path(cfg.fish.raw_dir),
            out_dir=Path(cfg.fish.data_dir),
            min_images_per_class=int(section.get("min_images_per_class", 100)),
            top_k=int(section.get("top_k", 13)),
            split_fractions=tuple(section.get("split_fractions", [0.8, 0.1, 0.1])),
        )
        logger.info("Curated F4K subset -> %s", out)
    except FileNotFoundError as exc:
        logger.warning("F4K raw directory missing (%s).", exc)
        logger.warning("Skipping F4K subset preparation. To train on fish, download the raw F4K archive into datasets/f4k/raw/. See docs/DATASETS.md")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)
    apply_platform(cfg, args.platform)
    override_from_args(cfg, args)

    if args.prepare_f4k or args.prepare_fishnet:
        prepare_f4k(cfg)
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

    if args.dataset in ("all", "f4k"):
        from src.data.downloaders import download_f4k

        fish_raw = Path(getattr(cfg.fish, "raw_dir", root / "f4k" / "raw"))
        fish_raw.mkdir(parents=True, exist_ok=True)
        path = download_f4k(fish_raw)
        logger.info("✅ F4K ready -> %s", path)
        prepare_f4k(cfg)

    logger.info("All requested datasets ready. Next: python train.py --task fish --model resnet34")
    return 0


if __name__ == "__main__":
    from src.utils.cli import run_cli

    raise SystemExit(run_cli(main))
