#!/usr/bin/env python
"""ReefScan AI — dataset cleaning CLI (report-only by default, --apply quarantines).

Examples
--------
    python clean_data.py --dataset coral            # scan + report only
    python clean_data.py --dataset coral --apply    # move flagged files to _quarantine/
    python clean_data.py --dataset fishnet --apply
    python clean_data.py --dataset seg --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, override_from_args
from src.data.cleaning import clean_imagefolder, clean_segmentation
from src.utils.logger import setup_logging

logger = logging.getLogger("clean")

RECIPES = {
    "coral": {
        "target": "ImageFolder (corrupt + min-size + MD5/dHash duplicates + split leakage)",
        "kind": "imagefolder",
        "dir": lambda cfg: Path(cfg.paths.data_root) / "coral" / "imagefolders",
    },
    "fishnet": {
        "target": "curated ImageFolder (corrupt + duplicates + leakage + class audit)",
        "kind": "imagefolder",
        "dir": lambda cfg: Path(cfg.fish.data_dir),
    },
    "seg": {
        "target": "DeepFish pairs (pairing + corrupt + min mask-area)",
        "kind": "segmentation",
        "dir": lambda cfg: Path(cfg.seg.data_dir),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(RECIPES),
                        help="Which dataset to clean")
    parser.add_argument("--apply", action="store_true",
                        help="Move flagged files to <dataset>/_quarantine/ (default: report only)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    override_from_args(cfg, args)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)

    recipe = RECIPES[args.dataset]
    target = recipe["dir"](cfg)
    report_path = Path(cfg.paths.results_dir) / "cleaning" / f"{args.dataset}_cleaning_report.json"
    logger.info("Cleaning %s dataset at %s (apply=%s)", args.dataset, target, args.apply)

    if recipe["kind"] == "imagefolder":
        report = clean_imagefolder(target, apply=args.apply, report_path=report_path)
        if args.dataset == "fishnet":
            _audit_classes(target, report)
    else:
        report = clean_segmentation(target, apply=args.apply, report_path=report_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Report -> %s", report_path)
    logger.info("Counts: %s", json.dumps(report.get("counts", {})))
    return 0


def _audit_classes(root: Path, report: dict) -> None:
    """Warn when a class has fewer images than the curated minimum."""
    counts: dict[str, int] = {}
    for path in root.rglob("*.jpg"):
        if "_quarantine" in path.parts or len(path.relative_to(root).parts) < 2:
            continue
        cls = path.relative_to(root).parts[1]
        counts[cls] = counts.get(cls, 0) + 1
    small = {c: n for c, n in counts.items() if n < 10}
    report["class_counts"] = counts
    if small:
        logger.warning("Classes with <10 images after cleaning: %s", small)


if __name__ == "__main__":
    raise SystemExit(main())
