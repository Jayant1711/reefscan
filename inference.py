#!/usr/bin/env python
"""ReefScan AI — inference CLI: image | video | webcam.

Examples
--------
    python inference.py --mode image  --input photo.jpg --report
    python inference.py --mode video  --input survey.mp4 --output results/realtime
    python inference.py --mode webcam --output results/realtime
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ensure_output_dirs, load_config, override_from_args
from src.utils import ensure_dir, save_json, timestamp_slug
from src.utils.device import resolve_device
from src.utils.logger import setup_logging

logger = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["image", "video", "webcam"])
    parser.add_argument("--input", default=None, help="Image/video path (not needed for webcam)")
    parser.add_argument("--output", default=None, help="Output directory (default results/realtime)")
    parser.add_argument("--cam", action="store_true", help="Overlay GradCAM heatmap")
    parser.add_argument("--report", action="store_true", help="Also generate a PDF report (image mode)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def analyze_image_mode(cfg, args, device) -> int:
    import cv2

    from src.realtime.annotate import annotate_frame
    from src.realtime.pipeline import ReefScanPipeline
    from src.reporting.pdf_report import ReefScanReportBuilder
    from src.utils import read_image_bgr, write_image_bgr

    if not args.input:
        raise ValueError("--mode image requires --input <path>")
    image = read_image_bgr(args.input)
    pipeline = ReefScanPipeline(cfg, device=device, enable_cam=args.cam)
    analysis = pipeline.analyze_image(image)

    out_dir = ensure_dir(args.output or str(Path(cfg.paths.results_dir) / "realtime"))
    stem = Path(args.input).stem
    annotated = annotate_frame(image, analysis.detections,
                               analysis.coral.assessment if analysis.coral else None,
                               analysis.heatmap)
    annotated_path = write_image_bgr(annotated, out_dir / f"{stem}_annotated.jpg")
    enhanced_path = write_image_bgr(analysis.enhanced, out_dir / f"{stem}_enhanced.jpg")
    heatmap_path = None
    if analysis.heatmap is not None:
        from src.explainability.visualize import save_heatmap

        heatmap_path = save_heatmap(analysis.heatmap, out_dir / f"{stem}_heatmap.png")

    result = {
        "image": str(args.input),
        "detections": [
            {"species": d.species, "confidence": d.confidence,
             "box": [d.x1, d.y1, d.x2, d.y2],
             "track_id": d.track_id}
            for d in analysis.detections
        ],
        "coral": None if analysis.coral is None else {
            "class": analysis.coral.assessment.class_name,
            "index": analysis.coral.assessment.index,
            "p_bleached": analysis.coral.assessment.p_bleached,
            "whiteness": analysis.coral.assessment.whiteness,
            "recommendations": analysis.coral.assessment.recommendations,
        },
        "top5_species": analysis.species_probs,
        "enhancer": pipeline.enhancer_name,
        "timings_ms": {k: round(v * 1000, 1) for k, v in analysis.timings.items()},
        "artifacts": {
            "annotated": str(annotated_path),
            "enhanced": str(enhanced_path),
            "heatmap": str(heatmap_path) if heatmap_path else None,
        },
    }
    save_json(result, out_dir / f"{stem}_result.json")
    logger.info("Annotated image -> %s", annotated_path)
    logger.info("Result JSON    -> %s", out_dir / f"{stem}_result.json")

    if args.report:
        builder = ReefScanReportBuilder(cfg)
        pdf = builder.build_image_report(
            args.input, analysis,
            Path(cfg.paths.reports_dir) / f"ReefScan_Report_{stem}_{timestamp_slug()}.pdf",
            assets={"original": Path(args.input), "enhanced": enhanced_path,
                    "annotated": annotated_path, "heatmap": heatmap_path},
        )
        logger.info("PDF report     -> %s", pdf)

    if analysis.detections:
        best = max(analysis.detections, key=lambda d: d.confidence)
        print(f"\n detections : {len(analysis.detections)}")
        print(f" top        : {best.species} ({best.confidence:.1%})")
    if analysis.coral is not None:
        a = analysis.coral.assessment
        print(f" coral      : {a.label} — BSI {a.index:.1f}/100 "
              f"(P_bleached={a.p_bleached:.2f}, whiteness={a.whiteness:.2f})")
    return 0


def analyze_video_mode(cfg, args, device) -> int:
    from src.realtime.pipeline import ReefScanPipeline

    if not args.input:
        raise ValueError("--mode video requires --input <path>")
    pipeline = ReefScanPipeline(cfg, device=device, enable_cam=args.cam)
    out_path = Path(args.output or str(Path(cfg.paths.results_dir) / "realtime")) / "annotated_video.mp4"
    summary = pipeline.process_video(args.input, out_path)
    save_json(summary, Path(out_path).with_suffix("").with_name("video_summary.json"))
    print(json.dumps({k: v for k, v in summary.items() if k != "trend"}, indent=2, default=str))
    print(f" trend      : {summary['trend']['trend_label']} "
          f"(slope {summary['trend']['slope_per_step']:+.2f}/obs)")
    return 0


def analyze_webcam_mode(cfg, args, device) -> int:
    from src.realtime.pipeline import ReefScanPipeline

    pipeline = ReefScanPipeline(cfg, device=device, enable_cam=args.cam)
    out_path = Path(args.output or str(Path(cfg.paths.results_dir) / "realtime")) / "webcam_annotated.mp4"
    summary = pipeline.process_webcam(out_path)
    print(json.dumps({k: v for k, v in summary.items() if k != "trend"}, indent=2, default=str))
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    override_from_args(cfg, args)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.file)
    ensure_output_dirs(cfg)
    device = resolve_device(cfg.device.prefer_gpu)

    if args.mode == "image":
        return analyze_image_mode(cfg, args, device)
    if args.mode == "video":
        return analyze_video_mode(cfg, args, device)
    return analyze_webcam_mode(cfg, args, device)


if __name__ == "__main__":
    from src.utils.cli import run_cli

    raise SystemExit(run_cli(main))
