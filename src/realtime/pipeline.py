"""ReefScanPipeline — the unified analysis engine used by CLI, dashboard and notebooks.

Pipeline per frame:

    enhance (auto-selected strategy)
      -> coral branch : binary CNN -> P(bleached) -> WGBS severity index/class
      -> fish branch  : U-Net mask -> boxes -> per-box species CNN (+ tracker)
      -> optional CAM heatmap on the species prediction
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from ..classification.models import build_model
from ..coral_analysis.severity import BleachingAssessor, SeverityAssessment
from ..coral_analysis.trend import BleachingTrendAnalyzer
from ..enhancement.registry import get_enhancer
from ..explainability.cam import GradCAM
from ..explainability.targets import find_target_layer
from ..segmentation.postprocess import BBox, masks_to_boxes
from ..segmentation.unet import SmallUNet
from ..utils import ensure_dir, load_json, read_image_bgr, save_json, write_image_bgr
from ..utils.device import resolve_device
from ..utils.logger import get_logger
from .annotate import annotate_frame
from .schema import CoralResult, FishDetection, FrameAnalysis
from .tracker import CentroidTracker

logger = get_logger("realtime.pipeline")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- loading
def load_best_classifier(
    task: str,
    models_dir: str | Path,
    device: torch.device,
    demo_dir: str | Path | None = None,
) -> tuple[torch.nn.Module, list[str], str, int]:
    """Load the best checkpoint for a task; fall back to demo checkpoint, then random init.

    Returns (model, class_names, source_description, image_size).
    """
    models_dir = Path(models_dir)
    candidates: list[Path] = []
    best_json = models_dir / task / "best_model.json"
    if best_json.exists():
        best_name = load_json(best_json)["best_model"]
        ckpt_path = models_dir / task / best_name / "best.pt"
        if ckpt_path.exists():
            candidates.append(ckpt_path)
    # Any task checkpoint at all.
    if not candidates:
        candidates.extend(sorted((models_dir / task).glob("*/best.pt")))
    # Demo fallback (smoke-test trained models).
    if demo_dir is not None:
        demo_dir = Path(demo_dir)
        demo_ckpt = demo_dir / f"{task}_demo.pt"
        if demo_ckpt.exists():
            candidates.append(demo_ckpt)

    for ckpt_path in candidates:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        class_names = ckpt.get("class_names") or [f"class_{i}" for i in range(ckpt.get("num_classes", 2))]
        model = build_model(ckpt["model_name"], len(class_names), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        logger.info("Loaded %s classifier from %s (%d classes)", task, ckpt_path, len(class_names))
        return model, list(class_names), str(ckpt_path), int(ckpt.get("image_size", 224))

    raise FileNotFoundError(
        f"No {task} checkpoint found under {models_dir} (looked for best_model.json, */best.pt, demo). "
        "Train a model first: python train.py --task " + task + " --model resnet34"
    )


def load_segmenter(models_dir: str | Path, device: torch.device) -> SmallUNet | None:
    """Load the fish-presence U-Net if trained; None otherwise."""
    ckpt_path = Path(models_dir) / "seg" / "best.pt"
    if not ckpt_path.exists():
        logger.info("No segmentation checkpoint (%s) — fish boxes disabled.", ckpt_path)
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SmallUNet()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    logger.info("Loaded fish segmenter from %s", ckpt_path)
    return model


def load_best_enhancer(results_dir: str | Path, enhancement_params: dict | None = None):
    """Load the auto-selected best enhancement strategy; default to CLAHE."""
    report_path = Path(results_dir) / "enhancement" / "enhancement_report.json"
    if report_path.exists():
        best_name = load_json(report_path)["best_strategy"]
    else:
        best_name = "clahe"
        logger.info("No enhancement report found — defaulting to '%s'.", best_name)
    return get_enhancer(best_name, enhancement_params), best_name


# --------------------------------------------------------------------------- pipeline
class ReefScanPipeline:
    """End-to-end underwater analysis pipeline (images, video frames, webcam)."""

    def __init__(self, cfg: Any, device: torch.device | None = None, enable_cam: bool | None = None) -> None:
        self.cfg = cfg
        self.device = device or resolve_device(cfg.device.prefer_gpu)
        enable_cam = cfg.realtime.cam_enabled if enable_cam is None else enable_cam

        params = dict(cfg.enhancement.params)
        self.enhancer, self.enhancer_name = load_best_enhancer(cfg.paths.results_dir, params)
        logger.info("Enhancement strategy: %s", self.enhancer_name)

        self.fish_model, self.fish_classes, self.fish_src, self.fish_img_size = load_best_classifier(
            "fish", cfg.paths.models_dir, self.device, cfg.paths.demo_dir
        )
        self.coral_model, self.coral_classes, self.coral_src, self.coral_img_size = load_best_classifier(
            "coral", cfg.paths.models_dir, self.device, cfg.paths.demo_dir
        )
        self.segmenter = load_segmenter(cfg.paths.models_dir, self.device)

        sev = cfg.coral.severity
        self.assessor = BleachingAssessor(
            fusion_lambda=sev.fusion_lambda,
            thresholds=tuple(sev.thresholds),
            s_max=sev.whiteness_s_max,
            v_min=sev.whiteness_v_min,
        )
        self._cam: GradCAM | None = GradCAM(self.fish_model, find_target_layer(self.fish_model)) if enable_cam else None

        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.tracker = CentroidTracker(
            max_disappeared=cfg.realtime.tracker_max_disappeared,
            min_distance=cfg.realtime.tracker_min_distance,
        )

    # ------------------------------------------------------------- preprocess
    def _preprocess(self, image_bgr: np.ndarray, size: int) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
        tensor = T.functional.to_tensor(pil)
        return self.normalize(tensor)[None].to(self.device)

    @torch.no_grad()
    def _predict(self, model: torch.nn.Module, image_bgr: np.ndarray, size: int) -> tuple[int, np.ndarray]:
        logits = model(self._preprocess(image_bgr, size))
        probs = F.softmax(logits.float(), dim=1)[0]
        return int(probs.argmax().item()), probs.cpu().numpy()

    # ---------------------------------------------------------------- image
    def analyze_image(self, image_bgr: np.ndarray, frame_idx: int | None = None,
                      track: bool = False) -> FrameAnalysis:
        """Analyse one BGR image. ``track=True`` enables unique-ID counting (video)."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        enhanced = self.enhancer.enhance(image_bgr) if self.cfg.enhancement.enabled else image_bgr
        timings["enhance"] = time.perf_counter() - t0

        # ---- coral branch
        t0 = time.perf_counter()
        coral_result: CoralResult | None = None
        if self.coral_model is not None:
            coral_idx, coral_probs = self._predict(self.coral_model, enhanced, self.coral_img_size)
            p_coral = float(coral_probs[coral_idx])
            p_bleached = (
                float(coral_probs[self.coral_classes.index("CORAL_BL")])
                if "CORAL_BL" in self.coral_classes
                else 1.0 - p_coral
            )
            assessment = self.assessor.assess(enhanced, p_bleached)
            coral_result = CoralResult(assessment=assessment, is_coral_scene=p_coral >= 0.5, p_coral=p_coral)
        timings["coral"] = time.perf_counter() - t0

        # ---- fish branch
        t0 = time.perf_counter()
        boxes: list[BBox]
        if self.segmenter is not None:
            prob_mask = self.segmenter.predict_mask(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            boxes = masks_to_boxes(
                prob_mask,
                threshold=self.cfg.seg.threshold,
                min_area=self.cfg.seg.min_box_area,
                max_boxes=self.cfg.seg.max_boxes,
                original_size=enhanced.shape[:2],
            )
        else:
            boxes = []
        if not boxes:
            # No segmenter / nothing found: analyse the whole frame as one region
            # so species classification always produces a usable result.
            h, w = enhanced.shape[:2]
            boxes = [BBox(0, 0, w, h, 1.0, h * w)]

        detections: list[FishDetection] = []
        best_box_idx = int(np.argmax([b.area for b in boxes]))
        species_probs: dict[str, float] | None = None
        for i, box in enumerate(boxes):
            crop = enhanced[max(0, box.y1):box.y2, max(0, box.x1):box.x2]
            if crop.size == 0:
                continue
            if self.fish_model is None:
                continue
            idx, probs = self._predict(self.fish_model, crop, self.fish_img_size)
            det = FishDetection(box=box, species=self.fish_classes[idx], confidence=float(probs[idx]))
            detections.append(det)
            if i == best_box_idx:
                top = np.argsort(-probs)[:5]
                species_probs = {self.fish_classes[j]: float(probs[j]) for j in top}

        if track:
            species_by_box = [(d.species, d.confidence) for d in detections]
            live = self.tracker.update([d.box for d in detections], frame_idx or 0, species_by_box)
            for det in detections:
                for tr in live:
                    if tr.centroid == (det.box.cx, det.box.cy):
                        det.track_id = tr.track_id
        timings["fish"] = time.perf_counter() - t0

        # ---- optional heatmap
        heatmap = None
        if self._cam is not None and detections:
            t0 = time.perf_counter()
            with self._cam:
                cam = self._cam.generate(self._preprocess(enhanced, self.fish_img_size))
            heatmap = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
            timings["cam"] = time.perf_counter() - t0

        return FrameAnalysis(
            detections=detections, coral=coral_result, enhanced=enhanced,
            heatmap=heatmap, timings=timings, species_probs=species_probs,
        )

    def analyze_file(self, image_path: str | Path, out_dir: str | Path | None = None,
                     save_artifacts: bool = True) -> FrameAnalysis:
        """Analyse an image file; optionally persist annotated + enhanced + heatmap."""
        image = read_image_bgr(image_path)
        analysis = self.analyze_image(image)
        if save_artifacts and out_dir is not None:
            out = ensure_dir(out_dir)
            stem = Path(image_path).stem
            analysis.annotated = annotate_frame(image, analysis.detections, analysis.coral.assessment if analysis.coral else None)
            write_image_bgr(analysis.annotated, out / f"{stem}_annotated.jpg")
            write_image_bgr(analysis.enhanced, out / f"{stem}_enhanced.jpg")
            if analysis.heatmap is not None:
                from ..explainability.visualize import save_heatmap

                analysis.heatmap_path = save_heatmap(analysis.heatmap, out / f"{stem}_heatmap.png")
        return analysis

    # ---------------------------------------------------------------- video
    def process_video(self, source: str | int, out_path: str | Path | None = None,
                      display: bool = False, progress_cb=None) -> dict[str, Any]:
        """Analyse a video file (or webcam index). Returns aggregated summary.

        Writes an annotated MP4 and a per-frame CSV; aggregates species counts
        (unique tracks) and the coral severity trajectory.
        """
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip = max(1, int(self.cfg.realtime.frame_skip))
        max_frames = int(self.cfg.realtime.max_frames) or None

        writer = None
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / frame_skip, (width, height))

        self.tracker = CentroidTracker(self.cfg.realtime.tracker_max_disappeared, self.cfg.realtime.tracker_min_distance)
        trend = BleachingTrendAnalyzer(self.cfg.analytics.trend_ewma_alpha)
        rows: list[dict] = []
        frame_idx = processed = 0
        t_start = time.perf_counter()

        while True:
            ok, frame = capture.read()
            if not ok or (max_frames and processed >= max_frames):
                break
            if frame_idx % frame_skip == 0:
                analysis = self.analyze_image(frame, frame_idx=frame_idx, track=True)
                if analysis.coral is not None:
                    trend.update(analysis.coral.assessment.index)
                annotated = annotate_frame(
                    frame, analysis.detections,
                    analysis.coral.assessment if analysis.coral else None,
                    analysis.heatmap,
                    fps=None, frame_idx=frame_idx,
                )
                if writer is not None:
                    writer.write(annotated)
                if display:
                    cv2.imshow("ReefScan AI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                rows.append({
                    "frame": frame_idx,
                    "n_fish": len(analysis.detections),
                    "dominant_species": analysis.dominant_species or "",
                    "best_confidence": max((d.confidence for d in analysis.detections), default=0.0),
                    "coral_class": analysis.coral.assessment.class_name if analysis.coral else "",
                    "bsi": analysis.coral.assessment.index if analysis.coral else np.nan,
                    "unique_fish_so_far": self.tracker.unique_count,
                })
                processed += 1
                if progress_cb:
                    progress_cb(frame_idx, total)
            frame_idx += 1

        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - t_start

        species_counts = self.tracker.species_counts()
        summary = {
            "frames_read": frame_idx,
            "frames_analyzed": processed,
            "elapsed_s": elapsed,
            "processing_fps": processed / max(elapsed, 1e-6),
            "unique_fish": self.tracker.unique_count,
            "species_counts": species_counts,
            "trend": trend.summary().__dict__,
            "video_out": str(out_path) if out_path else None,
        }
        if rows:
            csv_path = (Path(out_path).with_suffix("") if out_path else Path(self.cfg.paths.results_dir) / "realtime" / "frame_log").with_suffix(".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            summary["frame_log"] = str(csv_path)
        return summary

    # ---------------------------------------------------------------- webcam
    def process_webcam(self, out_path: str | Path | None = None) -> dict[str, Any]:
        """Live webcam analysis (Press 'q' to quit)."""
        return self.process_video(0, out_path, display=True)
