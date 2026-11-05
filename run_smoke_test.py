#!/usr/bin/env python
"""ReefScan AI — end-to-end smoke test on synthetic data (CPU, no downloads).

Proves every phase works, end to end, in minutes on a laptop:

    Phase 1  enhancement benchmark + auto-selection report
    Phase 2  fish classification (2 backbones, 2 epochs) + leaderboard
    Phase 3  coral binary model + WGBS severity assessment
    Phase 3.5 segmentation (tiny U-Net, 1 epoch) + box extraction
    Phase 4  GradCAM / GradCAM++ / ScoreCAM grids
    Phase 5  image analysis + synthetic video pipeline
    Phase 6  MBS / RHS / population / trend / env-risk analytics
    Phase 8  single-image PDF + survey PDF reports

Artifacts land under datasets/synthetic, models/demo, results/, reports/.
Demo checkpoints are copied to models/demo/ so the dashboard runs instantly.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils import save_json
from src.utils.device import resolve_device
from src.utils.logger import setup_logging
from src.utils.seed import seed_everything

logger = logging.getLogger("smoke")

FISH_CLASSES = [
    "Amphiprion_ocellaris", "Chromis_viridis", "Pomacanthus_imperator",
    "Sphyraena_barracuda", "Zebrasoma_flavescens", "Lutjanus_kasmira",
]

PASSED: list[str] = []
TIMINGS: dict[str, float] = {}


def step(name: str):
    def decorator(fn):
        def wrapper(*a, **kw):
            logger.info("=" * 66)
            logger.info("STEP %s", name)
            t0 = time.perf_counter()
            result = fn(*a, **kw)
            TIMINGS[name] = time.perf_counter() - t0
            PASSED.append(name)
            logger.info("STEP %s OK (%.1fs)", name, TIMINGS[name])
            return result
        return wrapper
    return decorator


# --------------------------------------------------------------------- data
def _texture(class_id: int, size: int, bleached: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic 'underwater' patch with class-correlated structure."""
    rng = rng or np.random.default_rng()
    hues = np.linspace(0, 179, len(FISH_CLASSES))
    base_hue = float(hues[class_id]) + rng.normal(0, 4)
    x = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, x)
    stripes = np.sin(6 * np.pi * xx + class_id) * np.cos(3 * np.pi * yy)
    value = 120 + 60 * stripes + rng.normal(0, 10, (size, size))
    sat = np.full((size, size), 150.0)

    if bleached:
        # Pale, desaturated, bright → the WGBS whiteness signature.
        sat = np.full((size, size), 30.0) + rng.normal(0, 8, (size, size))
        value = 200 + 30 * stripes + rng.normal(0, 8, (size, size))
    elif class_id == -1:  # healthy coral style: saturated warm
        base_hue = float(rng.uniform(5, 25))
        value = 150 + 40 * stripes + rng.normal(0, 10, (size, size))
        sat = np.full((size, size), 180.0)

    hsv = np.stack([
        np.clip(base_hue + 10 * stripes, 0, 179),
        np.clip(sat, 0, 255),
        np.clip(value, 0, 255),
    ], axis=2).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def build_synthetic_datasets(root: Path, size: int = 64) -> dict[str, Path]:
    """Create fish / coral / seg / enhance mini-datasets."""
    rng = np.random.default_rng(42)
    root.mkdir(parents=True, exist_ok=True)

    fish_dir = root / "fish"
    if not fish_dir.exists():
        for split, n in (("train", 24), ("validation", 8)):
            for cid, cls in enumerate(FISH_CLASSES):
                out = fish_dir / split / cls
                out.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    img = _texture(cid, size, rng=rng)
                    img = np.roll(img, rng.integers(-6, 6), axis=1)
                    cv2.imwrite(str(out / f"{cls}_{split}_{i:03d}.jpg"), img)

    coral_dir = root / "coral" / "imagefolders"
    if not coral_dir.exists():
        for split, n in (("train", 24), ("validation", 8)):
            for cls, bleached in (("CORAL", False), ("CORAL_BL", True)):
                out = coral_dir / split / cls
                out.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    img = _texture(-1 if not bleached else 0, size, bleached=bleached, rng=rng)
                    cv2.imwrite(str(out / f"{cls}_{split}_{i:03d}.jpg"), img)

    seg_dir = root / "seg"
    if not seg_dir.exists():
        images, masks = seg_dir / "images", seg_dir / "masks"
        images.mkdir(parents=True, exist_ok=True)
        masks.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            img = _texture(int(rng.integers(0, 6)), 256, rng=rng)
            mask = np.zeros((256, 256), dtype=np.uint8)
            for _ in range(rng.integers(1, 4)):
                cx, cy = int(rng.integers(40, 216)), int(rng.integers(40, 216))
                axes = (int(rng.integers(12, 30)), int(rng.integers(8, 18)))
                angle = int(rng.integers(0, 180))
                cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
                cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (30, 30, 30), 2)
            cv2.imwrite(str(images / f"seg_{i:02d}.jpg"), img)
            cv2.imwrite(str(masks / f"seg_{i:02d}.png"), mask)

    enh_dir = root / "enhance"
    if not enh_dir.exists():
        enh_dir.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            img = _texture(i % 6, 128, rng=rng)
            # Simulate underwater green-blue attenuation (red suppressed).
            img[..., 2] = (0.45 * img[..., 2]).astype(np.uint8)
            img = cv2.addWeighted(img, 0.9, np.full_like(img, 60), 0.1, 0)
            cv2.imwrite(str(enh_dir / f"enh_{i:02d}.jpg"), img)

    return {"fish": fish_dir, "coral": coral_dir, "seg": seg_dir, "enhance": enh_dir}


# -------------------------------------------------------------------- steps
@step("Phase 1: enhancement benchmark")
def run_enhancement(cfg, paths) -> str:
    from src.enhancement.selector import EnhancementSelector
    from src.utils import list_images, read_image_bgr

    images = [read_image_bgr(p) for p in list_images(paths["enhance"])]
    selector = EnhancementSelector(weights=dict(cfg.enhancement.weights),
                                   strategies=cfg.enhancement.strategies,
                                   params=dict(cfg.enhancement.params))
    best, df, grid = selector.select_best(images, out_dir=Path(cfg.paths.results_dir) / "enhancement")
    selector.save_report(df, best, Path(cfg.paths.results_dir) / "enhancement")
    assert grid and grid.exists(), "enhancement comparison grid missing"
    assert len(df) == len(cfg.enhancement.strategies), "not all strategies evaluated"
    return best


@step("Phase 2: fish classification training")
def train_fish(cfg, device) -> None:
    import torch

    from src.classification.class_balancing import class_weights
    from src.classification.datasets import build_dataloaders
    from src.classification.engine import Trainer, TrainConfig
    from src.classification.evaluate import build_leaderboard, run_model_evaluation

    cfg.set_nested("fish.data_dir", str(PROJECT_ROOT / "datasets/synthetic/fish"))
    bundle = build_dataloaders(cfg.fish.data_dir, image_size=cfg.fish.image_size,
                               batch_size=cfg.fish.batch_size, num_workers=0)
    assert bundle.num_classes == len(FISH_CLASSES)

    weights = torch.tensor(class_weights(bundle.class_counts), dtype=torch.float32)
    demo_models_dir = PROJECT_ROOT / "models/demo"
    for model_name in ("resnet34", "mobilenet_v3"):
        out_dir = demo_models_dir / "fish" / model_name
        trainer = Trainer(model_name, bundle.num_classes, bundle.train, bundle.val,
                          TrainConfig.from_cfg(cfg.fish), bundle.class_names, device, out_dir,
                          class_weight=weights)
        trainer.fit()
        run_model_evaluation("fish", model_name, cfg.fish.data_dir, demo_models_dir,
                             cfg.paths.results_dir, bundle.class_names,
                             image_size=cfg.fish.image_size, batch_size=cfg.fish.batch_size,
                             num_workers=0, device=device)
    df, best = build_leaderboard("fish", ["resnet34", "mobilenet_v3"], cfg.paths.results_dir,
                                 demo_models_dir)
    logger.info("Fish leaderboard:\n%s", df.to_string(index=False))


@step("Phase 3: coral classification training + WGBS")
def train_coral(cfg, device) -> None:
    import torch

    from src.classification.class_balancing import class_weights
    from src.classification.datasets import build_dataloaders
    from src.classification.engine import Trainer, TrainConfig
    from src.classification.evaluate import build_leaderboard, run_model_evaluation
    from src.coral_analysis.severity import BleachingAssessor
    from src.utils import read_image_bgr

    cfg.set_nested("coral.data_dir", str(PROJECT_ROOT / "datasets/synthetic/coral/imagefolders"))
    bundle = build_dataloaders(cfg.coral.data_dir, image_size=cfg.coral.image_size,
                               batch_size=cfg.coral.batch_size, num_workers=0)
    weights = torch.tensor(class_weights(bundle.class_counts), dtype=torch.float32)
    demo_models_dir = PROJECT_ROOT / "models/demo"
    out_dir = demo_models_dir / "coral" / "resnet34"
    trainer = Trainer("resnet34", bundle.num_classes, bundle.train, bundle.val,
                      TrainConfig.from_cfg(cfg.coral), bundle.class_names, device, out_dir,
                      class_weight=weights)
    trainer.fit()
    run_model_evaluation("coral", "resnet34", cfg.coral.data_dir, demo_models_dir,
                         cfg.paths.results_dir, bundle.class_names,
                         image_size=cfg.coral.image_size, batch_size=cfg.coral.batch_size,
                         num_workers=0, device=device)
    build_leaderboard("coral", ["resnet34"], cfg.paths.results_dir, demo_models_dir)

    # WGBS sanity: bleached-looking synthetic patch must score higher than saturated one.
    assessor = BleachingAssessor()
    pale = _texture(0, 96, bleached=True)
    vivid = _texture(-1, 96, bleached=False)
    pale_assess = assessor.assess(pale, p_bleached=0.9)
    vivid_assess = assessor.assess(vivid, p_bleached=0.05)
    assert pale_assess.index > vivid_assess.index, "WGBS ordering violated"
    logger.info("WGBS: pale BSI=%.1f (%s) vs vivid BSI=%.1f (%s)",
                pale_assess.index, pale_assess.class_name, vivid_assess.index, vivid_assess.class_name)


@step("Phase 3.5: segmentation training + boxes")
def train_seg(cfg) -> None:
    from src.segmentation.engine_seg import SegmentationTrainer
    from src.segmentation.postprocess import masks_to_boxes

    cfg.set_nested("seg.data_dir", str(PROJECT_ROOT / "datasets/synthetic/seg"))
    trainer = SegmentationTrainer(image_size=128, batch_size=4, epochs=2, out_dir=PROJECT_ROOT / "models/seg")
    result = trainer.train(PROJECT_ROOT / "datasets/synthetic/seg")
    img = cv2.imread(str(PROJECT_ROOT / "datasets/synthetic/seg/images/seg_00.jpg"))
    mask = trainer.predict_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = masks_to_boxes(mask, threshold=0.3, min_area=80)
    logger.info("Segmentation dice=%.3f, sample boxes=%d", result["best_val_dice"], len(boxes))
    assert result["best_val_dice"] > 0.0


@step("Phase 4: explainability grids")
def run_xai(cfg, device) -> None:
    import torch

    from src.classification.models import build_model
    from src.explainability.visualize import generate_xai_grid
    from src.utils import read_image_bgr

    ckpt = torch.load(PROJECT_ROOT / "models/demo/fish/mobilenet_v3/best.pt",
                      map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    image = read_image_bgr(sorted((PROJECT_ROOT / "datasets/synthetic/enhance").glob("*.jpg"))[0])
    from torchvision import transforms as T
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = T.Compose([T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(
        Image.fromarray(rgb).resize((cfg.fish.image_size, cfg.fish.image_size)))[None].to(device)

    paths = generate_xai_grid(
        model, tensor, image, PROJECT_ROOT / "results/xai",
        methods=["gradcam", "gradcam_plusplus", "scorecam"],
        class_names=ckpt["class_names"], scorecam_batch=8, scorecam_max_channels=8,
    )
    assert paths["grid"].exists()
    for method in ("gradcam", "gradcam_plusplus", "scorecam"):
        assert paths[method].exists(), f"{method} heatmap missing"


@step("Phase 5: realtime pipeline (image + video)")
def run_realtime(cfg, device):
    import pandas as pd

    from src.realtime.pipeline import ReefScanPipeline

    # Use the demo checkpoints for the pipeline: point models_dir at models/demo
    # and stage the segmenter there so every branch (boxes + species + coral) is exercised.
    demo_dir = PROJECT_ROOT / "models/demo"
    seg_src = PROJECT_ROOT / "models/seg/best.pt"
    if seg_src.exists():
        seg_dst = demo_dir / "seg/best.pt"
        seg_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seg_src, seg_dst)
    cfg.set_nested("paths.models_dir", str(demo_dir))
    cfg.set_nested("paths.demo_dir", str(demo_dir))
    pipeline = ReefScanPipeline(cfg, device=device, enable_cam=False)

    analyses, images = [], []
    for img_path in sorted((PROJECT_ROOT / "datasets/synthetic/enhance").glob("*.jpg"))[:3]:
        img = cv2.imread(str(img_path))
        analyses.append(pipeline.analyze_image(img, frame_idx=len(analyses), track=True))
        images.append(img)
    assert all(a.coral is not None for a in analyses)
    assert len(analyses[0].detections) >= 1, "expected whole-frame fallback detection"

    # Synthetic video: 40 frames with moving fish blob.
    video_path = PROJECT_ROOT / "datasets/synthetic/clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
    for i in range(40):
        frame = _texture(i % 6, 128)
        cx = 20 + i * 2
        cv2.ellipse(frame, (cx, 64), (16, 9), 0, 0, 360, (25, 25, 25), -1)
        writer.write(frame)
    writer.release()

    summary = pipeline.process_video(str(video_path),
                                     PROJECT_ROOT / "results/realtime/smoke_annotated.mp4")
    assert summary["frames_analyzed"] > 0
    assert Path(summary["video_out"]).exists()
    assert Path(summary["frame_log"]).exists()
    logger.info("Video: %s | unique fish=%d | trend=%s",
                summary["video_out"], summary["unique_fish"], summary["trend"]["trend_label"])
    return pipeline, analyses, images


@step("Phase 6: analytics bundle")
def run_analytics(cfg, analyses, images) -> dict:
    from src.analytics.survey import SurveyAnalyzer

    bundle = SurveyAnalyzer(cfg).analyze(analyses, images)
    data = bundle.to_dict()
    assert 0 <= data["biodiversity"]["mbs"] <= 100
    assert 0 <= data["reef_health"]["score"] <= 100
    save_json(data, PROJECT_ROOT / "results/analytics/smoke_survey.json")
    logger.info("MBS=%.1f RHS=%.1f (%s) trend=%s env=%s",
                data["biodiversity"]["mbs"], data["reef_health"]["score"],
                data["reef_health"]["category"], data["trend"]["trend_label"],
                data["env_risk"]["category"])
    return data


@step("Phase 8: PDF reports")
def run_reports(cfg, analyses, images, analytics_data) -> None:
    from src.analytics.survey import SurveyAnalyzer
    from src.realtime.annotate import annotate_frame
    from src.reporting.pdf_report import ReefScanReportBuilder
    from src.utils import write_image_bgr

    builder = ReefScanReportBuilder(cfg)
    out_dir = PROJECT_ROOT / "reports"

    img = images[0]
    analysis = analyses[0]
    annotated = annotate_frame(img, analysis.detections,
                               analysis.coral.assessment if analysis.coral else None)
    annotated_path = write_image_bgr(annotated, out_dir / "assets" / "smoke_annotated.jpg")
    enhanced_path = write_image_bgr(analysis.enhanced, out_dir / "assets" / "smoke_enhanced.jpg")

    pdf1 = builder.build_image_report(
        "synthetic_sample.jpg", analysis, out_dir / "ReefScan_Image_Report_SMOKE.pdf",
        assets={"original": None, "enhanced": enhanced_path, "annotated": annotated_path, "heatmap": None},
    )
    assert pdf1.exists() and pdf1.stat().st_size > 20_000, "image report too small / missing"

    bundle = SurveyAnalyzer(cfg).analyze(analyses, images)
    pdf2 = builder.build_survey_report(bundle, out_dir / "ReefScan_Survey_Report_SMOKE.pdf",
                                       site_name="Synthetic Reef (smoke test)")
    assert pdf2.exists() and pdf2.stat().st_size > 20_000, "survey report too small / missing"


def promote_demo_checkpoints() -> None:
    """Copy the best trained demo checkpoints where the dashboard looks for them."""
    demo = PROJECT_ROOT / "models/demo"
    demo.mkdir(parents=True, exist_ok=True)

    fish_src = demo / "fish/mobilenet_v3/best.pt"
    if fish_src.exists():
        shutil.copy2(fish_src, demo / "fish_demo.pt")
    coral_src = demo / "coral/resnet34/best.pt"
    if coral_src.exists():
        shutil.copy2(coral_src, demo / "coral_demo.pt")
    logger.info("Demo checkpoints promoted to %s", demo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[], help="Step names to skip")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    seed_everything(cfg.project.seed)
    device = resolve_device(prefer_gpu=False)  # smoke test is CPU by design

    # Speed: tiny images + few epochs for the smoke run.
    for task in ("fish", "coral"):
        cfg.set_nested(f"{task}.image_size", 64)
        cfg.set_nested(f"{task}.batch_size", 16)
        cfg.set_nested(f"{task}.epochs", 2)
        cfg.set_nested(f"{task}.warmup_epochs", 0)
        cfg.set_nested(f"{task}.early_stopping", 0)

    paths = build_synthetic_datasets(PROJECT_ROOT / "datasets/synthetic")
    logger.info("Synthetic datasets ready under %s", PROJECT_ROOT / "datasets/synthetic")

    t_all = time.perf_counter()
    best_enh = run_enhancement(cfg, paths)
    if "Phase 1" not in args.skip:
        logger.info("Auto-selected enhancement strategy: %s", best_enh)

    train_fish(cfg, device)
    train_coral(cfg, device)
    train_seg(cfg)
    run_xai(cfg, device)
    _pipeline, analyses, images = run_realtime(cfg, device)
    analytics_data = run_analytics(cfg, analyses, images)
    run_reports(cfg, analyses, images, analytics_data)
    promote_demo_checkpoints()

    logger.info("=" * 66)
    logger.info("SMOKE TEST PASSED — all %d steps (%.1fs total)", len(PASSED), time.perf_counter() - t_all)
    for name, secs in TIMINGS.items():
        logger.info("  %-45s %6.1fs", name, secs)
    logger.info("=" * 66)
    logger.info("Artifacts: results/ · reports/ · models/demo/ · datasets/synthetic/")
    logger.info("Now try: streamlit run app.py   (demo models will be active)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
