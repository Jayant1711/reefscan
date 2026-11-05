# API Documentation — ReefScan AI (Python)

Everything below is importable from the project root (`sys.path` includes it).

## 1. Configuration
```python
from src.config import load_config, ensure_output_dirs
cfg = load_config("configs/default.yaml", overrides=["fish.epochs=5"])
ensure_output_dirs(cfg)
cfg.fish.image_size            # dot access, nested
cfg.get_nested("coral.severity.fusion_lambda", 0.6)
```

## 2. Enhancement (`src.enhancement`)
```python
from src.enhancement import get_enhancer, EnhancementSelector, psnr, ssim, uciqe, uiqm

enh = get_enhancer("clahe", {"clahe_clip_limit": 2.0})
out = enh.enhance(image_bgr)                       # BGR uint8 -> BGR uint8

selector = EnhancementSelector(weights={"uciqe": .4, "uiqm": .3, "ssim": .3})
best, report_df, grid_png = selector.select_best(images, out_dir="results/enhancement")
```
Registry names: `clahe, dark_channel, udcp, white_balance, color_restoration, gamma, histogram_eq, fusion_wcg`.
Add a strategy: subclass `Enhancer`, decorate with `@register`, import in `strategies.py`.

## 3. Classification (`src.classification`)
```python
from src.classification import build_model, build_dataloaders, Trainer, TrainConfig
from src.classification.class_balancing import class_weights
import torch

bundle = build_dataloaders("datasets/fishnet/curated", image_size=224, batch_size=32)
weights = torch.tensor(class_weights(bundle.class_counts), dtype=torch.float32)
trainer = Trainer("resnet34", bundle.num_classes, bundle.train, bundle.val,
                  TrainConfig(epochs=15, lr=1e-4), bundle.class_names,
                  device=torch.device("cpu"), out_dir="models/fish/resnet34",
                  class_weight=weights)
history = trainer.fit()        # early stopping + best checkpoint handled
```
`build_model(name, num_classes, pretrained=True, dropout=0.2, freeze_backbone=False)` with
`name ∈ {resnet34, efficientnet_b0, mobilenet_v3, convnext_tiny}`.

## 4. Coral analysis (`src.coral_analysis`)
```python
from src.coral_analysis import BleachingAssessor, coral_risk_score, BleachingTrendAnalyzer

assessor = BleachingAssessor(fusion_lambda=0.6, thresholds=(15, 35, 60, 85))
verdict  = assessor.assess(image_bgr, p_bleached=0.73)   # -> SeverityAssessment
verdict.index, verdict.class_name, verdict.recommendations

trend = BleachingTrendAnalyzer(ewma_alpha=0.3); trend.update(42.0)
trend.summary().trend_label
```

## 5. Explainability (`src.explainability`)
```python
from src.explainability import GradCAM, GradCAMPlusPlus, ScoreCAM, find_target_layer
from src.explainability.visualize import generate_xai_grid

cam = GradCAM(model, find_target_layer(model))
with cam:
    heat = cam.generate(input_tensor, class_idx=None)    # np.ndarray [H,W] in [0,1]

paths = generate_xai_grid(model, tensor, image_bgr, "results/xai",
                          methods=["gradcam", "gradcam_plusplus", "scorecam"],
                          class_names=bundle.class_names)
```

## 6. Real-time pipeline (`src.realtime`)
```python
from src.realtime import ReefScanPipeline

pipeline = ReefScanPipeline(cfg, enable_cam=False)
analysis = pipeline.analyze_image(image_bgr)      # -> FrameAnalysis
analysis.detections      # list[FishDetection(.species, .confidence, .x1..y2, .track_id)]
analysis.coral.assessment.index / .class_name / .recommendations
analysis.enhanced, analysis.heatmap, analysis.timings

summary = pipeline.process_video("survey.mp4", "out.mp4")   # frames CSV + trends
summary = pipeline.process_webcam("out.mp4")
```

## 7. Analytics (`src.analytics`)
```python
from src.analytics import SurveyAnalyzer, marine_biodiversity_score, reef_health_score

bundle = SurveyAnalyzer(cfg).analyze(analyses, source_images)   # -> SurveyAnalytics
data = bundle.to_dict()   # JSON-serialisable: biodiversity / reef_health / population / trend / env_risk
```

## 8. Reporting (`src.reporting`)
```python
from src.reporting import ReefScanReportBuilder

builder = ReefScanReportBuilder(cfg)
builder.build_image_report(image_path, analysis, "reports/report.pdf", assets={...})
builder.build_survey_report(survey_analytics, "reports/survey.pdf", site_name="Site A")
```

## 9. Segmentation (`src.segmentation`)
```python
from src.segmentation import SmallUNet, masks_to_boxes, SegmentationTrainer

trainer = SegmentationTrainer(image_size=256, epochs=10)
trainer.train("datasets/deepfish/DeepFish")
prob = trainer.predict_mask(rgb_uint8)               # [H,W] in [0,1]
boxes = masks_to_boxes(prob, threshold=0.5, min_area=150)
```

## 10. Checkpoint format
```python
{
  "model": state_dict, "model_name": "resnet34", "num_classes": 25,
  "class_names": [...], "epoch": 12, "metric": 0.87, "image_size": 224
}
```
