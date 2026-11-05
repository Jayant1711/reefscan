# System Architecture — ReefScan AI

## 1. High-level data flow

```mermaid
flowchart TB
    A[Input: image / video / webcam] --> B[Phase 1 · Enhancement\nauto-selected strategy]
    B --> C1[Fish branch]
    B --> C2[Coral branch]
    C1 --> D1[SmallUNet segmentation\nDeepFish masks] --> E1[Boxes\nconnected components]
    E1 --> F1[Species CNN\nResNet34 / EffB0 / MBV3 / ConvNeXt-T]
    F1 --> G1[Centroid tracker\nunique IDs]
    C2 --> D2[Binary coral CNN\nNOAA patches]
    D2 --> E2[WGBS severity\nP(bleached) ⊕ whiteness]
    E2 --> F2[5-class condition\nBSI 0-100]
    F1 --> H[Phase 4 · XAI\nGradCAM / ++ / ScoreCAM]
    G1 --> I[Phase 6 · Analytics\nMBS · RHS · population · trend · env risk]
    F2 --> I
    I --> J[Phase 8 · PDF reports]
    F1 --> K[Phase 7 · Streamlit dashboard]
    F2 --> K
    H --> K
    J --> K
```

## 2. Module map (src/)

| Package | Responsibility | Key classes |
|---|---|---|
| `config` | YAML config, dot-access, CLI overrides | `DotDict`, `load_config` |
| `utils` | logging, device, seeding, IO, plotting | `setup_logging`, `resolve_device` |
| `enhancement` | 8 strategies + 4 metrics + selector | `Enhancer`, `EnhancementSelector` |
| `data` | downloaders + FishNet subset builder | `download_coral`, `prepare_fishnet_subset` |
| `classification` | 4-backbone benchmark, balancing, engine, eval | `Trainer`, `build_dataloaders` |
| `coral_analysis` | WGBS, risk, trend, health report | `BleachingAssessor`, `BleachingTrendAnalyzer` |
| `segmentation` | U-Net, pair discovery, boxes | `SmallUNet`, `masks_to_boxes` |
| `explainability` | 3 CAM methods, overlays, grids | `GradCAM`, `ScoreCAM` |
| `realtime` | unified pipeline, annotation, tracking | `ReefScanPipeline`, `CentroidTracker` |
| `analytics` | MBS/RHS/population/trend/env-risk | `SurveyAnalyzer` |
| `reporting` | charts + PDF | `ReefScanReportBuilder` |
| `dashboard` | Streamlit pages + cached loaders | `get_pipeline` |

## 3. Design principles
1. **Dependency direction:** `dashboard → realtime → {classification, coral_analysis, segmentation, enhancement} → utils/config`. No upward imports; the analytics layer depends only on `realtime.schema` dataclasses (avoids circulars).
2. **Registry pattern** for enhancers (open-closed): new strategies need zero changes to selector/CLI code.
3. **Single checkpoint contract** (`best.pt` + `best_meta.json`) consumed identically by CLI, pipeline and dashboard.
4. **Graceful degradation:** no GPU → CPU; no segmenter → whole-frame classification; no trained model → demo checkpoints (clearly banner-warned); no `datasets` lib → parquet fallback.
5. **Determinism:** global seeding in every entry point; smoke test pins seeds.

## 4. Model I/O contracts
- Images are **BGR uint8** everywhere except torch tensors (RGB, ImageNet-normalised).
- `FrameAnalysis` is the universal per-frame result; `SurveyAnalytics.to_dict()` is the universal survey result (feeds dashboard, JSON logs and PDF).
- Checkpoints carry `class_names` so inference never needs the training folder.

## 5. Threading & performance model
- Enhancement runs per-frame in pure OpenCV (no torch) → cheap.
- Torch inference is batch-of-1 for images; video uses frame-skip + tracker to amortise cost.
- ScoreCAM is the only expensive XAI method; it is batched and channel-capped.
