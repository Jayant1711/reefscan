# User Manual — ReefScan AI

A plain-language guide for non-programmers using the dashboard.

## 1. Starting the app
1. Open a terminal in the project folder.
2. Run `streamlit run app.py`.
3. Your browser opens at `http://localhost:8501`.

> The first run loads models (10–30 s). If you see an orange **Demo models** banner, the
> app is using tiny practice models trained on synthetic images — perfect for exploring,
> not for research results.

## 2. Pages

### 🏠 Home
Shows which models are loaded and a quick-start cheat sheet.

### 🐟 Species Classification (= Upload Image)
1. Click **Browse files**, pick an underwater photo.
2. Within seconds you get the annotated image (boxes + labels), per-detection confidences,
   top-5 probability bars, and the coral status banner.
3. Toggle **Overlay CAM heatmap** to see *why* the model decided.
4. Click **Download annotated image** to save the result.

### 🪸 Coral Analysis
Upload a coral patch photo. You get:
- severity gauge (0–100),
- the 5-class condition chip (Healthy → Dead),
- the WGBS breakdown (CNN probability vs measured whiteness),
- management recommendations.

### ✨ Enhancement Comparison
Upload one image; every enhancement strategy runs on it. Compare them side by side and
read the ranked metric table (PSNR/SSIM/UCIQE/UIQM). The 🏆 line tells you the winner.

### 🔥 Heatmaps (XAI)
Pick methods (GradCAM, GradCAM++, ScoreCAM) and download each heatmap. Red = regions that
drove the species decision.

### 🎬 Upload Video
1. Upload a survey clip (MP4/MOV/AVI).
2. Choose "analyze every Nth frame" (2 is a good CPU default) and a frame cap.
3. Press **Run analysis**. You get the annotated MP4, unique-fish counts, species bar
   chart, and the bleaching-severity trend line — all downloadable.

### 📊 Analytics
Upload *several* photos from the same site. ReefScan computes:
- **Marine Biodiversity Score** (species richness/evenness),
- **Reef Health Score** (severity + coverage + environment),
- bleaching trend and environmental risk,
- full species population table.

### 📄 Reports
One click builds a PDF from your latest analysis: images, detections, severity gauge,
scores, recommendations — ready to attach to a lab report.

## 3. Interpreting scores
| Score | 0–40 | 40–60 | 60–80 | 80–100 |
|---|---|---|---|---|
| Reef Health | Critical/Poor | Fair | Good | Excellent |
| Bleaching severity (BSI) | Healthy–Moderate | Moderate | Severe | Dead |
| Environmental risk | Low | Moderate | High | Critical |

## 4. Tips for good inputs
- Sharp, well-exposed frames; avoid heavy backscatter (particle flashes).
- Coral patches: fill the frame with the coral colony.
- Videos: 10–60 s per station is plenty; keep the camera drift slow so the tracker keeps IDs.

## 5. CLI equivalents (for advanced users)
```bash
python inference.py --mode image --input photo.jpg --cam --report
python inference.py --mode video --input survey.mp4
python evaluate.py --task fish
```
