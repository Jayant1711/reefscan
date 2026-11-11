# Deployment Guide — ReefScan AI

## 1. Local dashboard
```bash
streamlit run app.py
```
Model resolution order (per task): `models/<task>/best_model.json` → any `models/<task>/*/best.pt` → `models/demo/<task>_demo.pt`.

## 2. Google Colab (shareable demo)
```python
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501        # prints a public URL
```
(Alternative: `pyngrok` with a free authtoken.)

## 3. Docker
The repo root ships a production `Dockerfile` (CPU-wheel torch ~2.5 GB image,
non-root user, libglib for OpenCV, healthcheck on Streamlit's `/_stalker/health`)
plus `docker-compose.yml` with bind-mounts for `models/ results/ reports/`:

```bash
docker compose up --build       # dashboard on http://localhost:8501
# or without compose:
docker build -t reefscan .
docker run -p 8501:8501 -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results reefscan
```

Note: `provenance.git_sha()` reports `"unknown"` inside the container (`.git` is
not copied — keeps the image lean); manifests written in-container simply lack
the SHA. On Linux hosts whose uid ≠ 1000, chown the bind-mounted dirs once
(`sudo chown -R 1000:1000 models results reports`) so the non-root app user can
write manifests and logs.

## 4. CPU inference optimisation
- `torch.set_num_threads(k)` — cap threads (config `device.num_threads`).
- MobileNetV3-Small is the default CPU champion (~4× faster than ResNet34 at similar accuracy on small sets).
- Video: `realtime.frame_skip: 2` halves cost with negligible counting drift (tracker bridges gaps).
- Optional ONNX export for ~often ~2× (benchmark-dependent) CPU speedup:
```python
python scripts/export_onnx.py --task fish --model mobilenet_v3   # models/fish/mobilenet_v3/best.onnx
```

## 5. Edge / low-end GPU
- 4 GB VRAM is enough: batch 16 @ 224px with AMP fits all four backbones.
- Reduce `seg.image_size` to 192 for real-time box prediction on iGPUs.

## 6. Scheduled reef monitoring (reference architecture)
```
GoPro/RPi capture → nightly upload → batch inference.py --mode video
  → results/realtime/<date>/ → SurveyAnalyzer → PDF report → email/Drive
```
The pipeline is stateless per run; long-term trend analytics come from concatenating
`results/analytics/*.json` across dates (see `docs/RESEARCH_METHODOLOGY.md` §6).

## 7. Health checks
| Check | Command | Expected |
|---|---|---|
| Import graph | `python -c "import src.realtime"` | silent |
| Unit tests | `pytest tests/ -q` | all pass |
| End-to-end | `python run_smoke_test.py` | `SMOKE TEST PASSED` |
| Dashboard | `streamlit run app.py` | home page renders |
