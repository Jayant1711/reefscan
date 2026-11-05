# Optimization Guide — free-tier compute (CPU / Colab T4 / low-end GPU)

## 1. Where the time goes (per 224px frame, CPU)
| Stage | approx. cost | mitigation |
|---|---|---|
| Enhancement (classical) | 5–30 ms | negligible |
| U-Net mask (256px) | 40–120 ms | 192px input; frame-skip |
| Species CNN (1 box) | 15–60 ms | MobileNetV3; ONNX export |
| Coral CNN | 15–60 ms | share backbone in future work |
| ScoreCAM | 2–10 s | channel cap, offline only |

## 2. CPU playbook
- Keep `device.num_workers: 0` on Windows; on Linux set 2–4.
- `torch.set_num_threads(n)` to physical cores (config `device.num_threads`).
- Prefer `mobilenet_v3` for interactive use; it anchors the CPU column of the leaderboard.
- Video: `realtime.frame_skip: 2–3` — the centroid tracker bridges skipped frames, so unique
  counts stay stable while cost drops proportionally.
- Downscale inputs before the segmenter (`seg.image_size: 192`) when frames are ≥720p.
- Optional: `python scripts/export_onnx.py` then `onnxruntime` for ~1.5–2.5× inference speedup.

## 3. Colab T4 playbook
- AMP is automatic (`Trainer.use_amp`), effectively doubling throughput.
- `batch_size 32 @ 224px` fits all four backbones in 15 GB.
- Train fish overnight with `nohup`-style background cells; checkpoints resume per-model.
- Persist to Drive (notebook cell 8) so session resets do not lose weights.

## 4. Memory tricks
- `pin_memory=True` only when CUDA is available (already conditional).
- Evaluation is `@torch.no_grad()` everywhere; XAI methods release hooks via context managers.
- For 17k-class experiments (future), switch heads to ArcFace-style embeddings instead of softmax.

## 5. Quantisation (experimental)
```python
import torch
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```
Gives ~2–4× smaller Linear layers with <1% accuracy drift — safe for the classifier heads.

## 6. What we deliberately did NOT optimise
- Multi-GPU/DDP (out of scope for free tiers).
- FP16 inference on CPU (unsupported by most CPU builds).
- TensorRT (vendor-locked, not free-toolchain everywhere).
