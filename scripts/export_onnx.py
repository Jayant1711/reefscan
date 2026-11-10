#!/usr/bin/env python
"""Export a trained ReefScan classifier to ONNX for faster CPU inference.

Usage:
    python scripts/export_onnx.py --task fish --model mobilenet_v3
Requires: pip install onnx onnxruntime (optional extras).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.models import build_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="fish", choices=["fish", "coral"])
    parser.add_argument("--model", default="mobilenet_v3")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_path = root / "models" / args.task / args.model / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — train the model first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_path = Path(args.out) if args.out else ckpt_path.parent / "best.onnx"
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        out = session.run(None, {"image": dummy.numpy()})[0]
        ref = model(dummy).detach().numpy()
        max_diff = float(abs(out - ref).max())
        print(f"onnxruntime check: max |ONNX - torch| = {max_diff:.2e}")
    except ImportError:
        print("onnxruntime not installed — skipping parity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
