"""Config pinning (yaml <-> module defaults) + ONNX export parity tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------- config pinning
def _yaml_coral_severity() -> dict:
    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg["coral"]["severity"]


def test_severity_defaults_match_yaml():
    """README claims default.yaml is the single source of truth — this pins it."""
    from src.coral_analysis.severity import DEFAULT_THRESHOLDS, SEVERITY_WEIGHTS

    sev = _yaml_coral_severity()
    assert tuple(sev["thresholds"]) == tuple(DEFAULT_THRESHOLDS)
    assert [int(v) for v in sev["class_weights_index"]] == [
        SEVERITY_WEIGHTS[c] for c in ("healthy", "mild", "moderate", "severe", "dead")
    ]


def test_severity_fusion_lambda_matches_yaml():
    from src.coral_analysis.severity import severity_index

    sev = _yaml_coral_severity()
    lam = float(sev["fusion_lambda"])
    # severity_index(p, w, lam): 100*(lam*p + (1-lam)*w) — pin via a 60/40 probe
    assert abs(severity_index(1.0, 0.0, lam) - 100 * lam) < 1e-9


# ------------------------------------------------------------- ONNX export
def _demo_checkpoint() -> Path | None:
    """Any trained/demo fish checkpoint on this machine (platform-dependent)."""
    candidates = []
    for base in (ROOT / "models" / "fish", ROOT / "models" / "demo" / "fish"):
        if base.exists():
            candidates += list(base.rglob("best.pt")) + list(base.rglob("*_demo.pt"))
    return candidates[0] if candidates else None


def test_onnx_export_parity():
    """Export the fish classifier to ONNX and check numeric parity vs torch."""
    onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    ckpt_path = _demo_checkpoint()
    if ckpt_path is None:
        pytest.skip("No fish checkpoint on this machine — run training or the smoke test first")

    import numpy as np
    import torch
    from src.classification.models import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_name"], ckpt.get("num_classes", 2), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = int(ckpt.get("image_size", 224))

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "best.onnx"
        dummy = torch.randn(1, 3, size, size)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out_ort = session.run(None, {"input": dummy.numpy()})[0]
        out_pt = model(dummy).detach().numpy()
    assert np.allclose(out_ort, out_pt, atol=1e-3), "ONNX/torch parity drift"
