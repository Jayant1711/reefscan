"""Explainability tests: CAM methods produce valid, meaningful maps.

Two model regimes are exercised:
- ``random_model``  : random init — checks structural contracts and class sensitivity.
- ``positive_model``: deterministic positive classifier — guarantees non-degenerate CAMs
  (a random net can legitimately produce all-negative weightings -> ReLU -> empty map).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from src.explainability.targets import find_target_layer


class TinyCNN(nn.Module):
    """3x3-conv net with a single conv feature block for fast CAM tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@pytest.fixture(scope="module")
def random_model():
    torch.manual_seed(0)
    model = TinyCNN()
    model.eval()
    return model


@pytest.fixture(scope="module")
def positive_model():
    """All-positive classifier rows scaled per class -> non-degenerate CAMs."""
    torch.manual_seed(3)
    model = TinyCNN()
    with torch.no_grad():
        linear = model.classifier[1]
        linear.weight.fill_(0.0)
        for k in range(4):
            linear.weight[k, :] = 0.01 * (k + 1)          # positive, class-dependent scale
        linear.bias.zero_()
    model.eval()
    return model


@pytest.fixture
def input_tensor():
    torch.manual_seed(1)
    return torch.rand(1, 3, 64, 64)


ALL_METHODS = [
    (GradCAM, {}),
    (GradCAMPlusPlus, {}),
    (ScoreCAM, {"batch_size": 4, "max_channels": 4}),
]


@pytest.mark.parametrize("cls,kwargs", ALL_METHODS)
def test_cam_shape_and_range(positive_model, input_tensor, cls, kwargs):
    method = cls(positive_model, find_target_layer(positive_model), **kwargs)
    with method:
        cam = method.generate(input_tensor, class_idx=0)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert cam.max() > 0.0  # non-degenerate with positive weights


def test_cam_matches_manual_gradcam_reference(random_model, input_tensor):
    """GradCAM output must equal a direct manual recomputation."""
    target_layer = find_target_layer(random_model)
    acts = {}

    def hook(_m, _i, out):
        acts["a"] = out

    handle = target_layer.register_forward_hook(hook)
    logits = random_model(input_tensor)
    a = acts["a"]
    score = logits[0, 0]
    grads = torch.autograd.grad(score, a, retain_graph=False)[0]
    weights = grads.mean(dim=(2, 3))
    manual = torch.relu((weights[..., None, None] * a).sum(dim=1))[0]
    handle.remove()

    manual_up = torch.nn.functional.interpolate(
        manual[None, None], size=(64, 64), mode="bilinear", align_corners=False
    )[0, 0].detach().numpy()
    manual_np = (manual_up - manual_up.min()) / (manual_up.max() - manual_up.min() + 1e-8)

    cam_method = GradCAM(random_model, target_layer)
    with cam_method:
        produced = cam_method.generate(input_tensor, class_idx=0)
    assert np.allclose(produced, manual_np, atol=1e-5)


def test_gradcam_targets_predicted_class(random_model, input_tensor):
    """CAM for the predicted class should differ from CAM for another class."""
    with torch.no_grad():
        pred = int(random_model(input_tensor).argmax())
    cam_method = GradCAM(random_model, find_target_layer(random_model))
    with cam_method:
        cam_pred = cam_method.generate(input_tensor, class_idx=pred)
        cam_other = cam_method.generate(input_tensor, class_idx=(pred + 1) % 4)
    assert not np.allclose(cam_pred, cam_other)


def test_scorecam_matches_gradcam_support(positive_model, input_tensor):
    """ScoreCAM and GradCAM should both highlight something (correlation > 0)."""
    with GradCAM(positive_model, find_target_layer(positive_model)) as g:
        cam_g = g.generate(input_tensor, class_idx=0)
    with ScoreCAM(positive_model, find_target_layer(positive_model), batch_size=4, max_channels=4) as s:
        cam_s = s.generate(input_tensor, class_idx=0)
    corr = np.corrcoef(cam_g.ravel(), cam_s.ravel())[0, 1]
    assert corr > 0.0


def test_overlay_and_save(tmp_path):
    from src.explainability.visualize import overlay_cam_on_image, save_heatmap

    cam = np.random.rand(32, 32)
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    overlay = overlay_cam_on_image(image, cam)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8

    path = save_heatmap(cam, tmp_path / "heat.png")
    assert path.exists() and path.stat().st_size > 0
