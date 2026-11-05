"""CAM visualisation: overlays, heatmap files, multi-method comparison grids."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from ..utils import ensure_dir
from .cam import CAMMethod, GradCAM, GradCAMPlusPlus, ScoreCAM
from .targets import find_target_layer

METHOD_ALIASES = {
    "gradcam": GradCAM,
    "gradcam_plusplus": GradCAMPlusPlus,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
}


def overlay_cam_on_image(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """Blend a colourised CAM onto the original image (both same size)."""
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam_resized, 0, 1)), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, image_bgr, 1.0 - alpha, 0)


def save_heatmap(cam: np.ndarray, out_path: str | Path) -> Path:
    """Save the raw CAM as a JET-coloured PNG."""
    out_path = ensure_dir(Path(out_path).parent) / Path(out_path).name
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(out_path)
    return out_path


def build_cam_method(name: str, model: torch.nn.Module, arch: str | None = None,
                     scorecam_batch: int = 32, scorecam_max_channels: int | None = None) -> CAMMethod:
    """Instantiate a CAM method by name with auto-detected target layer."""
    key = name.lower()
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unknown CAM method '{name}'. Available: {sorted(set(METHOD_ALIASES))}")
    target_layer = find_target_layer(model, arch)
    if METHOD_ALIASES[key] is ScoreCAM:
        return ScoreCAM(model, target_layer, batch_size=scorecam_batch, max_channels=scorecam_max_channels)
    return METHOD_ALIASES[key](model, target_layer)


def generate_xai_grid(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_bgr: np.ndarray,
    out_dir: str | Path,
    methods: Sequence[str] = ("gradcam", "gradcam_plusplus", "scorecam"),
    class_idx: int | None = None,
    class_names: Sequence[str] | None = None,
    arch: str | None = None,
    overlay_alpha: float = 0.40,
    scorecam_batch: int = 32,
    scorecam_max_channels: int | None = None,
    save_individual: bool = True,
) -> dict[str, Path]:
    """Run all requested CAM methods and save an overlay grid + individual maps.

    Returns a dict: method name -> overlay PNG path, plus ``"grid"``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ensure_dir(out_dir)
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(input_tensor).argmax(dim=1).item())
    label = class_names[class_idx] if class_names and class_idx < len(class_names) else str(class_idx)

    panels: list[np.ndarray] = [cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)]
    titles = [f"Original\npred: {label}"]
    paths: dict[str, Path] = {}

    for method_name in methods:
        method = build_cam_method(method_name, model, arch, scorecam_batch, scorecam_max_channels)
        with method:
            cam = method.generate(input_tensor, class_idx)
        overlay = overlay_cam_on_image(original_bgr, cam, overlay_alpha)
        panels.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        titles.append(method_name.replace("_", " ").replace("gradcam++", "GradCAM++").title())
        if save_individual:
            paths[method_name] = save_heatmap(cam, out / f"cam_{method_name}.png")

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8))
    axes = np.atleast_1d(axes)
    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    grid_path = out / "xai_grid.png"
    fig.tight_layout()
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["grid"] = grid_path
    return paths
