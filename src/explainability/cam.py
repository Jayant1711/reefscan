"""Class Activation Maps implemented from first principles (no external CAM lib).

All methods share a common contract:

    cam = method.generate(input_tensor, class_idx=None) -> np.ndarray [H_in, W_in] in [0, 1]

References
----------
- GradCAM      : Selvaraju et al., ICCV 2017
- GradCAM++    : Chattopadhyay et al., ICCVW 2018
- ScoreCAM     : Wang et al., CVPR 2020
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import get_logger

logger = get_logger("xai.cam")


class CAMMethod(ABC):
    """Base class handling activation capture on a target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._save_activation)
        self.model.eval()

    def _save_activation(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        # Detach-free storage: gradients must flow back through activations.
        self._activations = output

    def release(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        cam = cam - cam.min()
        mx = cam.max()
        return cam / (mx + 1e-8) if mx > 1e-8 else np.zeros_like(cam)

    def _final_cam(self, weighted: torch.Tensor) -> np.ndarray:
        """ReLU -> spatial upsample -> [0, 1] numpy."""
        cam = F.relu(weighted.detach())
        if cam.dim() == 2:          # (h, w) -> (1, 1, h, w)
            cam = cam[None, None]
        elif cam.dim() == 3:        # (1, h, w) -> (1, 1, h, w)
            cam = cam.unsqueeze(0)
        cam = F.interpolate(cam, size=self.input_size, mode="bilinear", align_corners=False)[0, 0]
        return self._normalize(cam.cpu().numpy())

    @abstractmethod
    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute the CAM for ``class_idx`` (default: argmax prediction)."""

    # Context manager sugar
    def __enter__(self) -> "CAMMethod":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class GradCAM(CAMMethod):
    """Gradient-weighted CAM: w_k = GAP(dY/dA_k); cam = ReLU(sum_k w_k A_k)."""

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.input_size = tuple(input_tensor.shape[-2:])
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]

        grads = torch.autograd.grad(score, self._activations, retain_graph=True)[0]  # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)                               # GAP
        weighted = (weights * self._activations).sum(dim=1, keepdim=False)[0]
        return self._final_cam(weighted)


class GradCAMPlusPlus(CAMMethod):
    """Second-order refinement of GradCAM weights (Chattopadhyay et al., 2018).

    alpha_kl = g_kl^2 / (2 g_kl^2 + sum_ab(A_kab) * g_kl^3 + eps)
    w_k      = sum_l ReLU(g_kl) * alpha_kl
    """

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.input_size = tuple(input_tensor.shape[-2:])
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]

        grads = torch.autograd.grad(score, self._activations, retain_graph=True)[0]
        acts_sum = self._activations.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        eps = 1e-6
        alpha = grads_power_2 / (2 * grads_power_2 + acts_sum * grads_power_3 + eps)
        weights = (torch.clamp(grads, min=0.0) * alpha).sum(dim=(2, 3), keepdim=True)
        weighted = (weights * self._activations).sum(dim=1)[0]
        return self._final_cam(weighted)


class ScoreCAM(CAMMethod):
    """Perturbation-based CAM (Wang et al., 2020).

    Each normalised activation map masks the input; the resulting score drop
    gives the channel weight. Batched and optionally channel-capped so it is
    tractable on CPU.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 batch_size: int = 32, max_channels: int | None = None) -> None:
        super().__init__(model, target_layer)
        self.batch_size = max(1, batch_size)
        self.max_channels = max_channels

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.input_size = tuple(input_tensor.shape[-2:])
        with torch.no_grad():
            logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        acts = self._activations.detach()                     # (1, C, h, w)
        n_channels = acts.size(1)
        if self.max_channels is not None and n_channels > self.max_channels:
            # Keep the most "active" channels by energy for CPU tractability.
            energy = acts.flatten(2).pow(2).sum(dim=2)[0]
            keep = torch.topk(energy, self.max_channels).indices
            acts = acts[:, keep]
            n_channels = self.max_channels

        # Normalise each channel map then ReLU.
        flat = acts.flatten(2)
        mn = flat.min(dim=2, keepdim=True).values
        mx = flat.max(dim=2, keepdim=True).values
        norm = F.relu((flat - mn) / (mx - mn + 1e-8)).view_as(acts)

        # Upsample every channel map to input resolution -> masking masks.
        masks = F.interpolate(norm, size=self.input_size, mode="bilinear", align_corners=False)
        base = input_tensor.detach()

        scores: list[float] = []
        for start in range(0, n_channels, self.batch_size):
            chunk = masks[0, start:start + self.batch_size]           # (Bc, H, W)
            masked = base * chunk.unsqueeze(1)                        # (Bc,3,H,W) broadcast over RGB
            with torch.no_grad():
                out = self.model(masked)
            scores.extend(out[:, class_idx].cpu().tolist())

        weights = F.softmax(torch.tensor(scores), dim=0)              # (C,)
        weighted = (weights.view(1, -1, 1, 1).to(acts.device) * norm).sum(dim=1)[0]
        return self._final_cam(weighted)
