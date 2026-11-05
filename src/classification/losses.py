"""Loss functions: smoothed/weighted cross-entropy and an ordinal severity loss."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalPenaltyLoss(nn.Module):
    """Cross-entropy plus a distance penalty for ordinal class labels.

    For bleaching severity (Healthy < Mild < Moderate < Severe < Dead),
    confusing 'Healthy' with 'Dead' must hurt more than 'Mild' vs 'Healthy'.

        L = CE(y, ŷ) + lam * mean_c( p_ŷ(c) * |c - y| )

    where ``p_ŷ(c)`` is the predicted probability of class ``c``.
    """

    def __init__(self, num_classes: int, lam: float = 0.5,
                 class_weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.lam = lam
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
        positions = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer("positions", positions.view(1, -1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        target_pos = self.positions.expand(logits.size(0), -1)  # (B, C)
        dist = (self.positions.expand(logits.size(0), -1) - targets.view(-1, 1).float()).abs()
        penalty = (probs * dist).sum(dim=1).mean()
        return ce + self.lam * penalty


def build_loss(num_classes: int, class_weight: torch.Tensor | None = None,
               label_smoothing: float = 0.1, ordinal: bool = False,
               ordinal_lam: float = 0.5) -> nn.Module:
    """Loss factory used by the Trainer."""
    if ordinal:
        return OrdinalPenaltyLoss(num_classes, lam=ordinal_lam, class_weight=class_weight,
                                  label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
