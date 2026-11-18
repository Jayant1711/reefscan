"""Device-parity regression tests.

Guards the 2026-11-19 Kaggle T4 incident: ``OrdinalPenaltyLoss.positions``
(a registered buffer) stayed on CPU because the Trainer never moved its
criterion to the training device. Every tensor behind the training step must
live on ``Trainer.device``.

These tests only RUN where CUDA exists — which is exactly the environment the
bug breaks in (the Kaggle pytest gate on the T4). On CPU-only boxes the same
bug is unobservable (everything is CPU), so they skip; local WDAC-flaky
machines therefore never see them.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

CUDA = torch.cuda.is_available()


def _tiny_loader(n: int = 8, features: int = 8, classes: int = 2) -> DataLoader:
    xs = torch.randn(n, features)
    ys = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(xs, ys), batch_size=4)


@pytest.mark.skipif(not CUDA, reason="device-parity needs a real second device (CUDA)")
def test_trainer_criterion_buffers_follow_training_device(tmp_path, monkeypatch):
    """Criterion params AND buffers must sit on the training device, and one
    real GPU training step must complete end-to-end with ordinal loss on."""
    import src.classification.engine as eng

    # Swap the torchvision backbone for a tiny Linear so the test needs no
    # pretrained-weight download and stays fast/offline-safe in the gate.
    monkeypatch.setattr(
        eng, "build_model",
        lambda name, n, freeze_backbone=False: torch.nn.Linear(8, n),
    )

    cfg = eng.TrainConfig(
        epochs=1,
        warmup_epochs=0,
        ordinal_loss=True,          # the exact loss that broke on the T4
        ordinal_lam=0.5,
        strategy="feature_extraction",
        amp=False,
        scheduler="step",
        grad_clip=5.0,
    )
    device = torch.device("cuda")
    trainer = eng.Trainer(
        "resnet34", 2, _tiny_loader(), None, cfg, ["a", "b"], device, tmp_path,
        class_weight=torch.tensor([1.0, 2.0]),
    )

    for name, buf in trainer.criterion.named_buffers():
        assert buf.device == device, f"criterion buffer '{name}' left on {buf.device}"
    for name, p in trainer.criterion.named_parameters():
        assert p.device == device, f"criterion param '{name}' left on {p.device}"
    # The class-weight tensor passed into CrossEntropyLoss too.
    weight = trainer.criterion.ce.weight
    assert weight.device == device, f"CE class weight left on {weight.device}"

    # End-to-end: the forward/backward that crashed live must now run clean.
    loss, acc, grad_norm = trainer._train_epoch()
    assert torch.isfinite(torch.tensor(loss)), "non-finite training loss"
