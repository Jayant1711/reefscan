"""Phase 2 — Marine species classification (transfer-learning benchmark).

Torch/sklearn-heavy members load lazily; ``models`` is pure-torchvision and
still imported eagerly (no sklearn dependency).
"""

from typing import Any

from .models import SUPPORTED_MODELS, build_model

__all__ = [
    "SUPPORTED_MODELS",
    "build_model",
    "Trainer",
    "evaluate_model_loader",
    "build_dataloaders",
    "count_class_images",
    "run_model_evaluation",
    "build_leaderboard",
]

_LAZY = {
    "Trainer": ".engine",
    "evaluate_model_loader": ".engine",
    "build_dataloaders": ".datasets",
    "count_class_images": ".datasets",
    "run_model_evaluation": ".evaluate",
    "build_leaderboard": ".evaluate",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
