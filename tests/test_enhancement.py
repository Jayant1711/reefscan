"""Enhancement strategy + metric unit tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.enhancement import strategies  # noqa: F401  (populates the registry at collection)
from src.enhancement.metrics import psnr, ssim, uciqe, uiqm, evaluate_enhancement
from src.enhancement.registry import ENHANCER_REGISTRY, available, build_enhancers, get_enhancer


def test_registry_contains_all_eight_strategies():
    expected = {"clahe", "dark_channel", "udcp", "white_balance",
                "color_restoration", "gamma", "histogram_eq", "fusion_wcg"}
    assert expected.issubset(set(available()))
    assert set(available()) == expected


@pytest.mark.parametrize("name", sorted(ENHANCER_REGISTRY.keys()))
def test_every_strategy_contract(gradient_bgr, name):
    """All strategies: uint8 output, same shape, valid range, different from input."""
    enhancer = get_enhancer(name)
    out = enhancer.enhance(gradient_bgr)
    assert out.shape == gradient_bgr.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert not np.array_equal(out, gradient_bgr), f"{name} returned the input unchanged"


def test_metrics_identical_images(flat_bgr):
    assert psnr(flat_bgr, flat_bgr) == float("inf")
    assert abs(ssim(flat_bgr, flat_bgr) - 1.0) < 1e-6


def test_metrics_degradation_ordering(gradient_bgr):
    noisy = cv2.add(gradient_bgr, np.full_like(gradient_bgr, 40))
    more_noisy = cv2.add(gradient_bgr, np.full_like(gradient_bgr, 90))
    assert psnr(gradient_bgr, noisy) > psnr(gradient_bgr, more_noisy)
    assert ssim(gradient_bgr, noisy) > ssim(gradient_bgr, more_noisy)


def test_uciqe_positive_and_ordering(gradient_bgr, flat_bgr):
    """Structured colourful image must outscore a flat gray one."""
    assert uciqe(gradient_bgr) > uciqe(flat_bgr)
    assert uiqm(gradient_bgr) > 0


def test_evaluate_enhancement_keys(gradient_bgr):
    out = get_enhancer("clahe").enhance(gradient_bgr)
    m = evaluate_enhancement(gradient_bgr, out)
    assert {"psnr", "ssim", "uciqe", "uiqm", "uciqe_orig", "uiqm_orig"} <= set(m)


def test_selector_selects_best_strategy(gradient_bgr, tmp_path):
    from src.enhancement.selector import EnhancementSelector

    selector = EnhancementSelector()
    best, df, grid = selector.select_best([gradient_bgr], out_dir=tmp_path)
    assert best in ENHANCER_REGISTRY
    assert len(df) == len(ENHANCER_REGISTRY)
    assert (tmp_path / "enhancement_comparison.png").exists()
    assert df["score"].idxmax() == 0  # sorted descending


def test_guided_filter_shape(gradient_bgr):
    from src.enhancement.dark_channel import _guided_filter

    guide = np.random.rand(64, 64).astype(np.float32)
    src = np.random.rand(64, 64).astype(np.float32)
    out = _guided_filter(guide, src)
    assert out.shape == (64, 64)
