"""Data cleaning tests: corrupt scan, dedup, leakage, seg mask-area filter."""

from __future__ import annotations

import shutil

import numpy as np
from PIL import Image

from src.data.cleaning import (
    apply_quarantine,
    dhash_of,
    md5_of,
    scan_imagefolder,
    scan_segmentation_pairs,
)


def _write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def _gradient(w=96, h=96, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_dhash_stable_under_recompression(tmp_path):
    # Smooth gradient — random noise is the pathological case for JPEG + dHash.
    x = np.linspace(0, 255, 96, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    arr = np.stack([xx, yy, (xx + yy) / 2], axis=-2).astype(np.uint8).squeeze()
    arr = np.clip(np.stack([xx, yy, (xx + yy) / 2], axis=-1), 0, 255).astype(np.uint8)
    a = _write(tmp_path / "a.png", arr)
    b = _write(tmp_path / "b.jpg", arr)  # same pixels, different codec
    ha, hb = dhash_of(a), dhash_of(b)
    assert ha and hb
    assert ha == hb, "dHash must be identical across JPEG/PNG of the same pixels"


def test_md5_detects_exact_duplicates(tmp_path):
    a = _write(tmp_path / "train" / "A" / "a.png", _gradient(seed=2))
    b = _write(tmp_path / "train" / "A" / "b.png", _gradient(seed=2))
    assert md5_of(a) == md5_of(b)


def test_scan_flags_corrupt_small_duplicates(tmp_path):
    _write(tmp_path / "train" / "A" / "good1.png", _gradient(seed=3))
    _write(tmp_path / "train" / "A" / "good2.png", _gradient(seed=4))
    _write(tmp_path / "train" / "A" / "dup.png", _gradient(seed=3))       # duplicate of good1
    (tmp_path / "train" / "A" / "corrupt.png").write_bytes(b"not an image")
    _write(tmp_path / "train" / "A" / "tiny.png", np.zeros((8, 8, 3), np.uint8))  # < 32px

    report = scan_imagefolder(tmp_path)
    counts = report["counts"]
    assert counts["corrupt"] == 1
    assert counts["too_small"] == 1
    # good1+dup form one duplicate group -> 1 redundant image
    assert counts["duplicate_images"] == 1
    assert "corrupt.png" in report["corrupt"][0]


def test_leakage_same_image_across_splits(tmp_path):
    leaked = _gradient(seed=5)
    _write(tmp_path / "train" / "A" / "x.png", leaked)
    _write(tmp_path / "test" / "A" / "x_copy.png", leaked)
    _write(tmp_path / "train" / "A" / "unique.png", _gradient(seed=6))
    _write(tmp_path / "test" / "A" / "unique2.png", _gradient(seed=7))

    report = scan_imagefolder(tmp_path)
    assert len(report["leakage_groups"]) == 1
    paths = report["leakage_groups"][0]["paths"]
    assert len(paths) == 2 and {"train", "test"} == set(report["leakage_groups"][0]["splits"])


def test_quarantine_moves_but_keeps_first_copy(tmp_path):
    leaked = _gradient(seed=5)
    keep = _write(tmp_path / "train" / "A" / "x.png", leaked)
    drop = _write(tmp_path / "test" / "A" / "x_copy.png", leaked)
    _write(tmp_path / "train" / "A" / "unique.png", _gradient(seed=6))
    _write(tmp_path / "test" / "A" / "unique2.png", _gradient(seed=7))

    report = scan_imagefolder(tmp_path)
    moved = apply_quarantine(tmp_path, report)
    assert keep.exists(), "first (train) copy must be kept"
    assert not drop.exists()
    assert any("_quarantine" in m for m in moved)
    quarantined = list((tmp_path / "_quarantine").rglob("*"))
    assert any(p.is_file() for p in quarantined)


def test_scan_skips_quarantined_files(tmp_path):
    _write(tmp_path / "train" / "A" / "x.png", _gradient(seed=8))
    qdir = tmp_path / "_quarantine" / "corrupt"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "bad.png").write_bytes(b"junk")
    report = scan_imagefolder(tmp_path)
    assert report["counts"]["scanned"] == 1


def test_seg_pair_scan_flags_empty_mask(tmp_path):
    img = _gradient(seed=9)
    img_path = _write(tmp_path / "DeepFish" / "img" / "f1.png", img)
    mask_path = _write(tmp_path / "DeepFish" / "masks" / "f1_mask.png",
                       np.zeros((96, 96), dtype=np.uint8))     # empty
    img2 = _write(tmp_path / "DeepFish" / "img" / "f2.png", _gradient(seed=10))
    mask2 = _write(tmp_path / "DeepFish" / "masks" / "f2_mask.png",
                   np.zeros((96, 96), np.uint8) | 255)          # full foreground

    report = scan_segmentation_pairs(tmp_path / "DeepFish")
    assert report["pairs_found"] == 2
    assert report["counts"]["empty_masks"] == 1
    assert str(mask_path) in report["empty_masks"][0]
    assert img_path.exists() and mask_path.exists() and img2.exists() and mask2.exists()


def test_missing_root_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        scan_imagefolder(tmp_path / "does_not_exist")
    shutil.rmtree(tmp_path, ignore_errors=True)
