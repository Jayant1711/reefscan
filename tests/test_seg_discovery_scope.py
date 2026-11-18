"""discover_pairs must scope to the official DeepFish Segmentation split.

Live-caught on Kaggle (2026-11-19): the full DeepFish tar ships
Localization/{images,masks}/{empty,valid} — 3,200 extra pairs whose
fish-present masks are unreliable (all-black PNGs on fish-class frames).
Pairing the whole tree discovered 3,210 pairs instead of the official 620,
so E3 would have trained on diluted, partly-wrong targets.
"""

from __future__ import annotations

from pathlib import Path

from src.segmentation.engine_seg import discover_pairs


def _make_pair(root: Path, kind_dir: str, split: str, stem: str) -> None:
    img = root / kind_dir / "images" / split / f"{stem}.jpg"
    mask = root / kind_dir / "masks" / split / f"{stem}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"fake-jpeg")
    mask.write_bytes(b"fake-png")


def test_discovery_scopes_to_segmentation_subdir(tmp_path: Path) -> None:
    # Official split: 2 real pairs under Segmentation/
    _make_pair(tmp_path, "Segmentation", "train", "fish_a")
    _make_pair(tmp_path, "Segmentation", "val", "fish_b")
    # Localization junk: 3 extra pairs the tar ships alongside the split
    _make_pair(tmp_path, "Localization", "empty", "no_fish_1")
    _make_pair(tmp_path, "Localization", "valid", "fish_c")
    _make_pair(tmp_path, "Localization", "valid", "fish_d")

    pairs = discover_pairs(tmp_path)
    stems = {img.stem.lower() for img, _ in pairs}
    assert len(pairs) == 2, f"expected official 2 pairs, got {len(pairs)}: {stems}"
    assert stems == {"fish_a", "fish_b"}


def test_discovery_still_walks_bare_trees(tmp_path: Path) -> None:
    # No Segmentation/ subdir (generic folder tree): keep the old behaviour.
    _make_pair(tmp_path, "custom", "train", "frame_1")
    pairs = discover_pairs(tmp_path)
    assert len(pairs) == 1
