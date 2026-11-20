"""prepare_f4k head-cap: the F4K long tail is 84x (largest ~9.6K vs 114), which
fails the preflight imbalance gate (FAIL > 50x) and starves head epochs even
under the balanced sampler. Head classes are capped at max_images_per_class by
dropping WHOLE video-segment groups deterministically; tail classes keep every
image. Live-caught on the Kaggle T4, 2026-11-20 (DS_FISH FAIL 84x)."""

from __future__ import annotations

from pathlib import Path

from src.data.prepare_f4k import prepare_f4k_subset


def _build_raw(root: Path, class_name: str, n_groups: int, per_group: int) -> None:
    d = root / "fishRecognition_GT" / "fish_image" / class_name
    d.mkdir(parents=True, exist_ok=True)
    for g in range(n_groups):
        for j in range(per_group):
            # stem "fish_<group>_<idx>" -> video group "fish_<group>"
            (d / f"fish_g{g:03d}_{j:03d}.png").write_bytes(b"png")


def _count(out: Path, split: str, cls: str) -> int:
    return sum(1 for _ in (out / split / cls).glob("*.png"))


def test_head_class_capped_by_whole_groups(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "curated"
    _build_raw(raw, "fish_01", n_groups=10, per_group=250)  # 2500 images
    _build_raw(raw, "fish_02", n_groups=1, per_group=114)   # tail: 114

    prepare_f4k_subset(raw, out, min_images_per_class=100, top_k=2,
                       max_images_per_class=1000)

    # 4 whole groups of 250 = exactly 1000; the 5th group would overshoot
    # nothing (250*5=1250 > 1000 is allowed by the while-kept<cap greedy, so
    # assert the invariant: capped at cap when a group fits, never below one
    # group) — the deterministic guarantee is: 0 < kept <= cap when any group
    # fits, and every kept image comes from whole groups.
    kept = sum(_count(out, s, "fish_01") for s in ("train", "validation", "test"))
    assert 0 < kept <= 1000, kept
    assert kept % 250 == 0, "cap must drop whole video groups only"
    # Tail class untouched.
    assert sum(_count(out, s, "fish_02") for s in ("train", "validation", "test")) == 114
    assert (out / "_COMPLETE").exists()


def test_force_rebuilds_past_complete_marker(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "curated"
    _build_raw(raw, "fish_01", n_groups=3, per_group=100)
    prepare_f4k_subset(raw, out, min_images_per_class=100, top_k=1,
                       max_images_per_class=1000)
    (out / "_COMPLETE").write_text("stale\n", encoding="utf-8")
    # A stale file from a previous (uncapped) build must NOT survive a forced
    # rebuild — copying over it produced a 28K-file hybrid on the live run.
    stale = out / "train" / "fish_01" / "stale_leftover.png"
    stale.write_bytes(b"png")
    # Without force: skip. With force: purge + rebuild (marker rewritten).
    p1 = prepare_f4k_subset(raw, out, min_images_per_class=100, top_k=1,
                            max_images_per_class=1000)
    assert p1.exists() and stale.exists(), "skip path must not purge"
    p2 = prepare_f4k_subset(raw, out, min_images_per_class=100, top_k=1,
                            max_images_per_class=1000, force=True)
    assert (p2 / "_COMPLETE").exists()
    assert not stale.exists(), "force rebuild must purge stale output"


def test_uncapped_class_below_cap_untouched(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "curated"
    _build_raw(raw, "fish_01", n_groups=2, per_group=150)  # 300 < cap
    prepare_f4k_subset(raw, out, min_images_per_class=100, top_k=1,
                       max_images_per_class=1000)
    assert sum(_count(out, s, "fish_01")
               for s in ("train", "validation", "test")) == 300
