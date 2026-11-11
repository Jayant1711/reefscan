#!/usr/bin/env python
"""ReefScan AI — platform pre-flight validator (run BEFORE training cells).

Checks environment, disk, dataset integrity and config sanity on the target
platform and prints a PASS/WARN/FAIL table. Exit code: 0 = no FAIL (warnings
allowed), 1 = at least one FAIL, 2 = usage/crash error.

Examples
--------
    python scripts/preflight.py --platform kaggle --task coral
    python scripts/preflight.py --platform colab --task fish --with-pytest
    python scripts/preflight.py --full          # all dataset checks
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, merge_config_file, override_from_args  # noqa: E402
from src.utils.provenance import detect_platform, env_versions, git_sha, utc_now_iso  # noqa: E402

MIN_DISK_REPO_GB = 3.0
MIN_DISK_REPO_WARN_GB = 10.0
MIN_DISK_DATA_GB = 8.0
MIN_DISK_DATA_WARN_GB = 30.0
GB = 1024 ** 3


class Check:
    def __init__(self, check_id: str, status: str, detail: str) -> None:
        self.check_id = check_id
        self.status = status  # PASS | WARN | FAIL
        self.detail = detail


def _gb(free_bytes: int) -> float:
    return free_bytes / GB


def _disk(checks: list[Check], cid: str, path: Path, warn_gb: float, fail_gb: float) -> None:
    try:
        free = _gb(shutil.disk_usage(str(path)).free)
    except Exception as exc:
        checks.append(Check(cid, "WARN", f"disk probe failed at {path}: {exc}"))
        return
    if free < fail_gb:
        checks.append(Check(cid, "FAIL", f"{path}: {free:.1f} GB free (< {fail_gb} GB)"))
    elif free < warn_gb:
        checks.append(Check(cid, "WARN", f"{path}: {free:.1f} GB free (< {warn_gb} GB recommended)"))
    else:
        checks.append(Check(cid, "PASS", f"{path}: {free:.1f} GB free"))


def _class_counts(image_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        n = sum(1 for f in class_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if n:
            counts[class_dir.name] = n
    return counts


def _dataset_check(checks: list[Check], cid: str, data_dir: Path, min_per_class: int,
                   max_imbalance_warn: float = 10.0, max_imbalance_fail: float = 50.0) -> None:
    if not data_dir.exists():
        checks.append(Check(cid, "FAIL", f"{data_dir} missing — run download_data.py first"))
        return
    problems: list[str] = []
    notes: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        counts = _class_counts(split_dir)
        if counts:
            split_counts[split] = counts
        if not counts:
            problems.append(f"{split}: no class images")
    if not split_counts:
        checks.append(Check(cid, "FAIL", f"no populated splits under {data_dir}"))
        return
    train = split_counts.get("train", {})
    if train:
        min_class = min(train, key=train.get)
        imbalance = max(train.values()) / max(1, min(train.values()))
        if min(train.values()) == 0 or imbalance > max_imbalance_fail:
            problems.append(f"train imbalance {imbalance:.0f}x (min '{min_class}'={min(train.values())})")
        elif imbalance > max_imbalance_warn or min(train.values()) < min_per_class:
            notes.append(f"train min '{min_class}'={min(train.values())}, imbalance {imbalance:.1f}x")
    for problem in problems:
        checks.append(Check(cid, "FAIL", problem))
    if not problems:
        detail = "; ".join(
            f"{split}: {sum(c.values())} imgs / {len(c)} classes" for split, c in split_counts.items()
        )
        status = "WARN" if notes else "PASS"
        if notes:
            detail += " | " + "; ".join(notes)
        checks.append(Check(cid, status, detail))


def run_preflight(platform: str | None, task: str, with_pytest: bool, config_path: str | None,
                  extra_sets: list[str]) -> tuple[list[Check], dict]:
    checks: list[Check] = []

    # PY
    import sys

    v = sys.version_info
    checks.append(Check("PY", "PASS" if v >= (3, 10) else "FAIL",
                        f"python {sys.version.split()[0]}"))

    # IMPORTS
    missing = []
    for mod in ("torch", "torchvision", "cv2", "PIL", "numpy", "pandas", "sklearn",
                "matplotlib", "yaml", "tqdm", "streamlit", "reportlab"):
        try:
            __import__(mod)
        except ImportError as exc:
            missing.append(f"{mod} ({exc})")
    checks.append(Check("IMPORTS", "FAIL" if missing else "PASS",
                        "all core imports ok" if not missing else f"missing: {missing}"))

    # GPU
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / GB
            checks.append(Check("GPU", "PASS" if vram_gb >= 8 else "WARN",
                                f"{props.name}, {vram_gb:.1f} GB VRAM"))
        else:
            checks.append(Check("GPU", "WARN", "no CUDA device — training on CPU (expect ~10x slower)"))
    except Exception as exc:
        checks.append(Check("GPU", "FAIL", f"torch probe failed: {exc}"))

    # Config + platform merge (same order as train.py)
    cfg = load_config(config_path)
    if platform:
        overlay = ROOT / "configs" / "platform" / f"{platform}.yaml"
        if overlay.exists():
            merge_config_file(cfg, overlay)

    # DISK
    _disk(checks, "DISK_REPO", ROOT, MIN_DISK_REPO_WARN_GB, MIN_DISK_REPO_GB)
    data_root = Path(cfg.paths.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    _disk(checks, "DISK_DATA", data_root, MIN_DISK_DATA_WARN_GB, MIN_DISK_DATA_GB)

    # WRITABLE
    unwritable = []
    for key in ("results_dir", "reports_dir", "models_dir"):
        probe = Path(cfg.paths[key]) / ".preflight_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            unwritable.append(f"{key}: {exc}")
    checks.append(Check("WRITABLE", "FAIL" if unwritable else "PASS",
                        "results/reports/models dirs writable" if not unwritable else str(unwritable)))

    # DATASETS (respect --task)
    tasks = ["fish", "coral", "seg"] if task == "all" else [task]
    if "coral" in tasks:
        _dataset_check(checks, "DS_CORAL", Path(cfg.coral.data_dir), min_per_class=50)
    if "fish" in tasks:
        _dataset_check(checks, "DS_FISH", Path(cfg.fish.data_dir),
                       min_per_class=int(cfg.fish.prepare.get("min_images_per_class", 100)))
    if "seg" in tasks:
        from src.segmentation.engine_seg import discover_pairs

        seg_dir = Path(cfg.seg.data_dir)
        if not seg_dir.exists():
            checks.append(Check("DS_SEG", "FAIL", f"{seg_dir} missing — run download_data.py --dataset deepfish"))
        else:
            pairs = discover_pairs(seg_dir)
            if not pairs:
                checks.append(Check("DS_SEG", "FAIL", f"0 image/mask pairs under {seg_dir}"))
            elif len(pairs) < 500:
                checks.append(Check("DS_SEG", "WARN", f"{len(pairs)} pairs (< 500 — full release has ~620)"))
            else:
                checks.append(Check("DS_SEG", "PASS", f"{len(pairs)} image/mask pairs"))

    # CLEANING reports
    clean_notes = []
    for name in ("coral", "fishnet", "seg"):
        if name == "fishnet" and "fish" not in tasks:
            continue
        if name in ("coral", "seg") and name not in tasks and task != "all":
            continue
        report = Path(cfg.paths.results_dir) / "cleaning" / f"{name}_cleaning_report.json"
        if not report.exists():
            clean_notes.append(f"{name}: no report (run `python clean_data.py --dataset {name}`)")
            continue
        try:
            counts = json.loads(report.read_text(encoding="utf-8")).get("counts", {})
            if counts.get("corrupt", 0) or counts.get("leaked_images", 0):
                clean_notes.append(f"{name}: corrupt={counts.get('corrupt')} leaked={counts.get('leaked_images')}")
        except Exception as exc:
            clean_notes.append(f"{name}: unreadable report ({exc})")
    if clean_notes:
        checks.append(Check("CLEAN", "WARN", "; ".join(clean_notes)))
    else:
        checks.append(Check("CLEAN", "PASS", "cleaning reports present, no corrupt/leaked flags"))

    # CONFIG sanity
    cfg_problems = []
    thresholds = list(cfg.coral.severity.thresholds)
    if thresholds != sorted(thresholds) or len(thresholds) != 4:
        cfg_problems.append(f"severity.thresholds not 4 ascending: {thresholds}")
    lam = float(cfg.coral.severity.fusion_lambda)
    if not 0 <= lam <= 1:
        cfg_problems.append(f"fusion_lambda out of [0,1]: {lam}")
    fractions = [float(f) for f in cfg.fish.prepare.split_fractions]
    if abs(sum(fractions) - 1.0) > 1e-6:
        cfg_problems.append(f"fish split_fractions sum {sum(fractions):.4f} != 1.0")
    if int(cfg.fish.image_size) % 32:
        cfg_problems.append(f"fish.image_size {cfg.fish.image_size} not divisible by 32")
    if int(cfg.seg.image_size) % 16:
        cfg_problems.append(f"seg.image_size {cfg.seg.image_size} not divisible by 16")
    checks.append(Check("CONFIG", "FAIL" if cfg_problems else "PASS",
                        "thresholds/lambda/splits/sizes sane" if not cfg_problems else "; ".join(cfg_problems)))

    # PYTEST (optional)
    if with_pytest:
        import subprocess

        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=str(ROOT), check=False)
        checks.append(Check("PYTEST", "PASS" if r.returncode == 0 else "FAIL",
                            f"exit code {r.returncode}"))

    versions = env_versions()
    report = {
        "schema": "reefscan-preflight-v1",
        "recorded_at": utc_now_iso(),
        "platform": detect_platform(),
        "git_sha": git_sha(ROOT),
        "versions": versions,
        "checks": [{"id": c.check_id, "status": c.status, "detail": c.detail} for c in checks],
        "result": "FAIL" if any(c.status == "FAIL" for c in checks) else "PASS",
    }
    return checks, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default=None, choices=["kaggle", "colab", "lightning", "local"])
    parser.add_argument("--task", default="all", choices=["all", "fish", "coral", "seg"])
    parser.add_argument("--with-pytest", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--json", action="store_true", help="Print the JSON report too")
    args = parser.parse_args()

    checks, report = run_preflight(args.platform, args.task, args.with_pytest, args.config, args.set)

    out_dir = ROOT / "results" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    width = max(len(c.check_id) for c in checks)
    line = "=" * 76
    print(line)
    print(f" ReefScan AI — pre-flight checks  {report['recorded_at']}  "
          f"platform={report['platform']}  git_sha={report['git_sha']}")
    print(line)
    for c in checks:
        print(f" {c.check_id:<{width}}  {c.status:<5}  {c.detail}")
    fails = sum(1 for c in checks if c.status == "FAIL")
    warns = sum(1 for c in checks if c.status == "WARN")
    passes = sum(1 for c in checks if c.status == "PASS")
    print(line)
    verdict = "DO NOT START TRAINING (fix FAIL rows above)" if fails else "READY TO TRAIN"
    print(f" RESULT: {fails} FAIL, {warns} WARN, {passes} PASS  ->  {verdict}")
    print(line)
    print(f" JSON report -> {report_path}")
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
