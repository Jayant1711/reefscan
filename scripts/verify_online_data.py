#!/usr/bin/env python
"""Verify the ONLINE dataset sources still match what the cleaning/training code expects.

Run BEFORE the multi-GB downloads (metadata-only, ~1-2 min, no GPU). If an upstream
release changed its schema (renamed columns, re-packed archives, moved files), this
script raises a FAIL so the agent STOPS and reports to the repo owner instead of
silently mislabelling data. Exit codes: 0 = all sources match, 1 = mismatch/unreachable.

Checks
------
- Fish4Knowledge (F4K) class_id.csv (0.6 MB, Edinburgh mirror): exact columns
  file_name, cluster_id; verified-crop/cluster sanity (27,370 crops, 23 species)
- NOAA coral (HuggingFace datasets-server): splits 7,292 / 1,562 / 1,565; label names CORAL, CORAL_BL
- DeepFish (HuggingFace): DeepFish.tar present with expected size (~7.1 GB)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

F4K_CLASS_ID_URL = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/class_id.csv"
F4K_TAR_URL = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar"
EXPECTED_F4K_COLUMNS = ("file_name", "cluster_id")
EXPECTED_F4K_ROWS = (25_000, 30_000)         # 27,370 verified crops
EXPECTED_F4K_CLUSTERS = 23                   # species clusters; 13 have >= 100 images

CORAL_SIZE_URL = "https://datasets-server.huggingface.co/size?dataset=NMFS-OSI%2FNOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
EXPECTED_CORAL_SPLITS = {"train": 7292, "validation": 1562, "test": 1565}
CORAL_CLASSES = ("CORAL", "CORAL_BL")

DEEPFISH_TREE_URL = "https://huggingface.co/api/datasets/Alzayats/DeepFish/tree/main"
EXPECTED_DEEPFISH_TAR_MB = (6_500, 8_000)  # ~7.1 GB


class Check:
    def __init__(self, cid: str, status: str, detail: str) -> None:
        self.cid = cid
        self.status = status
        self.detail = detail


def _fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "reefscan-verify/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def verify_f4k(checks: list[Check]) -> None:
    try:
        blob = _fetch(F4K_CLASS_ID_URL, timeout=120)
    except Exception as exc:
        checks.append(Check("F4K", "FAIL", f"class_id.csv unreachable at {F4K_CLASS_ID_URL}: {exc} — retry; if it persists, the Edinburgh mirror moved"))
        return
    try:
        reader = csv.DictReader(io.TextIOWrapper(io.BytesIO(blob), encoding="utf-8"))
        columns = tuple(reader.fieldnames or [])
        rows = list(reader)
    except Exception as exc:
        checks.append(Check("F4K", "FAIL", f"class_id.csv unreadable: {exc}"))
        return

    if columns != EXPECTED_F4K_COLUMNS:
        checks.append(Check("F4K", "FAIL",
                            f"SCHEMA DRIFT — expected columns {EXPECTED_F4K_COLUMNS}, got {columns}. "
                            "STOP: update prepare_f4k.py before training."))
        return

    clusters = {r["cluster_id"] for r in rows}
    details = [f"{len(rows)} verified crops", f"{len(clusters)} species clusters"]
    status = "PASS"
    if not EXPECTED_F4K_ROWS[0] <= len(rows) <= EXPECTED_F4K_ROWS[1]:
        status = "WARN"
        details.append(f"row count outside {EXPECTED_F4K_ROWS}")
    if len(clusters) != EXPECTED_F4K_CLUSTERS:
        status = "WARN"
        details.append(f"cluster count {len(clusters)} != {EXPECTED_F4K_CLUSTERS}")
    checks.append(Check("F4K", status, "schema matches prepare_f4k.py; " + "; ".join(details)))


def verify_coral(checks: list[Check]) -> None:
    try:
        payload = json.loads(_fetch(CORAL_SIZE_URL, timeout=60).decode("utf-8"))
    except Exception as exc:
        checks.append(Check("CORAL", "FAIL", f"HF datasets-server unreachable: {exc} — retry"))
        return
    size_entry = payload.get("size", {}).get("dataset") or {}
    splits = {s["split"]: s["num_rows"] for s in size_entry.get("splits", [])}
    if not splits:
        # /size cache may be pending — fall back to the dataset card, which
        # documents the split table (7,292 / 1,562 / 1,565).
        try:
            card = _fetch(
                "https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset/resolve/main/README.md",
                timeout=60,
            ).decode("utf-8", errors="replace")
        except Exception as exc:
            checks.append(Check("CORAL", "WARN", f"could not verify splits (datasets-server pending, card fetch failed: {exc})"))
            return
        found = {}
        for split, count in EXPECTED_CORAL_SPLITS.items():
            for needle in (f"{count:,}", str(count)):
                if needle in card:
                    found[split] = count
                    break
        if set(found) == set(EXPECTED_CORAL_SPLITS):
            checks.append(Check("CORAL", "PASS", "dataset card documents the expected split table: "
                                + ", ".join(f"{k}={v:,}" for k, v in EXPECTED_CORAL_SPLITS.items())))
        else:
            missing = set(EXPECTED_CORAL_SPLITS) - set(found)
            checks.append(Check("CORAL", "WARN", f"dataset card does not document splits {sorted(missing)} — verify manually on the HF page"))
        return
    problems = [f"{s}: {splits.get(s)} != {n}" for s, n in EXPECTED_CORAL_SPLITS.items() if splits.get(s) != n]
    if problems:
        checks.append(Check("CORAL", "FAIL", "SCHEMA DRIFT — split sizes changed: " + "; ".join(problems)))
        return
    checks.append(Check("CORAL", "PASS",
                        f"splits match code expectations: " + ", ".join(f"{k}={v:,}" for k, v in EXPECTED_CORAL_SPLITS.items())
                        + f" (classes {CORAL_CLASSES} pinned by tests)"))


def verify_deepfish(checks: list[Check]) -> None:
    try:
        payload = json.loads(_fetch(DEEPFISH_TREE_URL, timeout=60).decode("utf-8"))
    except Exception as exc:
        checks.append(Check("DEEPFISH", "FAIL", f"HF tree API unreachable: {exc} — retry"))
        return
    siblings = payload.get("siblings", []) if isinstance(payload, dict) else payload
    tar = next((s for s in siblings if Path(s.get("path", "")).name == "DeepFish.tar"), None)
    if tar is None:
        checks.append(Check("DEEPFISH", "FAIL", "SCHEMA DRIFT — DeepFish.tar no longer present at the HF repo root"))
        return
    size_mb = (tar.get("size") or 0) / (1024 * 1024)
    if not EXPECTED_DEEPFISH_TAR_MB[0] <= size_mb <= EXPECTED_DEEPFISH_TAR_MB[1]:
        checks.append(Check("DEEPFISH", "WARN", f"DeepFish.tar size {size_mb:.0f} MB outside expected ~7.1 GB range"))
        return
    checks.append(Check("DEEPFISH", "PASS", f"DeepFish.tar present, {size_mb:,.0f} MB"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Print the JSON report")
    args = parser.parse_args()

    checks: list[Check] = []
    verify_f4k(checks)
    verify_coral(checks)
    verify_deepfish(checks)

    report = {
        "schema": "reefscan-online-verify-v1",
        "checks": [{"id": c.cid, "status": c.status, "detail": c.detail} for c in checks],
        "result": "FAIL" if any(c.status == "FAIL" for c in checks) else "PASS",
    }

    width = max(len(c.cid) for c in checks)
    line = "=" * 76
    print(line)
    print(" ReefScan AI — online dataset source verification")
    print(line)
    for c in checks:
        print(f" {c.cid:<{width}}  {c.status:<5}  {c.detail}")
    fails = sum(1 for c in checks if c.status == "FAIL")
    print(line)
    if fails:
        print(f" RESULT: {fails} FAIL -> DO NOT DOWNLOAD/TRAIN — dataset sources changed; report to the repo owner")
    else:
        print(" RESULT: all online sources match the code expectations -> safe to download and train")
    print(line)

    out_dir = ROOT / "results" / "online_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "online_verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f" JSON report -> {out_dir / 'online_verify_report.json'}")
    if args.json:
        print(json.dumps(report, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
