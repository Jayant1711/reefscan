#!/usr/bin/env python
"""Append run manifests into the experiment ledger (my_requested_thing.md).

Scans ``runs/**/manifest.json``, renders one Markdown row per run and updates
the "Experiment log" section of the ledger. Idempotent: rows are keyed by
run-id, re-running refreshes values instead of duplicating rows.

Usage:
    python scripts/log_experiment.py [--ledger my_requested_thing.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEDGER_DEFAULT = Path(__file__).resolve().parents[1] / "my_requested_thing.md"
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
SECTION_START = "<!-- EXPERIMENT-LOG:START -->"
SECTION_END = "<!-- EXPERIMENT-LOG:END -->"


def load_manifests() -> list[dict]:
    manifests = []
    for path in sorted(RUNS_DIR.rglob("manifest.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            m["_run_id"] = f"{m.get('task', '?')}::{m.get('model', '?')}::{m.get('strategy', '?')}"
            m["_source"] = str(path.relative_to(RUNS_DIR.parent))
            manifests.append(m)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return manifests


def fmt(v, digits=4) -> str:
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_row(m: dict) -> str:
    fm = m.get("final_metrics", {})
    params = m.get("params", {})
    total_m = params.get("params_total", 0) / 1e6
    trainable_m = params.get("params_trainable", 0) / 1e6
    minutes = m.get("train_seconds", 0) / 60
    return (
        f"| {m['_run_id']} | {m.get('platform', '?')} | {m.get('dataset', '?')} | "
        f"{m.get('strategy', '?')} | {total_m:.2f}M / {trainable_m:.2f}M | {minutes:.1f} min | "
        f"{fmt(fm.get('accuracy'))} | {fmt(fm.get('macro_ap'))} | {fmt(fm.get('macro_f1'))} | "
        f"{fmt(fm.get('weighted_f1'))} | {fmt(fm.get('iou'))} | {fmt(fm.get('dice'))} | "
        f"{fmt(fm.get('box_ap50'))} | {m.get('git_sha', '?')} | {m['_source']} |"
    )


def build_section(manifests: list[dict]) -> str:
    lines = [
        SECTION_START,
        "",
        "### Experiment log (auto-generated — do not edit by hand)",
        "",
        "Refresh with: `python scripts/log_experiment.py` after every training run.",
        "",
        "| run | platform | dataset | strategy | params (tot/train) | train time | acc | mAP | macro-F1 | wF1 | IoU | Dice | box-AP50 | git | source |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(manifests, key=lambda x: x["_run_id"]):
        lines.append(render_row(m))
    if not manifests:
        lines.append("| _(no runs recorded yet)_ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")
    lines += ["", SECTION_END]
    return "\n".join(lines)


def update_ledger(ledger_path: Path, manifests: list[dict]) -> None:
    if ledger_path.exists():
        text = ledger_path.read_text(encoding="utf-8")
    else:
        text = f"# ReefScan AI — experiment ledger\n\n" + build_section(manifests) + "\n"
        ledger_path.write_text(text, encoding="utf-8")
        print(f"Created {ledger_path}")
        return

    if SECTION_START in text and SECTION_END in text:
        pre = text[: text.index(SECTION_START)]
        post = text[text.index(SECTION_END) + len(SECTION_END):]
        text = pre + build_section(manifests) + post
    else:
        text = text.rstrip() + "\n\n" + build_section(manifests) + "\n"
    ledger_path.write_text(text, encoding="utf-8")
    print(f"Updated {ledger_path} with {len(manifests)} run(s).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(LEDGER_DEFAULT))
    args = parser.parse_args()
    manifests = load_manifests()
    update_ledger(Path(args.ledger), manifests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
