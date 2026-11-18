#!/usr/bin/env python
"""Generate ready-to-run platform notebooks (Kaggle / Colab / Lightning).

Every notebook is **fully auto-filled** — clone URL, install, GPU check, test
suite, dataset download, cleaning, training (with winner auto-selection from
leaderboards), evaluation, experiment-ledger update, artifact export and the
platform-specific download step. No placeholders; clone → Run All → download.

Output: notebooks/platform/{KAGGLE,COLAB,LIGHTNING}_training.ipynb
Re-run this script after changing the experiment commands.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "platform"
REPO_URL = "https://github.com/Jayant1711/reefscan.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def notebook(cells: list[dict], name: str) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "accelerator": "GPU", "colab": {"provenance": []}},
        "cells": cells,
    }


SETUP_COMMON = f"""# --- 0. Clone (or update) the repo + install dependencies ------------------
# Re-runs auto-update the code (git pull) while PRESERVING your trained
# artefacts inside the clone — never re-download, never lose checkpoints.
import os
repo = "REPO_DIR"
if not os.path.exists(repo):
    !git clone {REPO_URL} {{repo}}
else:
    # fetch + reset: tracked files update to origin even if a previous phase
    # regenerated tracked files locally (the ledger) — untracked artifacts
    # (models/results/runs) are never touched.
    !git -C {{repo}} fetch origin
    !git -C {{repo}} reset --hard origin/main
%cd {{repo}}
!pip install -q -r requirements.txt

if os.path.basename(os.getcwd()) != os.path.basename(repo):
    os.chdir(repo)  # kernel restarts land elsewhere — re-enter the repo"""

GPU_CHECK = """# --- 1. GPU + test-suite check (fails loudly if anything is broken) --------
import torch, subprocess, sys
assert torch.cuda.is_available(), "NO GPU — enable the accelerator and re-run!"
print("GPU:", torch.cuda.get_device_name(0))
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
assert r.returncode == 0, "TEST SUITE FAILED — fix before training (do not skip!)"
"""


EXPORT_KAGGLE = """# --- 12. Save everything: ledger + artifact zip -----------------------------
!python scripts/log_experiment.py
!python scripts/export_artifacts.py
import glob, os, shutil
zip_path = sorted(glob.glob("reefscan_artifacts_kaggle_*.zip"))[-1]
final = os.path.join("/kaggle/working", os.path.basename(zip_path))
if os.path.abspath(zip_path) != os.path.abspath(final):
    shutil.move(zip_path, final)
print("DOWNLOAD NOW (before the session ends):")
print("  Right sidebar -> Output panel ->", os.path.basename(final))
print("  Everything under /kaggle/working persists as notebook output.")
"""


def kaggle_cells() -> list[dict]:
    setup = SETUP_COMMON.replace("REPO_DIR", "/kaggle/working/reefscan")
    return [
        md("# ReefScan AI — Kaggle GPU training (FULL matrix: E1–E6, all models)\n\n"
           "Runs **every model** in one session: coral bleaching (feature-extraction screen → "
           "fine-tune top-2 → partial), fish segmentation, AND fish species classification "
           "(Fish4Knowledge crops).\n\n"
           "**Before Run All:** right sidebar → *Session options* → **Accelerator: GPU T4 x2** "
           "and **Internet: On**. Duration ≈ 5.5–7.5 h on a T4 (fits the 12 h session; "
           "30 h/week quota). At the end the artifact zip with ALL trained models appears in "
           "the **Output panel** — download it."),
        code(setup),
        code(GPU_CHECK),
        code("""# --- 2. Datasets (~8 GB to ephemeral /kaggle/tmp) ---------------------------
!python download_data.py --dataset coral --platform kaggle
!python download_data.py --dataset deepfish --platform kaggle"""),
        code("""# --- 3. Data cleaning (report + quarantine flagged files) -------------------
!python clean_data.py --dataset coral --apply --platform kaggle
!python clean_data.py --dataset seg --apply --platform kaggle
# --- 3b. Pre-flight validation: FAIL halts Run All before any GPU time ------
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/preflight.py", "--platform", "kaggle", "--task", "coral,seg"])
assert r.returncode == 0, "PREFLIGHT FAILED — fix the FAIL rows above before training. "
"""),
        code("""# --- 4. E1 — coral feature-extraction screen (all 4 backbones, ~35 min) ----
!python train.py --task coral --model all --strategy feature_extraction --platform kaggle
import pandas as pd
pd.read_csv("results/coral/model_leaderboard.csv")"""),
        code("""# --- 5. E2 — auto-pick top-2 from the screen and fine-tune them (~50 min) --
import pandas as pd, subprocess, sys
lb = pd.read_csv("results/coral/model_leaderboard.csv")
fe = lb[lb["strategy"] == "feature_extraction"]
assert len(fe) >= 2, f"FE screen has only {len(fe)}/4 backbones with metrics - cannot pick top-2"
if len(fe) < 4:
    print(f"WARNING: FE screen incomplete ({len(fe)}/4) - proceeding with available backbones")
top2 = fe.nlargest(2, "weighted_f1")["model"].tolist()
print("Fine-tuning top-2 from the FE screen:", top2)
ft_failures = []
for model in top2:
    r = subprocess.run([sys.executable, "train.py", "--task", "coral", "--model", model,
                        "--strategy", "finetune", "--platform", "kaggle"])
    if r.returncode != 0:
        ft_failures.append(model)
        print(f"WARNING: fine-tune failed for {model} - continuing")
print("Fine-tune failures:", ft_failures or "none")
# Optional middle ground on the overall winner (most memory-hungry step — non-fatal):
r = subprocess.run([sys.executable, "train.py", "--task", "coral", "--model", top2[0],
                    "--strategy", "partial_finetune", "--platform", "kaggle"])
if r.returncode != 0:
    print("WARNING: partial_finetune failed (often OOM) - finetune checkpoints above are kept")
!python evaluate.py --task coral --platform kaggle
pd.read_csv("results/coral/model_leaderboard.csv")"""),
        code("""# --- 6. E3 — fish segmentation U-Net (~30 min) + IoU/Dice/AP metrics -------
!python train.py --task seg --platform kaggle
!python evaluate.py --task seg --platform kaggle
import json; print(json.dumps(json.load(open("results/seg/metrics.json")), indent=2))"""),
        code("""# --- 7. XAI sanity check on the best coral model (Phase-3 innovation) -------
import glob, shlex
sample = (glob.glob("/kaggle/tmp/reefscan_data/coral/imagefolders/test/*/*.jpg") +
          glob.glob("datasets/coral/imagefolders/test/*/*.jpg"))
if sample:
    !python evaluate.py --xai --xai-task coral --image {shlex.quote(str(sample[0]))}
else:
    print("No coral test images found — skipping XAI (not fatal).")"""),
        code("""# --- 8. E4 — fish species data (F4K) + cleaning + preflight -----------------
!python download_data.py --dataset f4k --platform kaggle      # ~487 MB tar -> 27,370 crops
!python download_data.py --prepare-f4k --platform kaggle      # curated species ImageFolder
!python clean_data.py --dataset f4k --apply --platform kaggle
import subprocess, sys, glob
r = subprocess.run([sys.executable, "scripts/preflight.py", "--platform", "kaggle", "--task", "fish"])
assert r.returncode == 0, "PREFLIGHT FAILED for fish - fix the FAIL rows above before training."
print("F4K species crops:", len(glob.glob("/kaggle/tmp/reefscan_data/f4k/curated/*/*/*.png")))"""),
        code("""# --- 9. E4 — fish feature-extraction screen (all 4 backbones, ~1.5-2.5 h) ----
!python train.py --task fish --model all --strategy feature_extraction --platform kaggle
import pandas as pd
pd.read_csv("results/fish/model_leaderboard.csv")"""),
        code("""# --- 10. E5 — auto-pick the fish FE-screen winner and fine-tune (~1.5-2.5 h) --
import pandas as pd, subprocess, sys
lb = pd.read_csv("results/fish/model_leaderboard.csv")
fe = lb[lb["strategy"] == "feature_extraction"]
assert len(fe) >= 1, f"FE screen has no rows - E4 did not complete (only {len(lb)} rows total)"
if len(fe) < 4:
    print(f"WARNING: FE screen incomplete ({len(fe)}/4) - proceeding with available backbones")
winner = fe.nlargest(1, "weighted_f1")["model"].iloc[0]
print("Fine-tuning FE-screen winner:", winner)
r = subprocess.run([sys.executable, "train.py", "--task", "fish", "--model", winner,
                    "--strategy", "finetune", "--platform", "kaggle"])
if r.returncode != 0:
    print(f"WARNING: fine-tune failed for {winner} - evaluation below uses the FE checkpoint")
!python evaluate.py --task fish --platform kaggle
pd.read_csv("results/fish/model_leaderboard.csv")"""),
        code("""# --- 11. E6 — XAI sanity check on the best fish model -----------------------
import glob, shlex
sample = (glob.glob("/kaggle/tmp/reefscan_data/f4k/curated/test/*/*.png") or
          glob.glob("datasets/f4k/curated/test/*/*.png"))
if sample:
    !python evaluate.py --xai --xai-task fish --image {shlex.quote(str(sample[0]))}
else:
    print("No F4K test crops found — skipping XAI (not fatal).")"""),
        code(EXPORT_KAGGLE),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "KAGGLE_training.ipynb": kaggle_cells(),
    }
    for name, cells in targets.items():
        path = OUT / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(notebook(cells, name), fh, indent=1)
        print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
