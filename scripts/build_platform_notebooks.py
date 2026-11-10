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

EXPORT_COLAB = """# --- 9. Save everything: ledger + artifact zip + BROWSER DOWNLOAD ----------
!python scripts/log_experiment.py
!python scripts/export_artifacts.py
import glob
from google.colab import files
zip_path = sorted(glob.glob("reefscan_artifacts_colab_*.zip"))[-1]
files.download(zip_path)   # saves the trained models to your PC
"""

EXPORT_KAGGLE = """# --- 9. Save everything: ledger + artifact zip ------------------------------
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
        md("# ReefScan AI — Kaggle GPU training (E1 + E2 + E3)\n\n"
           "Runs the **coral bleaching** experiment matrix (feature-extraction screen → "
           "fine-tune the top-2) plus **fish segmentation**, fully automatically.\n\n"
           "**Before Run All:** right sidebar → *Session options* → **Accelerator: GPU T4 x2** "
           "and **Internet: On**. Duration ≈ 1.5–2.5 h on a T4. "
           "At the end the artifact zip appears in the **Output panel** — download it."),
        code(setup),
        code(GPU_CHECK),
        code("""# --- 2. Datasets (~8 GB to ephemeral /kaggle/tmp) ---------------------------
!python download_data.py --dataset coral --platform kaggle
!python download_data.py --dataset deepfish --platform kaggle"""),
        code("""# --- 3. Data cleaning (report + quarantine flagged files) -------------------
!python clean_data.py --dataset coral --apply --platform kaggle
!python clean_data.py --dataset seg --apply --platform kaggle"""),
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
        code(EXPORT_KAGGLE),
    ]


def colab_cells() -> list[dict]:
    setup = SETUP_COMMON.replace("REPO_DIR", "/content/reefscan")
    return [
        md("# ReefScan AI — Colab GPU training (E4 + E5 + E6)\n\n"
           "Runs the **fish species** experiment matrix on fishnet.ai crops "
           "(feature-extraction screen → fine-tune the winner) + XAI/enhancement benchmarks.\n\n"
           "**Before Run All:** Runtime → *Change runtime type* → **T4 GPU**. Duration ≈ 2.5–3.5 h. "
           "If *Allocating runtime* hangs, retry later — at the end the artifact zip downloads "
           "straight to your PC."),
        code(setup),
        code(GPU_CHECK),
        code("""# --- 2. FishNet v0.1.2 from fishnet.ai S3 (5.7 GB) -> bbox crops ----------
!python download_data.py --dataset fishnet --platform colab
!python download_data.py --prepare-fishnet --platform colab
import glob
n_crops = len(glob.glob("/content/reefscan_data/fishnet/curated/*/*/*.jpg"))
print("fish crops materialized:", n_crops)
assert n_crops > 1000, "too few crops - re-check the download and prepare cells above"
"""),
        code("""# --- 3. Data cleaning (corrupt/dup/leakage scan + quarantine) --------------
!python clean_data.py --dataset fishnet --apply --platform colab"""),
        code("""# --- 4. E4 — fish feature-extraction screen (all 4 backbones, ~1.5 h) ------
!python train.py --task fish --model all --strategy feature_extraction --platform colab
import pandas as pd
pd.read_csv("results/fish/model_leaderboard.csv")"""),
        code("""# --- 5. E5 — auto-pick the winner and fine-tune it (~1.5 h) -----------------
import pandas as pd, subprocess, sys
lb = pd.read_csv("results/fish/model_leaderboard.csv")
fe = lb[lb["strategy"] == "feature_extraction"]
winner = fe.nlargest(1, "weighted_f1")["model"].iloc[0]
print("Fine-tuning FE-screen winner:", winner)
r = subprocess.run([sys.executable, "train.py", "--task", "fish", "--model", winner,
                    "--strategy", "finetune", "--platform", "colab"])
if r.returncode != 0:
    print(f"WARNING: fine-tune failed for {winner} - evaluation below uses the FE checkpoint")
!python evaluate.py --task fish --platform colab
pd.read_csv("results/fish/model_leaderboard.csv")"""),
        code("""# --- 6. E6 — XAI grid + enhancement benchmark on real data ------------------
import glob, shlex
samples = (glob.glob("/content/reefscan_data/fishnet/curated/test/*/*.jpg") or
           glob.glob("/content/reefscan_data/fishnet/curated/train/*/*.jpg"))
!python evaluate.py --enhancement --data /content/reefscan_data/fishnet/curated/train
if samples:
    !python evaluate.py --xai --image {shlex.quote(str(samples[0]))}
else:
    print("No curated fish images found — skipping XAI (not fatal).")"""),
        code(EXPORT_COLAB),
    ]


def lightning_cells() -> list[dict]:
    setup = SETUP_COMMON.replace("REPO_DIR", "reefscan")
    return [
        md("# ReefScan AI — Lightning AI backup run (E7)\n\n"
           "Auto-detects which experiments are missing (from `runs/` manifests) and runs "
           "**the next missing one**. Re-run this notebook until it prints `ALL EXPERIMENTS COMPLETE`.\n\n"
           "**Before Run All:** Studio machine → **GPU (T4)**."),
        code(setup),
        code(GPU_CHECK),
        code("""# --- 2. Auto-detect what is missing and run the next experiment ------------
import subprocess, sys
from pathlib import Path

def done(task, model, strategy):
    marker = Path("runs") / task / f"{model}_{strategy}" / "manifest.json"
    return marker.exists()

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], check=True)

def ensure_coral(dataset="coral"):
    if not Path("/kaggle/tmp/reefscan_data/coral/imagefolders").exists() and \\
       not Path("datasets/coral/imagefolders").exists():
        run(["download_data.py", "--dataset", dataset])

if not done("coral", "resnet34", "feature_extraction"):
    ensure_coral()
    run(["clean_data.py", "--dataset", "coral", "--apply"])
    run(["train.py", "--task", "coral", "--model", "all", "--strategy", "feature_extraction", "--platform", "lightning"])
    print(">>> E1 done. Re-run this notebook to continue with E2.")
elif not any((Path("runs") / "coral" / d.name / "manifest.json").exists()
             for d in Path("runs").glob("coral/*_finetune")):
    import pandas as pd
    lb = pd.read_csv("results/coral/model_leaderboard.csv")
    fe = lb[lb["strategy"] == "feature_extraction"]
    if fe.empty:
        print(">>> No FE rows in the coral leaderboard - run E1 first.")
    else:
        top2 = fe.nlargest(2, "weighted_f1")["model"].tolist()
        for model in top2:
            if not done("coral", model, "finetune"):
                run(["train.py", "--task", "coral", "--model", model, "--strategy", "finetune", "--platform", "lightning"])
        print(">>> E2 done. Re-run this notebook to continue with E3.")
elif not done("seg", "unet_small", "from_scratch"):
    if not Path("datasets/deepfish/DeepFish").exists():
        run(["download_data.py", "--dataset", "deepfish"])
    run(["clean_data.py", "--dataset", "seg", "--apply"])
    run(["train.py", "--task", "seg", "--platform", "lightning"])
    run(["evaluate.py", "--task", "seg"])
    print(">>> E3 done. Re-run this notebook to continue with fish (E4/E5).")
elif not done("fish", "resnet34", "feature_extraction"):
    if not Path("datasets/fishnet/curated").exists() or not list(Path("datasets/fishnet/curated").rglob("*.jpg")):
        run(["download_data.py", "--dataset", "fishnet"])
        run(["download_data.py", "--prepare-fishnet"])
    run(["clean_data.py", "--dataset", "fishnet", "--apply"])
    run(["train.py", "--task", "fish", "--model", "all", "--strategy", "feature_extraction", "--platform", "lightning"])
    print(">>> E4 done. Re-run this notebook to continue with E5.")
elif not any((Path("runs") / "fish" / d.name / "manifest.json").exists()
             for d in Path("runs").glob("fish/*_finetune")):
    import pandas as pd
    lb = pd.read_csv("results/fish/model_leaderboard.csv")
    fe = lb[lb["strategy"] == "feature_extraction"]
    if fe.empty:
        print(">>> No FE rows in the fish leaderboard - run E4 first.")
    else:
        winner = fe.nlargest(1, "weighted_f1")["model"].iloc[0]
        if not done("fish", winner, "finetune"):
            run(["train.py", "--task", "fish", "--model", winner, "--strategy", "finetune", "--platform", "lightning"])
        run(["evaluate.py", "--task", "fish"])
        print(">>> E5 done. Re-run this notebook to confirm ALL EXPERIMENTS COMPLETE.")
else:
    print("ALL EXPERIMENTS COMPLETE — run the export cell below.")"""),
        code("""# --- 3. Save: ledger + artifact zip -----------------------------------------
!python scripts/log_experiment.py
!python scripts/export_artifacts.py
import glob
print("DOWNLOAD (Lightning file browser):", sorted(glob.glob("reefscan_artifacts_lightning_*.zip"))[-1])"""),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "KAGGLE_training.ipynb": kaggle_cells(),
        "COLAB_training.ipynb": colab_cells(),
        "LIGHTNING_backup.ipynb": lightning_cells(),
    }
    for name, cells in targets.items():
        path = OUT / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(notebook(cells, name), fh, indent=1)
        print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
