"""Kaggle-only training: remove Colab/Lightning notebooks, demote docs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sweep(path, replacements, label):
    fp = ROOT / path
    src = fp.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new); changed = True
    if changed:
        fp.write_text(src, encoding="utf-8")
    print(f"{label}: {'OK' if changed else 'no changes'}")


# 1) builder: drop colab/lightning cells + exports; main writes only the KAGGLE notebook
p = ROOT / "scripts" / "build_platform_notebooks.py"
src = p.read_text(encoding="utf-8")

# cut EXPORT_COLAB block
start = src.index('EXPORT_COLAB = """')
end = src.index('"""\n\n\nEXPORT_KAGGLE')
src = src[:start] + src[end:]

# cut colab_cells() function
start = src.index("def colab_cells() -> list[dict]:")
end = src.index("def lightning_cells() -> list[dict]:")
src = src[:start] + src[end:]

# cut lightning_cells() function
start = src.index("def lightning_cells() -> list[dict]:")
end = src.index("def main() -> int:")
src = src[:start] + '''def main() -> int:''' + src[end:]

# main(): write only the KAGGLE notebook
old_main = '''    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "KAGGLE_training.ipynb": kaggle_cells(),
        "COLAB_training.ipynb": colab_cells(),
        "LIGHTNING_backup.ipynb": lightning_cells(),
    }
    for name, cells in targets.items():'''
new_main = '''    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        "KAGGLE_training.ipynb": kaggle_cells(),
    }
    for name, cells in targets.items():'''
assert old_main in src, "main targets"
src = src.replace(old_main, new_main)

# docstring: single notebook
src = src.replace(
    'Output: notebooks/platform/{KAGGLE,COLAB,LIGHTNING}_training.ipynb',
    'Output: notebooks/platform/KAGGLE_training.ipynb (the only training lane)',
)
p.write_text(src, encoding="utf-8")
print("builder: Kaggle-only")

# 2) delete the two superseded notebooks
for nb in ("notebooks/platform/COLAB_training.ipynb", "notebooks/platform/LIGHTNING_backup.ipynb"):
    f = ROOT / nb
    if f.exists():
        f.unlink()
        print("deleted", nb)

# 3) README: cloud-training section -> Kaggle only
sweep_readme = [
    ('''Got your own NVIDIA GPU? Follow **[HANDOFF.md](HANDOFF.md)** — the full E1–E6 matrix on one machine in ~5–6 h.

Cloud training: open the ready-to-run platform notebooks — `notebooks/platform/KAGGLE_training.ipynb`,
`COLAB_training.ipynb`, `LIGHTNING_backup.ipynb` — they auto-clone, gate on the test suite,
train the E-matrix, and zip artifacts for download. See `PLATFORMS.md` + `STEPS_TO_DO.md`.''',
     '''**All training runs on Kaggle** (30 free GPU-h/week): open `notebooks/platform/KAGGLE_training.ipynb` —
it auto-clones, gates on the 102-test suite, downloads the data, trains every model
(coral FE screen → FT top-2 → partial · fish FE → FT winner · segmentation), and zips all
artefacts into the Output panel. ~5.5–7.5 h on a T4. See `PLATFORMS.md` + `STEPS_TO_DO.md`.'''),
]
sweep("README.md", sweep_readme, "README cloud lane")

# 4) STEPS_TO_DO.md: remove Colab/Lightning phases, renumber
p = ROOT / "STEPS_TO_DO.md"
src = p.read_text(encoding="utf-8")
start = src.index("## PHASE 2 — Colab")
end = src.index("## PHASE 4 — on your PC")
replacement = """## PHASE 2 — Kaggle continues: fish species (E4, E5, E6)

Same notebook session if quota/time allows, or a **fresh Kaggle session**: re-import
`notebooks/platform/KAGGLE_training.ipynb` — the clone-or-update cell + `--continue`
manifests skip everything already done, then run the fish lane:

| Step | Action |
|---|---|
| 2.1 | The fish cells download F4K (~487 MB) from the Edinburgh mirror, prepare species crops, clean, then run the fish FE screen ×4 and fine-tune the winner |
| 2.2 | Verify: `results/fish/model_leaderboard.csv` + `runs/fish/` manifests |
| 2.3 | Export: `python scripts/export_artifacts.py` → Output panel → download zip → `C:\\reefscan_runs\\artifacts\\` |

---

## PHASE 3 — on your PC: final selection (30 min, CPU only, no training)

"""
src = src[:start] + replacement + src[end:]
# remove the old Phase-3 Lightning section if present
if "## PHASE 3 — Lightning AI (backup" in src:
    start = src.index("## PHASE 3 — Lightning AI (backup")
    end = src.index("## PHASE 4 — on your PC")
    src = src[:start] + src[end:]
# renumber Phase 4 -> stays Phase 3? keep numbering simple: rename to "PHASE 3"
src = src.replace("## PHASE 4 — on your PC: final selection", "## PHASE 3 — on your PC: final selection")
p.write_text(src, encoding="utf-8")
print("STEPS_TO_DO phases rewritten")

# 5) PLATFORMS.md: drop Colab/Lightning sections
p = ROOT / "PLATFORMS.md"
src = p.read_text(encoding="utf-8")
start = src.index("## 2. Colab (secondary")
if start == -1:
    start = src.index("## 2. Colab")
end = src.index("## 4. After all platforms finish")
replacement = """## 2. Backup lanes (only if Kaggle is unavailable)

- **Lightning AI** — studio terminal: clone, `pip install -q -r requirements.txt`, switch machine to GPU (T4), then run the same E-commands with `--platform lightning`.
- **Colab** — `notebooks/platform/COLAB_training.ipynb` flow via the CLI with `--platform colab`.
- **A dedicated NVIDIA-GPU PC** — see **[HANDOFF.md](HANDOFF.md)**.

The owner's decision: **all training on Kaggle**. Backup lanes exist for resilience only.

---

"""
src = src[:start] + replacement + src[end:]
p.write_text(src, encoding="utf-8")
print("PLATFORMS backup lanes demoted")

# 6) ledger §6: platform plan -> Kaggle only
sweep("my_requested_thing.md", [
    ('''| Platform | GPU | Role in matrix |
|---|---|---|
| **Kaggle** (T4×2, 30 h/wk, phone-verified ✓) | primary — can run the FULL matrix E1–E6 | coral screen → FT → seg → fish FE/FT |
| **Colab** (T4, free tier) | secondary | E4 fish FE screen → E5 fish FT → E6 XAI/enhancement |
| **Lightning AI** (T4, free credits) | backup | E7 re-run any failure; overflow fish fine-tune |''',
     '''| Platform | GPU | Role in matrix |
|---|---|---|
| **Kaggle** (T4×2, 30 h/wk, phone-verified ✓) | **the only training platform** (owner decision) | Full matrix E1–E6: coral screen → FT → seg → fish FE/FT → evals |
| ~~Colab / Lightning AI~~ | — | Removed from the plan (owner decision, 2026-11-18); kept in repo as optional backup lanes only |'''),
], "ledger platform plan")

# 7) HANDOFF.md: demote header (alternative lane)
p = ROOT / "HANDOFF.md"
src = p.read_text(encoding="utf-8")
old = "# HANDOFF — Local GPU Workstation Training"
new = ("# HANDOFF — Local GPU Workstation Training\n\n"
       "> **STATUS: optional alternative lane.** The owner's decision (2026-11-18) is that ALL\n"
       "> training runs on **Kaggle** — see PLATFORMS.md and notebooks/platform/KAGGLE_training.ipynb.\n"
       "> This runbook remains valid if a local GPU workstation is ever used instead.")
assert old in src
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("HANDOFF demoted to alternative")

print("KAGGLE-ONLY PATCH COMPLETE")
