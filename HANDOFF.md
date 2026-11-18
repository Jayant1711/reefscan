# HANDOFF — Local GPU Workstation Training

**For:** the colleague running the ReefScan AI experiment matrix on a dedicated NVIDIA-GPU machine.
**Repo:** https://github.com/Jayant1711/reefscan (branch `main`)
**Your mission:** execute experiments **E1 → E6** from Table 4.3 of `my_requested_thing.md` on this one machine, then hand the results back.

> **If you are an AI agent (Antigravity or similar) operating this repo:** execute the steps **in order**; after every step run the given ✅ verify command and only proceed when it passes. Never skip `preflight`. Never regenerate files under `notebooks/platform/` (they are generated and QA-approved). Use `--continue` on every training command so re-runs are idempotent. At the end, complete **Step 9 (hand-back)** — the project's final model selection is written from your manifests.

Everything below was designed for exactly one command surface — no cloud paths, no Drive mounts, no notebook juggling. The whole matrix runs on one GPU in **~5–6 hours** (T4-class).

---

## TL;DR card

| | |
|---|---|
| Machine | NVIDIA GPU ≥ 8 GB VRAM (16 GB = T4-class comfortable; 24 GB allows bigger batches), ≥ 30 GB free disk, CUDA 12.x-capable driver (≥ 525) |
| OS | Linux or Windows (commands given for both where they differ) |
| Python | 3.11 or 3.12 (CI validates both) |
| Datasets | ~13.6 GB download (all three; one machine runs the whole matrix) |
| Total GPU time | ~5–6 h (E1 35m · E2 50m · E3 30m · E4 1.5h · E5 1.5h · E6 15m) |
| Platform flag | **`--platform gpu`** everywhere (profile: `configs/platform/gpu.yaml`) |
| Hand-back | `runs/` manifests + `results/comparison/` + updated `my_requested_thing.md` — via artifact zip or a `results/local-gpu` branch push |

---

## Step 0 — Get the repo

```bash
git clone https://github.com/Jayant1711/reefscan.git
cd reefscan
```
✅ Verify: `git log --oneline -1` prints the HEAD SHA the owner gave you (or later — the clone-or-update keeps you current).

## Step 1 — Hardware & driver check

```bash
nvidia-smi
```
✅ Verify: a GPU is listed and the driver version is **≥ 525** (CUDA 12.x). If `nvidia-smi` is missing → install the NVIDIA driver for your card before anything else.

## Step 2 — Python environment + the CUDA-torch gotcha

Create an isolated env (Python 3.11 or 3.12):
```bash
python -m venv .venv
# Linux:   source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

**⚠️ The #1 gotcha — the default `pip install torch` is CPU-only on Windows.** Install the CUDA build explicitly, WITH the repo's constraints so versions match the validated set:

```bash
# Linux (PyPI default already bundles CUDA — constraints still apply):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -c constraints.txt

# Windows (must use the CUDA index):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -c constraints.txt
```
(If `cu124` is superseded on pytorch.org, use the latest `cu1xx` index it offers — the `-c constraints.txt` part is what must stay.)

✅ Verify:
```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```
Must print `CUDA: True`. If False on Windows → you installed the CPU wheel; reinstall with the index above.

## Step 3 — Project dependencies + full test suite

```bash
pip install -r requirements.txt -c constraints.txt
python -m pytest tests/ -q
```
✅ Verify: **102 tests collected, all pass** (one may skip: ONNX parity needs `onnxruntime` — optional). If any test fails, STOP and report the traceback to the owner — do not patch the repo yourself.

## Step 3b — Verify the ONLINE dataset sources (metadata-only, ~1 min, before any download)

The pipeline's assumptions about the upstream data are pinned in code. This step fetches only
metadata (the 2.1 MB labels zip + HuggingFace APIs — no multi-GB downloads) and confirms the
sources still match, so a silent upstream change is caught BEFORE 13.6 GB are pulled:

```bash
python scripts/verify_online_data.py
```
✅ Verify: all three checks PASS —
- `F4K PASS` (columns file_name/cluster_id; 27,370 verified crops; 23 species clusters)
- `CORAL PASS` (splits 7,292/1,562/1,565; classes CORAL, CORAL_BL)
- `DEEPFISH PASS` (DeepFish.tar present, ~7.1 GB)

Any `SCHEMA DRIFT — FAIL` means the upstream release changed: **STOP**, do not download or train,
and report the printed detail to the repo owner (the cleaning/training code must be updated to the
new layout first). JSON report: `results/online_verify/online_verify_report.json`.
After the downloads, cleaning reports (Step 5) re-verify the materialised copy locally.

## Step 4 — Datasets (~13.6 GB, resumable, idempotent)

```bash
python download_data.py --dataset coral --platform gpu
python download_data.py --dataset deepfish --platform gpu
python download_data.py --dataset f4k --platform gpu        # Fish4Knowledge ~487 MB tar + class_id.csv (27,370 crops, 23 species)
```
✅ Verify: `python scripts/preflight.py --platform gpu --task coral` shows `DS_CORAL PASS` (fish/seg will WARN/FAIL until their downloads finish — run preflight again after each download if you want per-dataset confirmation).

Interrupted? Re-run the same command — archives resume (`curl -C -`), extraction and cropping are completion-marker guarded.

## Step 5 — Clean the data (quarantine, nothing deleted)

```bash
python clean_data.py --dataset coral --apply --platform gpu
python clean_data.py --dataset f4k --apply --platform gpu
python clean_data.py --dataset seg --apply --platform gpu
```
✅ Verify: reports at `results/cleaning/<name>_cleaning_report.json`; counts printed at the end.

## Step 6 — Pre-flight gate

```bash
python scripts/preflight.py --platform gpu --task coral,seg,fish --with-pytest
```
✅ Verify: **0 FAIL** (warnings OK). The command exits 1 on any FAIL — do not start training until it exits 0.

## Step 7 — The experiment matrix (in order; ONE GPU box runs E1–E6 sequentially)

| Step | Command | Verify after |
|---|---|---|
| **E1** coral FE screen (~35 m) | `python train.py --task coral --model all --strategy feature_extraction --platform gpu --continue` | `results/coral/model_leaderboard.csv` has 4 FE rows |
| **E2** coral FT top-2 (~50 m) | `python train.py --task coral --model efficientnet_b0 --strategy finetune --platform gpu --continue` then same for `convnext_tiny`; optional `--strategy partial_finetune` on the E1 winner (skip on OOM) | 2 finetune manifests under `runs/coral/` |
| **E3** segmentation (~30 m) | `python train.py --task seg --platform gpu --continue` then `python evaluate.py --task seg --platform gpu` | `results/seg/metrics.json` (IoU/Dice/pixel-AP/box-AP50) |
| **E4** fish FE screen (~1.5 h) | `python train.py --task fish --model all --strategy feature_extraction --platform gpu --continue` | `results/fish/model_leaderboard.csv` has 4 FE rows |
| **E5** fish FT winner (~1.5 h) | `python train.py --task fish --model <E4 winner> --strategy finetune --platform gpu --continue` | `runs/fish/<winner>_finetune/manifest.json` |
| **E6** evaluation sweep (~15 m) | `python evaluate.py --task coral --platform gpu` · `--task fish --platform gpu` · `--enhancement --data datasets/f4k/curated/train` | refreshed leaderboards + `results/enhancement/` |

The E2/E5 winners are picked from the leaderboards (rank by `weighted_f1` among `strategy == feature_extraction` rows). OOM on a backbone? It is caught automatically and the sweep continues; for a manual re-run add `--set coral.batch_size=32` (or 16). `--continue` makes every re-run skip completed work.

## Step 8 — Ledger + final comparison

```bash
python scripts/log_experiment.py      # auto-writes the Experiment log into my_requested_thing.md
python scripts/compare_runs.py        # auto-writes Final selection + results/comparison/
```
✅ Verify: `my_requested_thing.md` §Experiment log has one row per run; `results/comparison/final_comparison.csv` exists.

## Step 9 — Export and hand back

```bash
python scripts/export_artifacts.py    # -> reefscan_artifacts_<platform>_<sha8>_<date>.zip
```
Then hand back by EITHER:
- **(preferred)** push a branch with the committable results:
  ```bash
  git checkout -b results/local-gpu
  git add my_requested_thing.md results/comparison results/cleaning
  git commit -m "results: local-GPU run (E1-E6)"
  git push -u origin results/local-gpu
  ```
- **or** simply send `reefscan_artifacts_gpu_*.zip` back to the owner (contains models/, results/, runs/, reports/).

---

## Compute selection (Antigravity / agent rule)

Preference order when more than one GPU is reachable — **train on the best available, keep the
local PC's GPU free**:

| Priority | Compute | What to run | Why |
|---|---|---|---|
| 1st | **Linked Colab GPU** (if the agent has a Colab connector) | `notebooks/platform/COLAB_training.ipynb` cell-by-cell (E4 → E5 → E6), or the CLI in the Colab terminal with `--platform colab` | Free T4; keeps the workstation free; the notebook's gates (pytest → crops assert → preflight) protect the run |
| 2nd | **This workstation's NVIDIA GPU** | Steps 4–8 above with `--platform gpu` | Full matrix locally with `--continue` |
| never | CPU-only machine | — | Training on CPU is not a supported mode |

If training is executed on Colab while the workstation holds downloaded datasets, that is fine:
`--continue` + the manifests mean the other platform skips completed work (E-matrix is split by
platform in `my_requested_thing.md` §6 — E1–E3 coral/seg, E4–E5 fish).

---

## The first prompt to give Antigravity (copy-paste this entire block)

```text
You are operating the ReefScan AI repository — a final-year underwater computer-vision
system: underwater image enhancement -> fish species classification (Fish4Knowledge
crops) -> coral bleaching severity (NOAA patches, WGBS estimator) -> fish detection
(U-Net on DeepFish masks) -> biodiversity analytics -> PDF reports + Streamlit dashboard.
Full runbook: HANDOFF.md at the repo root. Dataset→model mapping + experiment matrix:
my_requested_thing.md §1/§4/§6.

REPO: https://github.com/Jayant1711/reefscan  (clone it if not already present)

MISSION: run the complete training matrix E1–E6 on the best GPU available to you —
preference: a linked Colab GPU first (use notebooks/platform/COLAB_training.ipynb or the
CLI there with --platform colab); otherwise THIS machine's NVIDIA GPU (--platform gpu).
NEVER train on CPU-only machines. Deliver every artefact the owner's GPU-less laptop
could not produce: trained checkpoints, leaderboards with average-precision metrics,
segmentation IoU/Dice/box-AP50, per-run manifests, the filled experiment ledger, the
final model-selection comparison, and the bundled artifact zip.

Execute in order, verifying after every step:

1. Environment (HANDOFF.md Steps 0–3): Python 3.11+ venv; install CUDA torch
   (Windows: default pip wheel is CPU-only — use the cu1xx index) with
   -c constraints.txt; then pip install -r requirements.txt -c constraints.txt.
   VERIFY: python -m pytest tests/ -q (102 tests) must pass, and
   python -c "import torch; print(torch.cuda.is_available())" must be True.
   NOTE: if imports crash with "WinError 4551 / Application Control policy has
   blocked this file", that is a WINDOWS machine policy (torch/pyarrow/matplotlib
   DLLs), NOT a repo bug — do NOT edit the repo; try reinstalling the affected
   package, then report to the user if it persists (the owner's laptop had the
   same policy and it later resolved itself).

2. Dataset-source verification (BEFORE the multi-GB downloads):
   python scripts/verify_online_data.py
   All three checks (F4K, CORAL, DEEPFISH) must PASS — schema drift on any
   upstream source means STOP and report to the user; do not patch the code.

3. Data (HANDOFF.md Step 4): download all three datasets with --platform matching
   your compute, then clean each with clean_data.py --apply. VERIFY:
   results/cleaning/*_cleaning_report.json exist and curated/imagefolder trees
   are populated (F4K ~26 K crops across 13 species classes).

4. Pre-flight gate (Step 6): python scripts/preflight.py --platform <compute>
   --task all — must exit 0. A FAIL halts the run: fix the environment, not the code.

5. Training matrix (HANDOFF.md Step 7) — dataset→model mapping:
   - E1  coral:   train.py --task coral --model all --strategy feature_extraction
                  (NOAA coral patches)
   - E2  coral:   fine-tune the top-2 by weighted_f1 among FE rows
                  (+ optional partial_finetune on the winner, non-fatal)
   - E3  seg:     train.py --task seg (U-Net on DeepFish masks), then
                  evaluate.py --task seg (IoU/Dice/pixel-AP/box-AP50)
   - E4  fish:    train.py --task fish --model all --strategy feature_extraction
                  (F4K species crops)
   - E5  fish:    fine-tune the E4 FE-screen winner
   - E6  eval:    evaluate.py --task coral / --task fish / --enhancement / --xai
   Use --continue on every train.py call (idempotent — completed runs are skipped)
   and the matching --platform on every CLI call. One CUDA OOM is caught per
   backbone automatically; retry that backbone with --set <task>.batch_size=32.

6. Ledger + hand-back (HANDOFF.md Step 9):
   python scripts/log_experiment.py     # fills my_requested_thing.md Experiment log
   python scripts/compare_runs.py       # writes results/comparison/final_comparison.csv
                                        # and the FINAL SELECTION into the ledger
   python scripts/export_artifacts.py   # bundles models/results/runs/reports into one zip
   Hand the zip back to the user AND push the results/local-gpu (or colab) branch with
   the committable results (comparison + cleaning reports).

RULES: never edit source/config files to fix a failure — report it to the user;
never skip preflight or the pytest gate; re-run scripts/verify_online_data.py if a
download fails mid-way (upstream may have drifted); --continue on every training call;
prefer the linked Colab GPU over this machine's GPU; do not run run_smoke_test.py on
the training box mid-matrix. Report every traceback verbatim to the user.
```

---

## VRAM → batch-size table

| VRAM | coral/fish batch | seg batch | Notes |
|---|---|---|---|
| 8 GB (RTX 3060 Ti class) | 32 | 8 | `--set coral.batch_size=32 --set fish.batch_size=32 --set seg.batch_size=8` |
| 16 GB (T4 / 4060 Ti / 4070) | 64 (profile default) | 16 | nothing to change |
| 24 GB (3090/4090) | 128 | 32 | faster wall-clock; results unchanged |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA: False` after Step 2 | Wrong wheel — reinstall torch with the `cu1xx` index (Step 2); check `nvidia-smi` driver ≥ 525 |
| `CUDA out of memory` | Caught automatically per backbone on `--model all`; for a single run add `--set <task>.batch_size=32` (or 16) |
| FishNet download stalls | Re-run the same command — `curl -C -` resumes; never delete the partial zip |
| Preflight FAIL on DS_* | The matching dataset is missing/partial — re-run its `download_data.py` command (idempotent) |
| `git pull` refuses on re-clone updates | Untracked artifacts are safe; run `git fetch origin && git reset --hard origin/main` (code only) |
| pytest fails on the box | Copy the traceback and send it to the owner — do not patch the repo |
| anything else | `results/reefscan.log` + the failing command's traceback → send both to the owner |

## Rules for AI agents (Antigravity et al.)

1. Execute steps **in order**; verify after each with the ✅ command before moving on.
2. `--continue` on every `train.py` invocation; `--platform gpu` on every CLI call.
3. Never edit source/config files to "fix" a failure — report it. The suite is QA-approved; failures are environment or genuine bugs the owner must see.
4. Never regenerate `notebooks/platform/` or edit `configs/` — profiles are pinned by tests.
5. Do not run `run_smoke_test.py` on the training box mid-matrix (it overwrites `models/demo/` and `results/`); it is a CI/laptop tool.
6. Finish with **Step 9** even if some experiments failed — partial manifests are valuable and the owner's ledger handles gaps.
7. Run `scripts/verify_online_data.py` before the first download AND whenever a download fails mid-way — upstream schema drift must stop the run, not be patched around.
8. Compute preference: linked Colab GPU > this workstation's GPU > (never) CPU-only training. Keep the workstation GPU load minimal when a Colab GPU is attached.
