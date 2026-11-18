# STEPS TO DO — train on free cloud GPUs, download models to your PC

Zero manual filling anywhere: the notebooks in `notebooks/platform/` already contain the
clone URL, installs, data download, cleaning, training, evaluation, ledger update and the
artifact export. You only enable the GPU, open the notebook, click **Run All**, and download.

**The one rule:** never let a cloud session end before `python scripts/export_artifacts.py`
has run and you have downloaded the zip. Sessions die; your laptop does not.

---

## PHASE 0 — one-time preparation (10 min, on your PC)

1. **Accounts** (all free):
   - Kaggle: https://www.kaggle.com — account `jayant17445` ✔ already exists.
   - Google: for Colab — already signed in ✔.
   - Lightning AI: https://lightning.ai → **Log in → Google** → pick your Google account.
2. **Kaggle phone verification** (unlocks the 30 GPU-hours/week quota):
   - kaggle.com → avatar (top-right) → **Settings** → *Phone verification* → **Phone verify** →
     country `+91` → enter your number → solve the reCAPTCHA → **Start verification** →
     type the SMS code. Status must change to *verified*. ✔ already done on 2026-11-09.
3. **Check your quota:** kaggle.com → Settings → *Quotas* → `Kaggle GPU 00:00 / 30 hrs`.

---

> **Alternative:** a dedicated NVIDIA-GPU PC can run the *entire* matrix locally — see **[HANDOFF.md](HANDOFF.md)**.

## PHASE 1 — Kaggle (FULL matrix: ALL models — E1–E6)

Duration: ~2–3.5 h of GPU (single GPU). Quota cost: ~3 h of your 30.

| Step | Action |
|---|---|
| 1.1 | Go to https://www.kaggle.com/code — **File → Import Notebook** (or New Notebook → File → Import) and upload `notebooks/platform/KAGGLE_training.ipynb` from this repo. Alternatively open your existing `notebook4d825dd1ea` and paste the notebook's cells in. |
| 1.2 | Right sidebar → **Session options** → *Accelerator* → **GPU T4 ×2**; *Internet* → **On**. (Internet needs the phone verification from Phase 0.) |
| 1.3 | Click **Run All** (top toolbar, ▶▶). The notebook: clones `github.com/Jayant1711/reefscan`, installs deps, runs the 102-test suite, downloads coral (~770 MB) + DeepFish (~7.1 GB) to `/kaggle/tmp`, cleans both, trains all 4 coral backbones with feature extraction (~35 min), **reads the leaderboard itself** and fine-tunes the top-2 (~50 min) + partial-finetunes the winner, trains the fish U-Net (~30 min), evaluates everything, and zips the artifacts. |
| 1.4 | Watch it: cells stream progress bars. If a cell fails with `CUDA out of memory` → interrupt, run `!python train.py ... --set coral.batch_size=16` manually for the failing step. If the session dies → *Save Version → Save & Run All (Commit)* later. |
| 1.5 | When the last cell prints the zip name: right sidebar → **Output** panel → you will see `reefscan_artifacts_kaggle_<sha8>_<date>.zip` → click the ⋮ → **Download** (~100–300 MB). |
| 1.6 | Save it on your PC at: `C:\reefscan_runs\artifacts\` (create this folder). Keep the zip — do not delete; it is your training record. |

**Alternative (no notebook import):** New Notebook → paste the §0 clone cell from
`PLATFORMS.md`, then copy cells one by one. The notebook route is easier.

---

## PHASE 2 — Kaggle continues: fish species (E4, E5, E6)

The FULL-matrix notebook already includes the fish lane (F4K download → crops →
clean → FE ×4 → FT winner → eval + XAI). If you imported the older coral-only
notebook, re-import the current `notebooks/platform/KAGGLE_training.ipynb` —
the clone-or-update cell + `--continue` manifests skip everything already done.

**Total Kaggle quota for the entire project: ~6 h of the 30 h/week.**

---

## PHASE 4 — on your PC: final selection (30 min, CPU only, no training)

1. **Unzip everything into the repo** (merge — per-run folders never collide):
   ```bat
   cd C:\Users\lenovo\.zcode\workspace\fabcc916-2ae6-4df7-bfff-ba77c94ffd99
   for %z in (C:\reefscan_runs\artifacts\reefscan_artifacts_*.zip) do tar -xf "%z"
   ```
   (or right-click each zip → Extract Here). You end up with populated
   `models\`, `results\`, `runs\`, `reports\` folders.
2. **Build the final comparison and the ledger's Final-selection section:**
   ```bat
   python scripts\log_experiment.py
   python scripts\compare_runs.py
   ```
   → `results\comparison\final_comparison.csv` + `final_comparison.md`, and
   **`my_requested_thing.md` §7 now names the chosen model per task** with its
   strategy, parameters, training time and metrics — all recorded, nothing to fill.
3. **Point the dashboard at the real models** (they are already in `models\` after
   unzipping) and try it:
   ```bat
   streamlit run app.py
   ```
4. **Commit the ledger + results** (models stay git-ignored):
   ```bat
   git add my_requested_thing.md results/comparison
   git commit -m "results: platform training complete + final model selection"
   git push
   # (models/, most of results/, runs/ are git-ignored by design - the zips in
   #  C:\reefscan_runs\artifacts\ are your permanent training record)
   ```
5. **Keep as your permanent model library** (recommended): copy
   `models\fish\<winner>\best.pt`, `models\coral\<winner>\best.pt`,
   `models\seg\best.pt` into `C:\reefscan_runs\final_models\` — these three files
   are the deliverable models of your project, backed up outside the repo.

---

## Where the trained models physically live (summary)

| Model | File (inside the zip / after unzip) | Trained on |
|---|---|---|
| Coral bleaching (FE screen ×4 → FT top-2 → partial winner) | `models/coral/<model>/best.pt` (last-trained strategy per model) | Kaggle |
| Fish species (FE screen + FT winner) | `models/fish/<model>/best.pt` | Colab |
| Fish segmentation U-Net | `models/seg/best.pt` | Kaggle |
| Metrics + leaderboards + PR curves | `results/<task>/...` | both |
| Run manifests (provenance) | `runs/<task>/<model>_<strategy>/manifest.json` | both |
| Everything bundled | `reefscan_artifacts_<platform>_<sha8>_<date>.zip` | per session |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Kaggle: "issue preventing notebooks from starting" banner | Transient outage — retry later; meanwhile run that phase on Colab/Lightning (they run the same notebooks). |
| Colab: stuck at *Allocating runtime* | Free-tier capacity — disconnect & reconnect, or try off-peak (early morning IST). |
| `CUDA out of memory` | Interrupt → re-run that step with `--set coral.batch_size=16` (or `fish.batch_size=16`, `seg.batch_size=8`). |
| Kaggle Output panel empty | The notebook must run inside `/kaggle/working` — it does (clone cell uses that path). Check the last cell printed a zip name. |
| F4K download slow/fails | It comes from the official Edinburgh mirror with resume (`curl -C -`); re-run the cell — it resumes, it never restarts from zero. |
| `pytest` fails on the platform | Copy the traceback, tell me — do not hand-edit cells; fix belongs in the repo (one `git pull` on the platform picks it up). |
| Lightning credits exhausted | Everything resumes exactly where it stopped next month — manifests make re-runs skip completed work. |

**That's it: Phase 0 once → Phase 1 → Phase 2 → (Phase 3 if needed) → Phase 4 on your PC.
Your trained models end up in `C:\reefscan_runs\artifacts\` → unzipped into the repo →
final three `.pt` files copied to `C:\reefscan_runs\final_models\`.**
