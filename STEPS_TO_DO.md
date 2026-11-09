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

## PHASE 1 — Kaggle (coral models + fish segmentation: E1, E2, E3)

Duration: ~1.5–2.5 h of GPU. Quota cost: ~2 h of your 30.

| Step | Action |
|---|---|
| 1.1 | Go to https://www.kaggle.com/code — **File → Import Notebook** (or New Notebook → File → Import) and upload `notebooks/platform/KAGGLE_training.ipynb` from this repo. Alternatively open your existing `notebook4d825dd1ea` and paste the notebook's cells in. |
| 1.2 | Right sidebar → **Session options** → *Accelerator* → **GPU T4 ×2**; *Internet* → **On**. (Internet needs the phone verification from Phase 0.) |
| 1.3 | Click **Run All** (top toolbar, ▶▶). The notebook: clones `github.com/Jayant1711/reefscan`, installs deps, runs the 65-test suite, downloads coral (~770 MB) + DeepFish (~7.1 GB) to `/kaggle/tmp`, cleans both, trains all 4 coral backbones with feature extraction (~35 min), **reads the leaderboard itself** and fine-tunes the top-2 (~50 min) + partial-finetunes the winner, trains the fish U-Net (~30 min), evaluates everything, and zips the artifacts. |
| 1.4 | Watch it: cells stream progress bars. If a cell fails with `CUDA out of memory` → interrupt, run `!python train.py ... --set coral.batch_size=16` manually for the failing step. If the session dies with Kaggle's "known issue" banner → *Save Version → Save & Run All (Commit)* later, or continue on Lightning (Phase 3). |
| 1.5 | When the last cell prints the zip name: right sidebar → **Output** panel → you will see `reefscan_artifacts_kaggle_<sha8>_<date>.zip` → click the ⋮ → **Download** (~100–300 MB). |
| 1.6 | Save it on your PC at: `C:\reefscan_runs\artifacts\` (create this folder). Keep the zip — do not delete; it is your training record. |

**Alternative (no notebook import):** New Notebook → paste the §0 clone cell from
`PLATFORMS.md`, then copy cells one by one. The notebook route is easier.

---

## PHASE 2 — Colab (fish species models: E4, E5, E6)

Duration: ~2.5–3.5 h of GPU. Best attempted 06:00–10:00 IST when free T4s are available.

| Step | Action |
|---|---|
| 2.1 | Go to https://colab.research.google.com → **File → Upload notebook** → pick `notebooks/platform/COLAB_training.ipynb`. |
| 2.2 | **Runtime → Change runtime type → T4 GPU → Save.** |
| 2.3 | **Runtime → Run all.** The notebook: clones the repo, installs deps, runs the test suite, downloads FishNet v0.1.2 from the fishnet.ai S3 mirror (5.7 GB), crops every fish bounding box into classification crops (~30–50 K crops across ~10 species classes), cleans them, trains all 4 backbones with feature extraction (~1.5 h), **auto-picks the winner** from the leaderboard and fine-tunes it (~1.5 h), runs XAI + enhancement benchmarks. |
| 2.4 | If the first cell hangs at *Allocating runtime* for >10 min: Runtime → **Disconnect and delete runtime**, then reconnect (free-tier capacity issue, not your fault). Off-peak hours almost always connect instantly. |
| 2.5 | The final cell runs `files.download(...)` — your **browser saves `reefscan_artifacts_colab_<sha8>_<date>.zip` to `C:\Users\lenovo\Downloads\`**. Move it to `C:\reefscan_runs\artifacts\`. |
| 2.6 | If the browser download is interrupted before the session dies: re-run only the last cell (it re-creates the zip). If the session already died — re-run the notebook; nothing needs re-typing. |

---

## PHASE 3 — Lightning AI (backup / whatever is missing: E7)

Duration: only the missing experiments. Free monthly credits (~15 T4 hours).

| Step | Action |
|---|---|
| 3.1 | https://lightning.ai → Log in with Google → **New Studio** (default CPU is fine to start). |
| 3.2 | Inside the studio open a **Terminal** and run: `git clone https://github.com/Jayant1711/reefscan.git && cd reefscan && pip install -q -r requirements.txt` |
| 3.3 | Switch to GPU: top-right machine selector → **Switch machine → GPU (T4)** (restarts the studio). |
| 3.4 | Upload the artifact zips you already downloaded (file browser → Upload button) and unzip them here so Lightning sees what is already done: `unzip ../reefscan_artifacts_*.zip -d .` — this restores `runs/` and `results/`, so the backup notebook knows what to skip. |
| 3.5 | Open `notebooks/platform/LIGHTNING_backup.ipynb` (Upload notebook) → **Run All**. It **auto-detects which experiments are missing** from `runs/` manifests and runs exactly one phase (E1 → E2 → E3 → E4 → E5). Re-run it until it prints `ALL EXPERIMENTS COMPLETE`. |
| 3.6 | The last cell prints the zip path → in the **file browser**, right-click `reefscan_artifacts_lightning_*.zip` → **Download** → save to `C:\reefscan_runs\artifacts\`. |

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
   git add my_requested_thing.md results runs
   git commit -m "results: platform training complete + final model selection"
   git push
   ```
5. **Keep as your permanent model library** (recommended): copy
   `models\fish\<winner>\best.pt`, `models\coral\<winner>\best.pt`,
   `models\seg\best.pt` into `C:\reefscan_runs\final_models\` — these three files
   are the deliverable models of your project, backed up outside the repo.

---

## Where the trained models physically live (summary)

| Model | File (inside the zip / after unzip) | Trained on |
|---|---|---|
| Coral bleaching (best of 4 × 3 strategies) | `models/coral/<model>/best.pt` | Kaggle |
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
| FishNet download slow/fails | It comes from the fishnet.ai S3 mirror with resume (`curl -C -`); just re-run the cell — it resumes, it never restarts from zero. |
| `pytest` fails on the platform | Copy the traceback, tell me — do not hand-edit cells; fix belongs in the repo (one `git pull` on the platform picks it up). |
| Lightning credits exhausted | Everything resumes exactly where it stopped next month — manifests make re-runs skip completed work. |

**That's it: Phase 0 once → Phase 1 → Phase 2 → (Phase 3 if needed) → Phase 4 on your PC.
Your trained models end up in `C:\reefscan_runs\artifacts\` → unzipped into the repo →
final three `.pt` files copied to `C:\reefscan_runs\final_models\`.**
