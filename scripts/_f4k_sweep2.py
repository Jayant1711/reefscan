"""F4K migration sweep 2: all remaining fishnet references."""
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
    print(f"{label}: OK")
    for old, new in replacements:
        if old in src:
            print(f"  !! {label}: NOT replaced -> {old[:60]!r}")


# download_data.py: backward-compat alias is fine, but the legacy --dataset fishnet choice should go
sweep("download_data.py", [], "download_data check-only")

# PLATFORMS.md: kaggle/colab fishnet dataset + notebook + verify step
sweep("PLATFORMS.md", [
    ("!python download_data.py --dataset fishnet --platform colab",
     "!python download_data.py --dataset f4k --platform colab"),
    ("!python download_data.py --prepare-fishnet --platform colab",
     "!python download_data.py --prepare-f4k --platform colab"),
    ("!python clean_data.py --dataset fishnet --apply --platform colab",
     "!python clean_data.py --dataset f4k --apply --platform colab"),
    ("# !python download_data.py --dataset fishnet", "# !python download_data.py --dataset f4k"),
    ("# FishNet (E4/E5 run on Colab; skip here unless re-assigned):",
     "# F4K (E4/E5 run on Colab; skip here unless re-assigned):"),
    ("--enhancement --data /content/reefscan_data/fishnet/curated/train",
     "--enhancement --data /content/reefscan_data/f4k/curated/train"),
    ("python scripts/verify_online_data.py   # upstream schema drift alarm (metadata-only, ~1 min)",
     "python scripts/verify_online_data.py   # upstream schema drift alarm (metadata-only, ~1 min)"),
], "PLATFORMS fishnet refs")

# STEPS_TO_DO.md: kaggle/colab fishnet commands
sweep("STEPS_TO_DO.md", [
    ("!python download_data.py --dataset fishnet --platform colab",
     "!python download_data.py --dataset f4k --platform colab"),
    ("!python download_data.py --prepare-fishnet --platform colab",
     "!python download_data.py --prepare-f4k --platform colab"),
    ("!python clean_data.py --dataset fishnet --apply --platform colab",
     "!python clean_data.py --dataset f4k --apply --platform colab"),
    ("--enhancement --data /content/reefscan_data/fishnet/curated/train",
     "--enhancement --data /content/reefscan_data/f4k/curated/train"),
    ("downloads FishNet v0.1.2 from the fishnet.ai S3 mirror (5.7 GB)",
     "downloads Fish4Knowledge from the official Edinburgh mirror (~487 MB tar)"),
], "STEPS fishnet refs")

# my_requested_thing.md cleaning command line
sweep("my_requested_thing.md", [
    ("`python clean_data.py --dataset {coral,fishnet,seg} --apply`",
     "`python clean_data.py --dataset {coral,f4k,seg} --apply`"),
], "ledger cleaning command")

# docs/DATASETS.md: remaining stale lines (raw layout, quickstart, curated recipe)
sweep("docs/DATASETS.md", [
    ("datasets/fishnet/curated/{train,validation,test}/<CLASS>/*.jpg",
     "datasets/f4k/curated/{train,validation,test}/<species_NN>/*.png"),
    ("Tune via config: `fish.prepare.{level, top_k, min_images_per_class}`.",
     "Tune via config: `fish.prepare.{top_k, min_images_per_class, split_fractions}`."),
    ("**Manual fallback:** open https://drive.google.com/file/d/1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3/view in a browser, place the zip at `datasets/fishnet/raw/FishNet.zip`, re-run `python download_data.py --dataset fishnet` (extraction resumes).",
     "**Manual fallback:** the Edinburgh mirror is quota-free — `curl -L -C - -o datasets/f4k/raw/fishRecognition_GT.tar https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar`, then re-run `python download_data.py --dataset f4k` (resume + MD5 verify)."),
    ("- **Download:** official Google Drive archive (~15 GB) — handled automatically by `src/data/downloaders/fishnet.py` (gdown)",
     "- **Download:** official Edinburgh mirror (~487 MB tar, resumable curl + MD5 verify) — handled automatically by `src/data/downloaders/f4k.py`"),
    ("## 1. Fish4Knowledge (F4K) — species classification (ICCV-2023 FishNet successor)",
     "## 1. Fish4Knowledge (F4K) — species classification"),
    ("Training 17k species on free compute is infeasible, so `prepare_fishnet.py` builds a\n**taxonomy-level subset** (default: `family`, top-25 classes with ≥60 train images):",
     "Free-compute curation keeps the 13 clusters with ≥ 100 images (`prepare_f4k.py`):"),
], "DATASETS fishnet refs")

# docs/TRAINING.md + API.md + ARCHITECTURE.md + USER_MANUAL: dataset dir refs
sweep("docs/TRAINING.md", [
    ("datasets/fishnet/curated", "datasets/f4k/curated"),
    ("fishnet.ai crops", "F4K species crops"),
    ("`prepare_fishnet.py`", "`prepare_f4k.py`"),
], "TRAINING fishnet refs")
sweep("docs/API.md", [
    ("prepare_fishnet", "prepare_f4k"),
    ("fishnet/curated", "f4k/curated"),
    ("download_fishnet", "download_f4k"),
    ("fishnet", "f4k"),
], "API fishnet refs")
sweep("docs/ARCHITECTURE.md", [
    ("fishnet", "f4k"),
], "ARCHITECTURE fishnet refs")
sweep("docs/USER_MANUAL.md", [
    ("fishnet", "f4k"),
], "USER_MANUAL fishnet refs")

# src/classification/datasets.py docstring example paths
sweep("src/classification/datasets.py", [
    ("datasets/fishnet/curated", "datasets/f4k/curated"),
], "classification datasets docstring")

# src/dashboard/pages/home.py dataset hints
sweep("src/dashboard/pages/home.py", [
    ("fishnet", "f4k"),
], "dashboard home hints")

# src/realtime/pipeline.py? grep said no. scripts/preflight.py? grep said no.
sweep("scripts/preflight.py", [], "preflight check-only")

print("F4K SWEEP 2 COMPLETE")
