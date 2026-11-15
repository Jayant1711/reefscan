"""F4K migration patch: clean_data recipe, configs, online verifier."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path, replacements, label):
    fp = ROOT / path
    src = fp.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new); changed = True
        elif new in src:
            continue
        else:
            print(f"  !! {label}: absent -> {old[:64]!r}")
    if changed:
        fp.write_text(src, encoding="utf-8")
    print(f"{label}: {'OK' if changed else 'no changes'}")


# 1) clean_data.py: fishnet recipe -> f4k
patch("clean_data.py", [
    ('''    "fishnet": {
        "target": "curated ImageFolder (corrupt + duplicates + leakage + class audit)",
        "kind": "imagefolder",
        "dir": lambda cfg: Path(cfg.fish.data_dir),
    },''',
     '''    "f4k": {
        "target": "curated species ImageFolder (corrupt + duplicates + leakage + class audit)",
        "kind": "imagefolder",
        "dir": lambda cfg: Path(cfg.fish.data_dir),
    },'''),
    ('''        if args.dataset == "fishnet":
            _audit_classes(target, report)''',
     '''        if args.dataset == "f4k":
            _audit_classes(target, report)'''),
    ('''                target, args.dataset if args.dataset != "fishnet" else "fishnet (+ --prepare-fishnet)",''',
     '''                target, args.dataset if args.dataset != "f4k" else "f4k (+ --prepare-f4k)",'''),
], "clean_data f4k recipe")

# 2) configs/default.yaml fish section
patch("configs/default.yaml", [
    ('''  raw_dir: datasets/fishnet/raw
  data_dir: datasets/fishnet/curated   # ImageFolder layout (built by prepare_fishnet.py)
  prepare:
    # fishnet.ai (TNC) bbox-annotation schema: img_id, bbox_id, x/y_min/max, label_l1, label_l2
    level: l1                # l1 = fine-grained species (28) | l2 = coarse codes (13)
    top_k: 12                # keep the K classes with the most fish boxes
    min_images_per_class: 100
    exclude_classes: [Human, Unknown, No fish]
    split_fractions: [0.8, 0.1, 0.1]   # seeded IMAGE-level split (no crop leakage)''',
     '''  raw_dir: datasets/f4k/raw
  data_dir: datasets/f4k/curated       # species ImageFolder (built by prepare_f4k.py)
  prepare:
    # Fish4Knowledge: 27,370 verified crops, 23 species clusters (fish_NN dirs);
    # labels carried by the tar directory structure (verified vs class_id.csv)
    top_k: 13                # clusters with >= 100 images
    min_images_per_class: 100
    split_fractions: [0.8, 0.1, 0.1]   # seeded VIDEO-GROUP split (crops of one video share a split)'''),
], "default.yaml fish section")

# 3) platform overlays: fish paths -> f4k
patch("configs/platform/kaggle.yaml", [
    ("fish:\n  batch_size: 64\n  raw_dir: /kaggle/tmp/reefscan_data/fishnet/raw\n  data_dir: /kaggle/tmp/reefscan_data/fishnet/curated",
     "fish:\n  batch_size: 64\n  raw_dir: /kaggle/tmp/reefscan_data/f4k/raw\n  data_dir: /kaggle/tmp/reefscan_data/f4k/curated"),
], "kaggle overlay fish paths")
patch("configs/platform/colab.yaml", [
    ("fish:\n  batch_size: 64\n  raw_dir: /content/reefscan_data/fishnet/raw\n  data_dir: /content/reefscan_data/fishnet/curated",
     "fish:\n  batch_size: 64\n  raw_dir: /content/reefscan_data/f4k/raw\n  data_dir: /content/reefscan_data/f4k/curated"),
], "colab overlay fish paths")

# 4) verify_online_data.py: fishnet check -> f4k check
patch("scripts/verify_online_data.py", [
    ('''FISHNET_LABELS_URL = "https://fishnet-data.s3-us-west-1.amazonaws.com/foid_labels_v012.zip"
EXPECTED_FISHNET_COLUMNS = ("img_id", "bbox_id", "x_min", "x_max", "y_min", "y_max", "label_l1", "label_l2")
EXPECTED_FISHNET_MIN_ROWS = 100_000          # release has 159,119 bbox rows
EXPECTED_FISHNET_FRAMES = (30_000, 40_000)   # 34,987 in v0.1.2
EXCLUDED_L1 = {"Human", "Unknown", "No fish"}
EXPECTED_FISH_CLASSES = {"Albacore", "Yellowfin tuna", "Bigeye tuna"}  # three largest fish classes''',
     '''F4K_CLASS_ID_URL = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/class_id.csv"
F4K_TAR_URL = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar"
EXPECTED_F4K_COLUMNS = ("file_name", "cluster_id")
EXPECTED_F4K_ROWS = (25_000, 30_000)         # 27,370 verified crops
EXPECTED_F4K_CLUSTERS = 23                   # species clusters; 13 have >= 100 images'''),
    ('''def verify_fishnet(checks: list[Check]) -> None:
    try:
        blob = _fetch(FISHNET_LABELS_URL, timeout=180)
    except Exception as exc:
        checks.append(Check("FISHNET", "FAIL", f"labels zip unreachable at {FISHNET_LABELS_URL}: {exc} — retry; if it persists, the S3 mirror moved"))
        return
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv") and "__macosx" not in n.lower()]
            non_freq = [n for n in members if "freq" not in n.lower()] or members
            with zf.open(non_freq[0]) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                reader = csv.DictReader(text)
                columns = tuple(reader.fieldnames or [])
                rows = list(reader)
    except Exception as exc:
        checks.append(Check("FISHNET", "FAIL", f"labels CSV unreadable: {exc}"))
        return

    if columns != EXPECTED_FISHNET_COLUMNS:
        checks.append(Check("FISHNET", "FAIL",
                            f"SCHEMA DRIFT — expected columns {EXPECTED_FISHNET_COLUMNS}, got {columns}. "
                            "STOP: update prepare_fishnet.py before training."))
        return

    frames = {r["img_id"] for r in rows}
    classes = {r["label_l1"] for r in rows}
    fish_classes = classes - EXCLUDED_L1
    missing_big = EXPECTED_FISH_CLASSES - fish_classes
    if missing_big:
        checks.append(Check("FISHNET", "FAIL", f"SCHEMA DRIFT — expected fish classes missing: {sorted(missing_big)}"))
        return

    details = [f"{len(rows)} bbox rows", f"{len(frames)} frames", f"{len(classes)} L1 classes"]
    status = "PASS"
    if len(rows) < EXPECTED_FISHNET_MIN_ROWS:
        status = "WARN"
        details.append(f"rows below expected {EXPECTED_FISHNET_MIN_ROWS:,}")
    if not EXPECTED_FISHNET_FRAMES[0] <= len(frames) <= EXPECTED_FISHNET_FRAMES[1]:
        status = "WARN"
        details.append(f"frame count outside {EXPECTED_FISHNET_FRAMES}")
    checks.append(Check("FISHNET", status, "schema matches prepare_fishnet.py; " + "; ".join(details)))''',
     '''def verify_f4k(checks: list[Check]) -> None:
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
    checks.append(Check("F4K", status, "schema matches prepare_f4k.py; " + "; ".join(details)))'''),
    ('''    checks: list[Check] = []
    verify_fishnet(checks)
    verify_coral(checks)
    verify_deepfish(checks)''',
     '''    checks: list[Check] = []
    verify_f4k(checks)
    verify_coral(checks)
    verify_deepfish(checks)'''),
], "verify_online_data f4k")

print("F4K PATCH A COMPLETE")
