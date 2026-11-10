"""QA final-verdict fixes: seg-test fixture, dice test, tracker video guard, notebook hardening."""
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


# ---- 1) seg fixture: create img/ and masks/ dirs (PIL never mkdirs parents) ----
patch("tests/test_seg_eval.py", [
    ('''    root = tmp_path_factory.mktemp("deepfish")
    size = 64
    rng = np.random.default_rng(0)
    for i in range(6):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[10 + i:30 + i, 15:40] = 255
        Image.fromarray(img).save(root / "img" / f"f{i}.png")
        Image.fromarray(mask).save(root / "masks" / f"f{i}_mask.png")''',
     '''    root = tmp_path_factory.mktemp("deepfish")
    (root / "img").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    size = 64
    rng = np.random.default_rng(0)
    for i in range(6):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[10 + i:30 + i, 15:40] = 255
        Image.fromarray(img).save(root / "img" / f"f{i}.png")
        Image.fromarray(mask).save(root / "masks" / f"f{i}_mask.png")'''),
    # ---- 2) dice test: background must be clearly negative (sigmoid(0)=0.5 == threshold)
    ('''    logits = torch.zeros(1, 1, 64, 64)
    logits[0, 0, 10:30, 10:30] = 20.0   # sigmoid -> ~1.0 exactly on target
    dice, iou = dice_iou_scores(logits, target)''',
     '''    logits = torch.full((1, 1, 64, 64), -20.0)  # background clearly negative
    logits[0, 0, 10:30, 10:30] = 20.0            # sigmoid -> ~1.0 exactly on target
    dice, iou = dice_iou_scores(logits, target)'''),
], "test_seg_eval fixture + dice test")

# ---- 3) tracker reset must NOT fire inside the video loop ----
patch("src/realtime/pipeline.py", [
    ('''        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.tracker = CentroidTracker(
            max_disappeared=cfg.realtime.tracker_max_disappeared,
            min_distance=cfg.realtime.tracker_min_distance,
        )''',
     '''        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.tracker = CentroidTracker(
            max_disappeared=cfg.realtime.tracker_max_disappeared,
            min_distance=cfg.realtime.tracker_min_distance,
        )
        # True while process_video owns the tracker (resetting per frame would
        # destroy unique-ID counting — video tracks must persist across frames).
        self._suppress_tracker_reset = False'''),
    ('''        if track and self.cfg.realtime.get("tracker_enabled", True):''',
     '''        if track and not self._suppress_tracker_reset and self.cfg.realtime.get("tracker_enabled", True):'''),
    ('''        self.tracker = CentroidTracker(self.cfg.realtime.tracker_max_disappeared, self.cfg.realtime.tracker_min_distance)
        trend = BleachingTrendAnalyzer(self.cfg.analytics.trend_ewma_alpha)''',
     '''        self.tracker = CentroidTracker(self.cfg.realtime.tracker_max_disappeared, self.cfg.realtime.tracker_min_distance)
        self._suppress_tracker_reset = True  # per-frame analyze_image must not reset it
        trend = BleachingTrendAnalyzer(self.cfg.analytics.trend_ewma_alpha)'''),
], "tracker video-loop guard")

# ---- 4) builder: partial_finetune non-fatal; pull -> fetch+reset (ledger-safe) ----
patch("scripts/build_platform_notebooks.py", [
    ('''# Optional middle ground on the overall winner:
subprocess.run([sys.executable, "train.py", "--task", "coral", "--model", top2[0],
                "--strategy", "partial_finetune", "--platform", "kaggle"], check=True)''',
     '''# Optional middle ground on the overall winner (most memory-hungry step — non-fatal):
r = subprocess.run([sys.executable, "train.py", "--task", "coral", "--model", top2[0],
                    "--strategy", "partial_finetune", "--platform", "kaggle"])
if r.returncode != 0:
    print("WARNING: partial_finetune failed (often OOM) - finetune checkpoints above are kept")'''),
    ('''if not os.path.exists(repo):
    !git clone {REPO_URL} {{repo}}
else:
    !git -C {{repo}} pull --ff-only''',
     '''if not os.path.exists(repo):
    !git clone {REPO_URL} {{repo}}
else:
    # fetch + reset: tracked files update to origin even if a previous phase
    # regenerated tracked files locally (the ledger) — untracked artifacts
    # (models/results/runs) are never touched.
    !git -C {{repo}} fetch origin
    !git -C {{repo}} reset --hard origin/main'''),
], "builder partial-FT + reset pull")

print("QA-FIX PATCH COMPLETE")
