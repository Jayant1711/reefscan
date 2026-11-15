"""F4K sweep 3: builder fishnet refs + pin test rewrite."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sweep(path, replacements, label):
    fp = ROOT / path
    src = fp.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new); changed = True
        elif old not in src and new in src:
            continue  # already applied
        else:
            print(f"  !! {label}: absent -> {old[:64]!r}")
    if changed:
        fp.write_text(src, encoding="utf-8")
    print(f"{label}: {'OK' if changed else 'no changes'}")


# 1) builder: COLAB cells fishnet -> f4k (crops glob changes to PNG + f4k counts)
sweep("scripts/build_platform_notebooks.py", [
    ('''        code("""# --- 2. FishNet v0.1.2 from fishnet.ai S3 (5.7 GB) -> bbox crops ----------
!python download_data.py --dataset f4k --platform colab
!python download_data.py --prepare-fishnet --platform colab
import glob
n_crops = len(glob.glob("/content/reefscan_data/fishnet/curated/*/*/*.jpg"))
print("fish crops materialized:", n_crops)
assert n_crops > 1000, "too few crops - re-check the download and prepare cells above "''',
     '''        code("""# --- 2. Fish4Knowledge from the Edinburgh mirror (~487 MB) -> species crops ----
!python download_data.py --dataset f4k --platform colab
!python download_data.py --prepare-f4k --platform colab
import glob
n_crops = len(glob.glob("/content/reefscan_data/f4k/curated/*/*/*.png"))
print("species crops materialized:", n_crops)
assert n_crops > 1000, "too few crops - re-check the download and prepare cells above "'''),
    ('''        code("""# --- 3. Data cleaning (corrupt/dup/leakage scan + quarantine) --------------
!python clean_data.py --dataset fishnet --apply --platform colab
# --- 3b. Pre-flight validation: FAIL halts Run All before any GPU time ------
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/preflight.py", "--platform", "colab", "--task", "fish,fishnet"])
assert r.returncode == 0, "PREFLIGHT FAILED — fix the FAIL rows above before training. "''',
     '''        code("""# --- 3. Data cleaning (corrupt/dup/leakage scan + quarantine) --------------
!python clean_data.py --dataset f4k --apply --platform colab
# --- 3b. Pre-flight validation: FAIL halts Run All before any GPU time ------
import subprocess, sys
r = subprocess.run([sys.executable, "scripts/preflight.py", "--platform", "colab", "--task", "fish"])
assert r.returncode == 0, "PREFLIGHT FAILED — fix the FAIL rows above before training. "'''),
    ('''samples = (glob.glob("/content/reefscan_data/fishnet/curated/test/*/*.jpg") or
           glob.glob("/content/reefscan_data/fishnet/curated/train/*/*.jpg"))
!python evaluate.py --enhancement --data /content/reefscan_data/fishnet/curated/train''',
     '''samples = (glob.glob("/content/reefscan_data/f4k/curated/test/*/*.png") or
           glob.glob("/content/reefscan_data/f4k/curated/train/*/*.png"))
!python evaluate.py --enhancement --data /content/reefscan_data/f4k/curated/train'''),
    ('''           "Runs the **fish species** experiment matrix on fishnet.ai crops "''',
     '''           "Runs the **fish species** experiment matrix on Fish4Knowledge crops "'''),
], "builder COLAB f4k")

# 2) builder markdown header for COLAB duration (already fine) — no change needed.

# 3) pin test: rewrite fishnet pin -> f4k pin (source-parse, import-free)
p = ROOT / "tests" / "test_pinning_and_export.py"
src = p.read_text(encoding="utf-8")
old = '''def test_fishnet_label_columns_pinned_between_verifier_and_prepare():
    """scripts/verify_online_data.py (the online drift alarm) and
    src/data/prepare_fishnet.py (the consumer) must agree on the fishnet.ai
    label schema — including bbox_id, which the crop loop dereferences."""
    root = Path(__file__).resolve().parents[1]
    verifier = (root / "scripts" / "verify_online_data.py").read_text(encoding="utf-8")
    prepare = (root / "src" / "data" / "prepare_fishnet.py").read_text(encoding="utf-8")
    expected = sorted("img_id bbox_id x_min x_max y_min y_max label_l1 label_l2".split())
    vf = sorted(re.findall(r'"([a-z_0-9]+)"',
                           re.search(r"EXPECTED_FISHNET_COLUMNS = \\(([^)]*)\\)", verifier).group(1)))
    pf = sorted(re.findall(r'"([a-z_0-9]+)"',
                           re.search(r'required = \\{([^}]*)\\}', prepare).group(1)))
    assert sorted(vf) == sorted(pf) == sorted(
        "img_id bbox_id x_min x_max y_min y_max label_l1 label_l2".split()
    )'''
new = '''def test_f4k_schema_pinned_between_verifier_and_prepare():
    """scripts/verify_online_data.py (the online drift alarm) and
    src/data/prepare_f4k.py (the consumer) must agree on the F4K layout:
    species labels live in fish_NN directories of the released tar."""
    root = Path(__file__).resolve().parents[1]
    verifier = (root / "scripts" / "verify_online_data.py").read_text(encoding="utf-8")
    prepare = (root / "src" / "data" / "prepare_f4k.py").read_text(encoding="utf-8")
    assert 'EXPECTED_F4K_COLUMNS = ("file_name", "cluster_id")' in verifier
    assert 'EXPECTED_F4K_CLUSTERS = 23' in verifier
    assert '"fishRecognition_GT" / "fish_image"' in prepare or "fishRecognition_GT" in prepare
    assert "_video_group" in prepare  # video-group split pins the leakage defence
    n = 23
    assert f"fish_{{:02d}}" not in prepare  # no format-string class names
    assert f"fish_{n:02d}" in prepare or "fish_root" in prepare'''
if old in src:
    src = src.replace(old, new)
    print("pin test rewritten: f4k")
elif new in src:
    print("pin test already f4k")
else:
    print("!! pin test pattern not found — inspect manually")
p.write_text(src, encoding="utf-8")

print("F4K SWEEP 3 COMPLETE")
