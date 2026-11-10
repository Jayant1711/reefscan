"""Production patch 3: CLI wrapper, builder cell-6 relax + shlex, tests."""
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


# ---- 1) CLI friendly-error wrapper ----
cli = ROOT / "src/utils/cli.py"
cli.write_text('''"""Consistent, friendly error delivery for CLIs.

Real bugs still traceback; the common user errors (missing dataset/checkpoint,
bad flag values, CUDA OOM) become one clean line and exit code 1.

Usage:
    if __name__ == "__main__":
        raise SystemExit(run_cli(main))
"""

from __future__ import annotations

import logging


def run_cli(main_fn) -> int:
    """Run ``main_fn`` and translate expected user errors into clean messages."""
    logger = logging.getLogger("cli")
    try:
        return int(main_fn() or 0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except ModuleNotFoundError as exc:
        logger.error(
            "Missing dependency: %s — install requirements (pip install -r requirements.txt) "
            "or the optional extras named in the message.", exc,
        )
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            logger.error(
                "CUDA out of memory — lower the batch size, e.g. "
                "--set <task>.batch_size=16 (see docs/TRAINING.md)."
            )
            return 1
        logger.error("%s", exc)
        return 1
''', encoding="utf-8")
print("src/utils/cli.py written")

# adopt in the four data/eval CLIs (train.py has per-backbone handling already)
patch("evaluate.py", [
    ("if __name__ == \"__main__\":\n    raise SystemExit(main())",
     "if __name__ == \"__main__\":\n    from src.utils.cli import run_cli\n\n    raise SystemExit(run_cli(main))"),
], "evaluate.py wrapper")
patch("download_data.py", [
    ("if __name__ == \"__main__\":\n    raise SystemExit(main())",
     "if __name__ == \"__main__\":\n    from src.utils.cli import run_cli\n\n    raise SystemExit(run_cli(main))"),
], "download_data.py wrapper")
patch("clean_data.py", [
    ("if __name__ == \"__main__\":\n    raise SystemExit(main())",
     "if __name__ == \"__main__\":\n    from src.utils.cli import run_cli\n\n    raise SystemExit(run_cli(main))"),
], "clean_data.py wrapper")
patch("scripts/export_onnx.py", [
    ("if __name__ == \"__main__\":\n    raise SystemExit(main())",
     "if __name__ == \"__main__\":\n    from src.utils.cli import run_cli\n\n    raise SystemExit(run_cli(main))"),
], "export_onnx wrapper")
p = ROOT / "inference.py"
src = p.read_text(encoding="utf-8")
if 'raise SystemExit(main())' in src and "run_cli" not in src:
    src = src.replace(
        'if __name__ == "__main__":\n    raise SystemExit(main())',
        'if __name__ == "__main__":\n    from src.utils.cli import run_cli\n\n    raise SystemExit(run_cli(main))',
    )
    p.write_text(src, encoding="utf-8")
    print("inference.py wrapper: OK")

# ---- 2) builder: relax FE assert to >=2, non-fatal FT loop, shlex.quote ----
patch("scripts/build_platform_notebooks.py", [
    ('''fe = lb[lb["strategy"] == "feature_extraction"]
assert len(fe) == 4, f"FE screen incomplete: {len(fe)}/4 backbones have metrics"
top2 = fe.nlargest(2, "weighted_f1")["model"].tolist()
print("Fine-tuning top-2 from the FE screen:", top2)
for model in top2:
    subprocess.run([sys.executable, "train.py", "--task", "coral", "--model", model,
                    "--strategy", "finetune", "--platform", "kaggle"], check=True)''',
     '''fe = lb[lb["strategy"] == "feature_extraction"]
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
print("Fine-tune failures:", ft_failures or "none")'''),
    ('''import glob
sample = (glob.glob("/kaggle/tmp/reefscan_data/coral/imagefolders/test/*/*.jpg") +
          glob.glob("datasets/coral/imagefolders/test/*/*.jpg"))
if sample:
    !python evaluate.py --xai --xai-task coral --image {sample[0]}
else:
    print("No coral test images found — skipping XAI (not fatal).")''',
     '''import glob, shlex
sample = (glob.glob("/kaggle/tmp/reefscan_data/coral/imagefolders/test/*/*.jpg") +
          glob.glob("datasets/coral/imagefolders/test/*/*.jpg"))
if sample:
    !python evaluate.py --xai --xai-task coral --image {shlex.quote(str(sample[0]))}
else:
    print("No coral test images found — skipping XAI (not fatal).")'''),
    ('''import glob
sample = (glob.glob("/content/reefscan_data/fishnet/curated/test/*/*.jpg") or
          glob.glob("/content/reefscan_data/fishnet/curated/train/*/*.jpg"))[0]
!python evaluate.py --enhancement --data /content/reefscan_data/fishnet/curated/train
!python evaluate.py --xai --image {sample} || true''',
     '''import glob, shlex
samples = (glob.glob("/content/reefscan_data/fishnet/curated/test/*/*.jpg") or
           glob.glob("/content/reefscan_data/fishnet/curated/train/*/*.jpg"))
!python evaluate.py --enhancement --data /content/reefscan_data/fishnet/curated/train
if samples:
    !python evaluate.py --xai --image {shlex.quote(str(samples[0]))}
else:
    print("No curated fish images found — skipping XAI (not fatal).")'''),
], "builder cell-6 relax + shlex")

# fish FT loop non-fatal too
patch("scripts/build_platform_notebooks.py", [
    ('''winner = fe.nlargest(1, "weighted_f1")["model"].iloc[0]
print("Fine-tuning FE-screen winner:", winner)
subprocess.run([sys.executable, "train.py", "--task", "fish", "--model", winner,
                "--strategy", "finetune", "--platform", "colab"], check=True)''',
     '''winner = fe.nlargest(1, "weighted_f1")["model"].iloc[0]
print("Fine-tuning FE-screen winner:", winner)
r = subprocess.run([sys.executable, "train.py", "--task", "fish", "--model", winner,
                    "--strategy", "finetune", "--platform", "colab"])
if r.returncode != 0:
    print(f"WARNING: fine-tune failed for {winner} - evaluation below uses the FE checkpoint")'''),
], "builder fish FT non-fatal")

print("PROD PATCH 3 COMPLETE")
