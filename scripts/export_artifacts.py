#!/usr/bin/env python
"""Zip all trained artefacts for download — RUN BEFORE CLOSING ANY SESSION.

Creates ``reefscan_artifacts_<platform>_<sha8>_<date>.zip`` containing
models/, results/, runs/ and reports/, plus a MANIFEST.txt. On Kaggle and
Colab the zip is placed at a stable, easy-to-download location and the exact
download snippet is printed.

Usage:
    python scripts/export_artifacts.py [--out /tmp] [--no-models]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.provenance import detect_platform, git_sha  # noqa: E402

ARTIFACT_DIRS = ["models", "results", "runs", "reports"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Directory for the zip (default: repo root)")
    parser.add_argument("--no-models", action="store_true", help="Exclude model checkpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    platform = detect_platform()
    sha8 = git_sha(root)[:8]
    date = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_name = f"reefscan_artifacts_{platform}_{sha8}_{date}.zip"
    out_dir = Path(args.out) if args.out else root
    zip_path = out_dir / zip_name

    manifest_lines = [
        "ReefScan AI artefact bundle",
        f"created_utc : {datetime.now(timezone.utc).isoformat()}",
        f"platform    : {platform}",
        f"git_sha     : {git_sha(root)}",
        "",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dir_name in ARTIFACT_DIRS:
            if args.no_models and dir_name == "models":
                continue
            folder = root / dir_name
            if not folder.exists():
                continue
            for file_path in sorted(folder.rglob("*")):
                if not file_path.is_file():
                    continue
                if file_path.suffix in {".pt", ".pth"} and args.no_models:
                    continue
                # Skip demo placeholder checkpoints (no trained value).
                if "models/demo" in file_path.as_posix():
                    continue
                arcname = file_path.relative_to(root)
                zf.write(file_path, arcname=str(arcname))
        zf.writestr("MANIFEST.txt", "\n".join(manifest_lines))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n✅ Artefacts zipped: {zip_path}  ({size_mb:.1f} MB)\n")

    if platform == "kaggle":
        # /kaggle/working is the notebook Output panel — no code needed.
        print("📥 DOWNLOAD (Kaggle):")
        print(f"   The zip is inside /kaggle/working -> open the notebook 'Output' panel")
        print(f"   and download {zip_name}.")
        print("   Or in a cell:  import shutil; shutil.move(str(zip_path), '/kaggle/working/')"
              if out_dir != Path("/kaggle/working") else "")
    elif platform == "colab":
        print("📥 DOWNLOAD (Colab) — run in the next cell:")
        print("   from google.colab import files")
        print(f'   files.download("{zip_path}")')
    elif platform == "lightning":
        print("📥 DOWNLOAD (Lightning): open the studio file browser and download:")
        print(f"   {zip_path}")
    else:
        print(f"📥 Local machine — zip already at: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
