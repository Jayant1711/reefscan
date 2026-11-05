"""FishNet 2023 downloader (official Google Drive archive, ~15 GB).

Site: https://fishnet-2023.github.io/  (ICCV 2023)
Drive file id: 1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3

If gdown hits the Drive quota, the function prints manual-download
instructions and raises; users can drop the zip into the target folder and
re-run extraction only.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

FISHNET_DRIVE_ID = "1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3"
MANUAL_NOTE = """
FishNet download blocked or quota-limited. Manual fallback:
  1. Open https://drive.google.com/file/d/1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3/view
  2. Download the archive with a browser (Google account may be required).
  3. Place the file at: {zip_path}
  4. Re-run this script — extraction will resume automatically.
"""


def download_fishnet(raw_dir: str | Path, keep_archive: bool = False) -> Path:
    """Download + extract FishNet into ``raw_dir``. Returns the extracted root."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "FishNet.zip"
    extracted = raw_dir / "FishNet"

    if extracted.exists() and any(extracted.iterdir()):
        logger.info("FishNet already extracted at %s — skipping.", extracted)
        return extracted

    if not zip_path.exists():
        logger.info("Downloading FishNet (~15 GB) via gdown ...")
        try:
            import gdown

            gdown.download(id=FISHNET_DRIVE_ID, output=str(zip_path), quiet=False)
        except Exception as exc:
            logger.error("gdown failed: %s%s", exc, MANUAL_NOTE.format(zip_path=zip_path))
            raise
    else:
        logger.info("Found existing archive %s — extracting.", zip_path)

    logger.info("Extracting FishNet ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    if not keep_archive:
        try:
            zip_path.unlink()
        except OSError:  # pragma: no cover
            pass

    if not extracted.exists():
        # Some archives unpack contents directly into raw_dir.
        extracted = raw_dir
    return extracted
