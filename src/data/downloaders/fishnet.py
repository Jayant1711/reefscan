"""FishNet downloader (official S3 archives, https://www.fishnet.ai/download).

Versioned releases are hosted on Amazon S3 with direct, quota-free links:

    v1.0.0  images 36.4 GB + labels 10 MB   (July 2022, full FishNet)
    v0.2.0  images 12.9 GB + labels 5.3 MB  (September 2020)
    v0.1.2  images  5.7 GB + labels 2.1 MB  (January 2020)
    v0.1.1  images  5.7 GB + labels 2.1 MB  (May 2019)

Default: v0.1.2 (smallest full-release with species labels) — configurable
via ``version`` argument. Images land in ``raw_dir``, labels zip is
extracted alongside and left for :mod:`src.data.prepare_fishnet`.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Official S3 mirrors — see https://www.fishnet.ai/download
S3_BASE = "https://fishnet-data.s3-us-west-1.amazonaws.com"
FISHNET_RELEASES: dict[str, dict[str, str]] = {
    "v1.0.0": {
        "images": f"{S3_BASE}/foid_images_v100.zip",
        "labels": f"{S3_BASE}/foid_labels_v100.zip",
    },
    "v0.2.0": {
        "images": f"{S3_BASE}/foid_images_v020.zip",
        "labels": f"{S3_BASE}/foid_labels_v020.zip",
    },
    "v0.1.2": {
        "images": f"{S3_BASE}/foid_images_v012.zip",
        "labels": f"{S3_BASE}/foid_labels_v012.zip",
    },
    "v0.1.1": {
        "images": f"{S3_BASE}/ofid_images_v011.zip",
        "labels": f"{S3_BASE}/ofid_labels_v011.zip",
    },
}
DEFAULT_VERSION = "v0.1.2"


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` with curl (resumable, no quota limits)."""
    if dest.exists():
        logger.info("Archive already present: %s — skipping download.", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    import subprocess

    # curl -C - resumes partial downloads; Colab images ship curl.
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "3", "-o", str(dest), url],
        check=False,
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(f"Download failed for {url} (curl exit {result.returncode})")
    return dest


def download_fishnet(
    raw_dir: str | Path,
    version: str = DEFAULT_VERSION,
    keep_archive: bool = False,
) -> Path:
    """Download + extract a FishNet release into ``raw_dir``. Returns the extracted root."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if version not in FISHNET_RELEASES:
        raise KeyError(f"Unknown FishNet version '{version}'. Available: {sorted(FISHNET_RELEASES)}")
    urls = FISHNET_RELEASES[version]

    images_zip = raw_dir / f"fishnet_{version}_images.zip"
    labels_zip = raw_dir / f"fishnet_{version}_labels.zip"
    extracted = raw_dir / "FishNet"

    if extracted.exists() and any(extracted.iterdir()):
        logger.info("FishNet already extracted at %s — skipping.", extracted)
        return extracted

    _download(urls["labels"], labels_zip)
    _download(urls["images"], images_zip)

    logger.info("Extracting FishNet labels ...")
    with zipfile.ZipFile(labels_zip) as zf:
        zf.extractall(raw_dir)

    logger.info("Extracting FishNet images (this can take several minutes) ...")
    with zipfile.ZipFile(images_zip) as zf:
        zf.extractall(raw_dir)

    if not keep_archive:
        for archive in (images_zip, labels_zip):
            try:
                archive.unlink()
            except OSError:  # pragma: no cover
                pass

    if not extracted.exists():
        # Some archives unpack contents directly into raw_dir.
        extracted = raw_dir
    return extracted
