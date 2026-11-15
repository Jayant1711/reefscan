"""Fish4Knowledge (F4K) downloader — official University of Edinburgh mirror.

Source: https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/
Archive layout (verified from the released tar):
    fish_image/fish_01..fish_23/*.png   27,394 fish crops, dir N = species cluster N
    mask_image/mask_01..mask_23/*.png   matching segmentation masks (bonus)
    class_id.csv (separate small file)  official annotation: file_name, cluster_id

The downloader fetches class_id.csv + the single fishRecognition_GT.tar (487 MB)
via resumable curl, verifies integrity against the published .md5, extracts, and
marks completion. See src/data/prepare_f4k.py for the curated ImageFolder build.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

F4K_BASE = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG"
F4K_CLASS_ID_URL = f"{F4K_BASE}/class_id.csv"
F4K_TAR_URL = f"{F4K_BASE}/Archive/fishRecognition_GT.tar"
F4K_TAR_MD5_URL = f"{F4K_BASE}/Archive/fishRecognition_GT.md5"


def _md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while data := fh.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> Path:
    """Resumable curl download with integrity expectations set by the caller."""
    if dest.exists():
        logger.info("File already present: %s — skipping download.", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    result = subprocess.run(
        ["curl", "-fL", "-C", "-", "--retry", "3", "--connect-timeout", "30", "-o", str(dest), url],
        check=False,
    )
    if result.returncode != 0 or not dest.exists():
        hint = ""
        if dest.exists():
            hint = " A partial file exists — if the server does not support ranges, delete it and re-run."
        raise RuntimeError(f"Download failed for {url} (curl exit {result.returncode}).{hint}")
    return dest


def download_f4k(data_dir: str | Path, keep_archive: bool = False) -> Path:
    """Download + extract the F4K ground-truth package (idempotent, crash-safe)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    sentinel = data_dir / "_EXTRACTED"
    extracted = data_dir / "fishRecognition_GT"
    if sentinel.exists():
        logger.info("F4K already extracted under %s — skipping.", data_dir)
        return extracted if extracted.exists() else data_dir

    class_id = _download(F4K_CLASS_ID_URL, data_dir / "class_id.csv")
    tar_path = _download(F4K_TAR_URL, data_dir / "fishRecognition_GT.tar")

    # Integrity: verify against the published .md5 before extracting.
    md5_url = F4K_TAR_MD5_URL
    try:
        expected = _download(md5_url, data_dir / "fishRecognition_GT.md5").read_text(encoding="utf-8").split()[0].strip().lower()
        actual = _md5_of(tar_path)
        if actual != expected:
            logger.warning("MD5 mismatch (expected %s, got %s) — deleting archive; re-run to re-download.", expected, actual)
            tar_path.unlink(missing_ok=True)
            raise RuntimeError("fishRecognition_GT.tar failed MD5 verification — deleted; re-run to resume/re-download.")
        logger.info("MD5 verified: %s", actual)
    except RuntimeError:
        raise
    except Exception as exc:  # md5 file unavailable — proceed with a warning
        logger.warning("Could not verify MD5 (%s) — continuing.", exc)

    import tarfile

    logger.info("Extracting F4K (fish images + masks, this can take several minutes) ...")
    with tarfile.open(tar_path, "r") as tf:
        try:
            tf.extractall(data_dir, filter="data")  # py3.12+: safe members
        except TypeError:  # py<3.11.4: manual member validation
            unsafe = [
                m for m in tf.getmembers()
                if m.issym() or m.islnk()
                or m.name.startswith(("/", "\\"))
                or ".." in m.name.replace("\\", "/").split("/")
            ]
            if unsafe:
                raise RuntimeError(f"Unsafe tar members: {[u.name for u in unsafe[:5]]}")
            tf.extractall(data_dir)
    sentinel.write_text("ok\n", encoding="utf-8")

    if not keep_archive:
        try:
            tar_path.unlink()
        except OSError:  # pragma: no cover
            pass

    n_fish = sum(1 for _ in (data_dir / "fishRecognition_GT" / "fish_image").rglob("*.png")) if (data_dir / "fishRecognition_GT" / "fish_image").exists() else 0
    logger.info("F4K ready: %d fish crops extracted (annotation: %s).", n_fish, class_id.name)
    return extracted if extracted.exists() else data_dir
