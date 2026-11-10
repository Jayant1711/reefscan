"""DeepFish downloader (HuggingFace mirror of the James Cook University dataset).

Dataset: Saleh et al., Scientific Reports 10:14671 (2020), CC BY 4.0.
Mirror : https://huggingface.co/datasets/Alzayats/DeepFish  (single 7.1 GB tar)
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEEPFISH_REPO = "Alzayats/DeepFish"


def download_deepfish(data_dir: str | Path, keep_archive: bool = False) -> Path:
    """Download + extract DeepFish (Classification / Localization / Segmentation)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "DeepFish"
    sentinel = data_dir / "_EXTRACTED"
    if sentinel.exists() and extracted.exists() and any(extracted.rglob("*.jpg")):
        logger.info("DeepFish already extracted at %s — skipping.", extracted)
        return extracted

    from huggingface_hub import hf_hub_download

    logger.info("Downloading DeepFish.tar (~7.1 GB) from HuggingFace ...")
    tar_path = Path(
        hf_hub_download(repo_id=DEEPFISH_REPO, filename="DeepFish.tar", repo_type="dataset")
    )

    logger.info("Extracting DeepFish ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(data_dir)
    sentinel.write_text("ok\n", encoding="utf-8")  # marks a COMPLETE extraction

    if not keep_archive:
        try:
            Path(tar_path).unlink()
        except OSError:  # pragma: no cover
            pass

    if not extracted.exists():
        extracted = data_dir
    return extracted
