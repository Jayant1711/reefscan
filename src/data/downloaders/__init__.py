"""Per-dataset downloaders. Every downloader is resumable and idempotent."""

from .f4k import download_f4k
from .deepfish import download_deepfish
from .coral import download_coral

__all__ = ["download_f4k", "download_deepfish", "download_coral"]
