"""Per-dataset downloaders. Every downloader is resumable and idempotent."""

from .fishnet import download_fishnet
from .deepfish import download_deepfish
from .coral import download_coral

__all__ = ["download_fishnet", "download_deepfish", "download_coral"]
