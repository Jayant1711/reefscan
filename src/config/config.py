"""Configuration management: YAML loading with dot-access and CLI overrides.

Example
-------
>>> cfg = load_config("configs/default.yaml", overrides=["fish.epochs=3"])
>>> cfg.fish.epochs
3
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class DotDict(dict):
    """Dictionary with attribute-style access, recursively applied."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        for key, value in dict(*args, **kwargs).items():
            self[key] = self.wrap(value)

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"Config has no key '{key}'") from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo: dict) -> "DotDict":
        return DotDict(copy.deepcopy(dict(self), memo))

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def get_nested(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value via a dotted path, e.g. ``fish.epochs``."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_nested(self, dotted: str, value: Any) -> None:
        """Set a nested value via a dotted path (created intermediates stay DotDicts)."""
        parts = dotted.split(".")
        node: MutableMapping[str, Any] = self
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, MutableMapping):
                child = DotDict()
                node[part] = child
            node = child
        node[parts[-1]] = value


def _coerce_scalar(text: str) -> Any:
    """Coerce a CLI string into int/float/bool/null/list when possible."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(t) for t in inner.split(",")]
    return text


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place on a copy)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: Sequence[str] | None = None,
) -> DotDict:
    """Load a YAML config and apply ``--set key=value`` style overrides.

    Parameters
    ----------
    path:
        YAML file path. Falls back to the packaged default config.
    overrides:
        Sequence of ``section.key=value`` strings (dotted paths supported).
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logger.warning("Config '%s' not found; using packaged default.", cfg_path)
        cfg_path = DEFAULT_CONFIG_PATH

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = DotDict.wrap(raw)

    for token in overrides or []:
        if "=" not in token:
            raise ValueError(f"Invalid override '{token}'. Expected key=value.")
        key, _, value = token.partition("=")
        cfg.set_nested(key.strip(), _coerce_scalar(value))
        logger.debug("Override %s=%r", key.strip(), cfg.get_nested(key.strip()))

    # Resolve all paths relative to the project root for portability.
    root = Path(__file__).resolve().parents[2]
    if not cfg.get_nested("paths.data_root", "").startswith(("datasets", "models", "results", "reports")):
        cfg.paths.data_root = str(root / cfg.paths.data_root)

    return cfg


def ensure_output_dirs(cfg: DotDict) -> None:
    """Create models/results/reports directories if missing."""
    for key in ("models_dir", "results_dir", "reports_dir"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)


def override_from_args(cfg: DotDict, args: Any) -> DotDict:
    """Apply ``args.set`` (list of key=value) onto a loaded config."""
    sets = getattr(args, "set", None) or []
    for token in sets:
        if "=" not in token:
            raise ValueError(f"Invalid --set value '{token}'. Expected key=value.")
        key, _, value = token.partition("=")
        cfg.set_nested(key.strip(), _coerce_scalar(value))
    return cfg
