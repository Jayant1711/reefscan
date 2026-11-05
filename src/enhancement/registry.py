"""Registry of enhancement strategies (open for extension, closed for modification)."""

from __future__ import annotations

from typing import Callable, Type

from .base import Enhancer

ENHANCER_REGISTRY: dict[str, Type[Enhancer]] = {}


def register(cls: Type[Enhancer]) -> Type[Enhancer]:
    """Class decorator: ``@register`` adds the enhancer to the global registry."""
    if cls.name in ENHANCER_REGISTRY:
        raise KeyError(f"Enhancer '{cls.name}' already registered.")
    ENHANCER_REGISTRY[cls.name] = cls
    return cls


def get_enhancer(name: str, params: dict | None = None) -> Enhancer:
    """Instantiate a registered enhancer by name (lazily importing built-ins)."""
    if not ENHANCER_REGISTRY:
        from . import strategies  # noqa: F401  (registers all built-ins)
    if name not in ENHANCER_REGISTRY:
        raise KeyError(f"Unknown enhancer '{name}'. Available: {sorted(ENHANCER_REGISTRY)}")
    return ENHANCER_REGISTRY[name](params)


def build_enhancers(names: list[str] | None = None, params: dict | None = None) -> list[Enhancer]:
    """Build all (or selected) enhancers, importing strategy modules first."""
    from . import strategies  # noqa: F401  (registers all built-ins)

    names = names or list(ENHANCER_REGISTRY.keys())
    return [get_enhancer(n, params) for n in names]


def available() -> list[str]:
    from . import strategies  # noqa: F401

    return sorted(ENHANCER_REGISTRY.keys())
