"""Component registry.

Concrete components register themselves under a ``(category, name)`` pair. The
pipeline assembler resolves components from configuration through this registry, so
no module needs to import concrete component classes directly -- which is what keeps
ablations to configuration changes and avoids import cycles.

Registration is via decorator::

    @register("router", "egr")
    class EntropyGatedRouter: ...

and resolution via :func:`get`::

    cls = get("router", config.router.variant)
    router = cls(config.router, ...)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from sage.core.errors import RegistryError

__all__ = ["available", "categories", "get", "register"]

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}


def register(category: str, name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a component under ``(category, name)``."""

    def decorator(cls: type[T]) -> type[T]:
        bucket = _REGISTRY.setdefault(category, {})
        if name in bucket:
            raise RegistryError(f"{category!r}/{name!r} is already registered")
        bucket[name] = cls
        return cls

    return decorator


def get(category: str, name: str) -> type:
    """Resolve a registered component class, or raise a helpful error."""
    bucket = _REGISTRY.get(category)
    if not bucket or name not in bucket:
        known = sorted(bucket) if bucket else []
        raise RegistryError(f"no component registered for {category!r}/{name!r}; known: {known}")
    return bucket[name]


def available(category: str) -> Iterable[str]:
    """Names registered under ``category``."""
    return sorted(_REGISTRY.get(category, {}))


def categories() -> Iterable[str]:
    """All registered categories."""
    return sorted(_REGISTRY)
