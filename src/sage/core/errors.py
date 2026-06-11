"""Exception hierarchy for the package."""

from __future__ import annotations

__all__ = [
    "BackendError",
    "ConfigError",
    "EmbeddingDimensionError",
    "RegistryError",
    "SageError",
]


class SageError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(SageError):
    """Raised when a configuration is invalid or internally inconsistent."""


class RegistryError(SageError):
    """Raised when a component cannot be resolved from the registry."""


class BackendError(SageError):
    """Raised when an external backend (oMLX/OpenRouter) call fails terminally."""


class EmbeddingDimensionError(BackendError):
    """Raised when an embedding backend returns an unexpected dimensionality."""
