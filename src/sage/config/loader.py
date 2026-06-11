"""Load a :class:`PipelineConfig` from a YAML or JSON file.

Environment-variable interpolation (``${VAR}``) is supported in string values so
that secrets and machine-specific paths stay out of committed config files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from sage.config.schema import PipelineConfig
from sage.core.errors import ConfigError

__all__ = ["dump_config", "load_config"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_config(path: str | Path) -> PipelineConfig:
    """Parse and validate a pipeline configuration file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    return PipelineConfig.model_validate(_interpolate(raw))


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Write a configuration to a YAML file."""
    Path(path).write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
