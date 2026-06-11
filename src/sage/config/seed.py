"""Centralized seeding for reproducible runs.

Every stochastic component draws its randomness from a named, derived generator so
that runs are reproducible and independent components do not interfere with one
another's streams.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

__all__ = ["rng_for", "set_global_seed"]

_GLOBAL_SEED = 42


def set_global_seed(seed: int) -> None:
    """Seed the global Python/NumPy RNGs and hash randomization.

    Torch (if installed) is seeded lazily to avoid importing it here.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


def rng_for(name: str) -> np.random.Generator:
    """Return a NumPy generator seeded deterministically from ``name``.

    Deriving per-component seeds from a stable hash of the name keeps each
    component's stream independent of call order elsewhere in the pipeline.
    """
    digest = hashlib.sha256(f"{_GLOBAL_SEED}:{name}".encode()).digest()
    derived = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(derived)
