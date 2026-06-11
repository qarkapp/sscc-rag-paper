"""Deterministic cache keys for backend calls.

A key is a SHA-256 over a canonical JSON encoding of everything that affects the
result: the call kind, provider, model, the normalized inputs, and the parameters.
A ``version`` field allows selective invalidation (e.g. when a prompt template
changes) without discarding unrelated cached data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["call_key", "canonical_json"]


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a stable, whitespace-free JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def call_key(
    kind: str,
    provider: str,
    model: str,
    payload: dict[str, Any],
    params: dict[str, Any] | None = None,
    *,
    version: str = "v1",
) -> str:
    """Compute the content-addressed cache key for a single backend call."""
    blob = canonical_json(
        {
            "kind": kind,
            "provider": provider,
            "model": model,
            "payload": payload,
            "params": params or {},
            "version": version,
        }
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
