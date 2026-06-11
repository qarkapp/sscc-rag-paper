"""On-disk cache for backend calls.

Every embedding, rerank, generation, and judge call is content-addressed and
replayable, so a full experiment re-runs offline at near-zero cost. A SQLite index
holds metadata; payloads are stored as content-addressed blobs (``.npy`` for
arrays, ``.json`` for everything else) so the database stays small.

Cache modes:
    * ``read_write`` -- read hits, write misses (default)
    * ``read_only``  -- read hits, never call the backend on a miss (raise)
    * ``refresh``    -- ignore existing entries, recompute and overwrite
    * ``off``        -- bypass the cache entirely
"""

from __future__ import annotations

import json
import sqlite3
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from sage.core.errors import SageError

__all__ = ["CacheMissError", "CacheMode", "CallCache"]


class CacheMode(StrEnum):
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    REFRESH = "refresh"
    OFF = "off"


class CacheMissError(SageError):
    """Raised in ``read_only`` mode when a key is absent."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    blob_path  TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_kind ON calls(kind);
"""


class CallCache:
    """Thread-safe SQLite-backed cache with content-addressed blob storage."""

    def __init__(
        self,
        root: str | Path,
        mode: CacheMode = CacheMode.READ_WRITE,
        *,
        clock: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.mode = mode
        self._clock = clock  # injected timestamp keeps writes deterministic
        self._blobs = self.root / "blobs"
        self._lock = threading.Lock()
        if mode is not CacheMode.OFF:
            self._blobs.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.root / "calls.db", check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached payload for ``key`` or ``None`` if absent."""
        if self.mode in (CacheMode.OFF, CacheMode.REFRESH):
            return None
        with self._lock:
            row = self._conn.execute("SELECT blob_path FROM calls WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return self._load_blob(self.root / row[0])

    def require(self, key: str) -> Any:
        """Return the cached payload or raise :class:`CacheMiss`."""
        value = self.get(key)
        if value is None and self.mode is CacheMode.READ_ONLY:
            raise CacheMissError(f"cache miss for {key!r} in read_only mode")
        return value

    def put(self, key: str, kind: str, provider: str, model: str, payload: Any) -> None:
        """Store ``payload`` under ``key`` (no-op when caching is off)."""
        if self.mode is CacheMode.OFF:
            return
        blob_rel = self._write_blob(key, payload)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls "
                "(key, kind, provider, model, blob_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, kind, provider, model, str(blob_rel), self._clock),
            )
            self._conn.commit()

    def close(self) -> None:
        if self.mode is not CacheMode.OFF:
            self._conn.close()

    # -- blob handling -----------------------------------------------------

    def _blob_dir(self, key: str) -> Path:
        return self._blobs / key[:2]

    def _write_blob(self, key: str, payload: Any) -> Path:
        target_dir = self._blob_dir(key)
        target_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, np.ndarray):
            final = target_dir / f"{key}.npy"
            tmp = final.with_suffix(".npy.tmp")
            # Use a file handle so numpy does not append a second ``.npy``.
            with tmp.open("wb") as fh:
                np.save(fh, payload.astype(np.float32, copy=False))
        else:
            final = target_dir / f"{key}.json"
            tmp = final.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(final)  # atomic
        return final.relative_to(self.root)

    @staticmethod
    def _load_blob(path: Path) -> Any:
        if path.suffix == ".npy":
            return np.load(path)
        return json.loads(path.read_text(encoding="utf-8"))
