"""Speculative-prefetch effectiveness metrics."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PrefetchMetrics"]


@dataclass(slots=True)
class PrefetchMetrics:
    """Tracks prefetch hit rate over a sequence of lookups."""

    hits: int = 0
    total: int = 0

    def record(self, hit: bool) -> None:
        self.total += 1
        if hit:
            self.hits += 1

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0
