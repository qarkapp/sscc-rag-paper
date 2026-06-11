"""Deterministic in-memory fakes for offline testing.

These implement the backend protocols without any network access, so the entire
pipeline can be exercised in CI deterministically.
"""

from sage.testing.fakes import FakeEmbedder, FakeGenerator, FakeReranker

__all__ = ["FakeEmbedder", "FakeGenerator", "FakeReranker"]
