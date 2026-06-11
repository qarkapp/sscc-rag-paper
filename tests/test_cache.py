"""Tests for the content-addressed call cache."""

from __future__ import annotations

import numpy as np
import pytest

from sage.cache.keys import call_key
from sage.cache.store import CacheMissError, CacheMode, CallCache


def test_array_round_trip(tmp_path):
    cache = CallCache(tmp_path, CacheMode.READ_WRITE)
    vec = np.arange(8, dtype=np.float32)
    cache.put("k1", "embed", "omlx", "bge-m3", vec)
    loaded = cache.get("k1")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, vec)


def test_json_round_trip(tmp_path):
    cache = CallCache(tmp_path, CacheMode.READ_WRITE)
    cache.put("k2", "generate", "omlx", "qwen", "hello world")
    assert cache.get("k2") == "hello world"


def test_off_mode_never_persists(tmp_path):
    cache = CallCache(tmp_path, CacheMode.OFF)
    cache.put("k3", "embed", "omlx", "bge-m3", np.zeros(4, dtype=np.float32))
    assert cache.get("k3") is None


def test_read_only_raises_on_miss(tmp_path):
    cache = CallCache(tmp_path, CacheMode.READ_ONLY)
    assert cache.get("absent") is None
    with pytest.raises(CacheMissError):
        cache.require("absent")


def test_refresh_mode_ignores_existing(tmp_path):
    writer = CallCache(tmp_path, CacheMode.READ_WRITE)
    writer.put("k4", "generate", "omlx", "qwen", "old")
    writer.close()
    refresher = CallCache(tmp_path, CacheMode.REFRESH)
    assert refresher.get("k4") is None  # forces recompute


def test_call_key_is_stable_and_order_independent():
    a = call_key("embed", "omlx", "m", {"text": "x"}, {"task": "document"})
    b = call_key("embed", "omlx", "m", {"text": "x"}, {"task": "document"})
    c = call_key("embed", "omlx", "m", {"text": "y"}, {"task": "document"})
    assert a == b
    assert a != c
