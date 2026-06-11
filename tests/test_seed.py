"""Tests for deterministic seeding."""

from __future__ import annotations

from sage.config.seed import rng_for, set_global_seed


def test_rng_is_deterministic():
    set_global_seed(42)
    a = rng_for("router").standard_normal(5)
    set_global_seed(42)
    b = rng_for("router").standard_normal(5)
    assert (a == b).all()


def test_named_streams_are_independent():
    set_global_seed(42)
    a = rng_for("router").standard_normal(5)
    b = rng_for("graph").standard_normal(5)
    assert not (a == b).all()


def test_seed_changes_stream():
    set_global_seed(1)
    a = rng_for("router").standard_normal(5)
    set_global_seed(2)
    b = rng_for("router").standard_normal(5)
    assert not (a == b).all()
