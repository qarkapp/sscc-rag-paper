"""Tests for the component registry."""

from __future__ import annotations

import pytest

from sage.core import registry
from sage.core.errors import RegistryError


def test_register_and_get():
    @registry.register("widget", "alpha")
    class Alpha:
        pass

    assert registry.get("widget", "alpha") is Alpha
    assert "alpha" in registry.available("widget")


def test_duplicate_registration_raises():
    @registry.register("gadget", "one")
    class One:
        pass

    with pytest.raises(RegistryError):

        @registry.register("gadget", "one")
        class Two:
            pass


def test_unknown_lookup_raises():
    with pytest.raises(RegistryError):
        registry.get("nonexistent", "whatever")
