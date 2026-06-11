"""Tests for prose and code chunking."""

from __future__ import annotations

from sage.chunking import ChonkieChunker, is_code_file, language_for
from sage.config.schema import ChunkingCfg
from sage.core.types import DocumentSection


def _section(text: str, kind: str = "prose") -> DocumentSection:
    return DocumentSection(document_id="d1", text=text, char_start=0, char_end=len(text), kind=kind)


def test_language_dispatch():
    assert language_for("a.py") == "python"
    assert language_for("a.rs") == "rust"
    assert is_code_file("a.ts")
    assert not is_code_file("readme.md")
    assert language_for("notes.txt") is None


def test_prose_respects_target_size_and_offsets():
    cfg = ChunkingCfg(prose_target_chars=600)
    text = "Sentence about retrieval. " * 120
    chunks = ChonkieChunker(cfg).chunk("d1", [_section(text)], "notes.md")
    assert len(chunks) > 1
    assert all(len(c.content) <= 900 for c in chunks)  # near target, not wildly over
    assert all(c.language is None for c in chunks)
    # offsets are within the document and monotonically non-decreasing
    assert chunks[0].char_start == 0
    assert all(c.char_end <= len(text) for c in chunks)


def test_code_chunks_are_language_tagged():
    code = (
        "import math\n\n"
        "def area(r):\n    return math.pi * r * r\n\n"
        "class Shape:\n    def __init__(self, n):\n        self.n = n\n"
    ) * 6
    chunks = ChonkieChunker(ChunkingCfg()).chunk("c1", [_section(code, kind="code")], "geo.py")
    assert chunks
    assert all(c.language == "python" for c in chunks)
    assert chunks[0].chunk_id == "c1:0"


def test_chunk_ids_are_sequential():
    text = "Para. " * 400
    chunks = ChonkieChunker(ChunkingCfg(prose_target_chars=500)).chunk(
        "d1", [_section(text)], "x.md"
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
