"""Document chunking built on chonkie.

Prose is split with chonkie's character-based :class:`RecursiveChunker` (paragraph
and line aware); code is split with chonkie's AST-aware :class:`CodeChunker`, which
respects syntactic boundaries via tree-sitter. Both produce chunks sized in
characters so the configured targets map directly. Chunk character offsets are kept
relative to the source document so that benchmark gold spans can be matched against
chunks regardless of chunk size.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sage.chunking.languages import is_code_file, language_for
from sage.config.schema import ChunkingCfg
from sage.core.types import Chunk, DocumentSection

__all__ = ["ChonkieChunker"]


class ChonkieChunker:
    """Implements :class:`sage.core.protocols.Chunker`."""

    def __init__(self, cfg: ChunkingCfg) -> None:
        self._cfg = cfg
        self._prose: Any = None  # lazily constructed chonkie chunkers
        self._code: dict[str, Any] = {}

    def chunk(
        self, document_id: str, sections: Sequence[DocumentSection], filename: str
    ) -> list[Chunk]:
        code = is_code_file(filename)
        chunks: list[Chunk] = []
        for section in sections:
            pieces = (
                self._chunk_code(section.text, filename)
                if code
                else self._chunk_prose(section.text)
            )
            for text, rel_start, rel_end in pieces:
                if len(text.strip()) < self._cfg.prose_min_chars and not code:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}:{len(chunks)}",
                        document_id=document_id,
                        chunk_index=len(chunks),
                        content=text,
                        char_start=section.char_start + rel_start,
                        char_end=section.char_start + rel_end,
                        page_number=section.page_number,
                        section_name=section.section_name,
                        language=language_for(filename) if code else None,
                    )
                )
        return chunks

    # -- backends ----------------------------------------------------------

    def _chunk_prose(self, text: str) -> list[tuple[str, int, int]]:
        if not text.strip():
            return []
        if self._prose is None:
            from chonkie import RecursiveChunker

            self._prose = RecursiveChunker(
                tokenizer="character", chunk_size=self._cfg.prose_target_chars
            )
        return [(c.text, c.start_index, c.end_index) for c in self._prose(text)]

    def _chunk_code(self, text: str, filename: str) -> list[tuple[str, int, int]]:
        if not text.strip():
            return []
        lang = language_for(filename) or "auto"
        chunker: Any = self._code.get(lang)
        if chunker is None:
            from chonkie import CodeChunker

            chunker = CodeChunker(
                tokenizer="character",
                chunk_size=self._cfg.code_target_chars,
                language=lang,
            )
            self._code[lang] = chunker
        return [(c.text, c.start_index, c.end_index) for c in chunker(text)]
