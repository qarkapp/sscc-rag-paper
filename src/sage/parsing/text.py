"""Plain-text, markdown, and source-code parsing."""

from __future__ import annotations

from sage.chunking.languages import is_code_file
from sage.core.types import DocumentSection

__all__ = ["TextParser"]


class TextParser:
    """Decode UTF-8 text into a single section.

    Implements :class:`sage.core.protocols.Parser`. Structure-aware splitting is the
    chunker's responsibility; the parser only recovers text and marks code vs prose.
    """

    def parse(self, data: bytes, filename: str) -> list[DocumentSection]:
        text = data.decode("utf-8", errors="replace")
        if not text:
            return []
        return [
            DocumentSection(
                document_id=filename,
                text=text,
                char_start=0,
                char_end=len(text),
                kind="code" if is_code_file(filename) else "prose",
            )
        ]
