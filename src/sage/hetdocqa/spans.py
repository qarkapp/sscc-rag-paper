"""Locate evidence text as character spans within a source document.

Generated questions cite evidence snippets; to produce chunker-agnostic gold labels
we map each snippet back to a character span in its source. Exact substring matches
are used when possible, falling back to a fuzzy best-window search for snippets that
were lightly paraphrased.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from sage.eval.span_mapping import GoldSpan

__all__ = ["locate_span", "snippets_to_spans"]


def locate_span(document: str, snippet: str, *, min_ratio: float = 0.8) -> tuple[int, int] | None:
    """Return ``(start, end)`` of ``snippet`` in ``document``, or ``None``.

    Tries exact match first, then a fuzzy alignment that accepts a best window with
    similarity at least ``min_ratio``.
    """
    snippet = snippet.strip()
    if not snippet:
        return None
    exact = document.find(snippet)
    if exact != -1:
        return (exact, exact + len(snippet))

    matcher = SequenceMatcher(None, document, snippet, autojunk=False)
    match = matcher.find_longest_match(0, len(document), 0, len(snippet))
    if match.size == 0:
        return None
    start = match.a - match.b
    end = start + len(snippet)
    start, end = max(0, start), min(len(document), max(match.a + match.size, end))
    window = document[start:end]
    if SequenceMatcher(None, window, snippet).ratio() >= min_ratio:
        return (start, end)
    return None


def snippets_to_spans(
    document_id: str, document: str, snippets: list[str], *, grade: int = 1
) -> list[GoldSpan]:
    """Map evidence snippets in one document to gold spans (unlocatable ones dropped)."""
    spans: list[GoldSpan] = []
    for snippet in snippets:
        located = locate_span(document, snippet)
        if located is not None:
            spans.append(GoldSpan(document_id, located[0], located[1], grade=grade))
    return spans
