"""Parent-chunk neighbour expansion.

Each leaf result is augmented with up to ``window`` characters from its immediately
adjacent chunks in the same document, giving the generator surrounding context
without changing ranking. Summary and corpus nodes are left untouched.
"""

from __future__ import annotations

import re
from dataclasses import replace

from sage.core.protocols import VectorStore
from sage.core.types import SearchResult

__all__ = ["expand_parent_chunks"]

_LEAF_ID = re.compile(r"^(?P<doc>.+):(?P<idx>\d+)$")


async def expand_parent_chunks(
    store: VectorStore, results: list[SearchResult], window: int
) -> list[SearchResult]:
    if window <= 0 or not results:
        return results

    neighbours: dict[str, tuple[str, int]] = {}
    wanted: set[str] = set()
    for r in results:
        match = _LEAF_ID.match(r.chunk_id)
        if r.level == 0 and match:
            doc, idx = match["doc"], int(match["idx"])
            neighbours[r.chunk_id] = (doc, idx)
            wanted.add(f"{doc}:{idx - 1}")
            wanted.add(f"{doc}:{idx + 1}")

    if not wanted:
        return results
    rows = await store.get_by_ids(sorted(wanted))
    by_id = {row.chunk_id: row.content for row in rows}

    expanded: list[SearchResult] = []
    for r in results:
        if r.chunk_id not in neighbours:
            expanded.append(r)
            continue
        doc, idx = neighbours[r.chunk_id]
        content = r.content
        if (prev := by_id.get(f"{doc}:{idx - 1}")) is not None:
            content = prev[-window:] + " " + content
        if (nxt := by_id.get(f"{doc}:{idx + 1}")) is not None:
            content = content + " " + nxt[:window]
        expanded.append(replace(r, content=content))
    return expanded
