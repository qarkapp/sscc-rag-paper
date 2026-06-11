"""Indexing pipeline: parse -> chunk -> embed -> store -> RAPTOR -> cross-document.

A document is parsed into sections, chunked, embedded, and stored as leaf rows; a
per-document RAPTOR tree is then built over the leaves. After all documents are
indexed, a corpus-level cross-document tier is built over the per-document top
summaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sage.config.schema import PipelineConfig
from sage.core.protocols import Chunker, Embedder, Generator, Parser, VectorStore
from sage.core.types import Chunk, StoreRow
from sage.raptor.cross_doc import build_cross_document_tier
from sage.raptor.tree import build_tree

__all__ = ["Document", "IndexingPipeline"]


@dataclass(frozen=True, slots=True)
class Document:
    """A document to index."""

    document_id: str
    filename: str
    data: bytes


class IndexingPipeline:
    """Builds the full index for a corpus of documents."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        generator: Generator,
        store: VectorStore,
        chunker: Chunker,
        parser: Parser,
        config: PipelineConfig,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._store = store
        self._chunker = chunker
        self._parser = parser
        self._cfg = config

    async def index_document(self, document: Document) -> list[StoreRow]:
        """Index one document and build its RAPTOR tree. Returns its top nodes."""
        sections = self._parser.parse(document.data, document.filename)
        chunks = self._chunker.chunk(document.document_id, sections, document.filename)
        leaves = await self._embed_leaves(chunks, document.filename)
        if not leaves:
            return []
        await self._store.upsert(leaves)

        if not self._cfg.raptor.enabled:
            return leaves

        summaries = await build_tree(
            leaves,
            document_id=document.document_id,
            embedder=self._embedder,
            generator=self._generator,
            store=self._store,
            cfg=self._cfg.raptor,
            seed=self._cfg.seed,
        )
        return self._top_nodes(summaries, leaves)

    async def index_corpus(self, documents: Sequence[Document]) -> None:
        """Index every document, then build the cross-document tier."""
        top_nodes: list[StoreRow] = []
        for document in documents:
            top_nodes.extend(await self.index_document(document))

        if self._cfg.raptor.enabled and self._cfg.raptor.cross_doc:
            await build_cross_document_tier(
                top_nodes,
                embedder=self._embedder,
                generator=self._generator,
                store=self._store,
                cfg=self._cfg.raptor,
                seed=self._cfg.seed,
            )

    # -- internals ---------------------------------------------------------

    async def _embed_leaves(self, chunks: list[Chunk], filename: str) -> list[StoreRow]:
        if not chunks:
            return []
        embeddings = await self._embedder.embed_documents([c.content for c in chunks])
        return [
            StoreRow(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embeddings[i],
                level=0,
                filename=filename,
                page_number=chunk.page_number,
                section_name=chunk.section_name,
                language=chunk.language,
            )
            for i, chunk in enumerate(chunks)
        ]

    @staticmethod
    def _top_nodes(summaries: list[StoreRow], leaves: list[StoreRow]) -> list[StoreRow]:
        if not summaries:
            return leaves  # too small to summarize; represent the doc by its leaves
        top_level = max(s.level for s in summaries)
        return [s for s in summaries if s.level == top_level]
