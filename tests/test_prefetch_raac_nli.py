"""Tests for speculative prefetching, adaptive chunking, and entailment edges."""

from __future__ import annotations

import numpy as np

from sage.chunking.raac import ChunkStats, RaacOpType, plan_operations
from sage.config.schema import PrefetchCfg, RaacCfg
from sage.core.types import Chunk
from sage.graph.nli_edges import EntailmentLabel, build_entailment_edges
from sage.graph.traversal import entailment_chains
from sage.prefetch import PrefetchBuffer, SpeculativeRetrievalPrefetcher, extract_entities
from sage.testing import FakeEmbedder

# -- SRP -------------------------------------------------------------------


def test_extract_entities():
    ents = extract_entities("Models like NASA data show that Paris differs from a generic city.")
    assert "NASA" in ents
    assert "Paris" in ents
    assert "city" not in ents  # lowercase common words are not entities


async def test_prefetch_buffer_hit_and_miss():
    embedder = FakeEmbedder(dim=16)
    buffer = PrefetchBuffer(threshold=0.99)
    vec = await embedder.embed_query("Paris")
    buffer.put(vec, [])
    assert buffer.lookup(vec) is not None  # identical vector -> hit
    assert buffer.lookup(await embedder.embed_query("Tokyo")) is None  # unrelated -> miss


async def test_srp_prefetches_and_serves():
    embedder = FakeEmbedder(dim=16)
    calls: list[str] = []

    async def retrieve(query: str):
        calls.append(query)
        return []

    srp = SpeculativeRetrievalPrefetcher(PrefetchCfg(hit_cosine_threshold=0.99), embedder, retrieve)
    n = await srp.observe("According to NASA and the Hubble Telescope, the result holds.")
    assert n >= 2
    assert "NASA" in calls
    # A later query matching a prefetched entity is served from the buffer.
    hit = await srp.maybe_hit(await embedder.embed_query("NASA"))
    assert hit is not None
    assert srp.metrics.hit_rate == 1.0


# -- RAAC ------------------------------------------------------------------


def test_raac_plans_split_merge_reanchor():
    stats = [
        ChunkStats(
            "d:0",
            "d",
            0,
            hit_rate=0.8,
            precision=0.2,
            co_retrieval_entropy=0.9,
            generation_utility=0.5,
        ),
        ChunkStats(
            "d:1",
            "d",
            1,
            hit_rate=0.9,
            precision=0.9,
            co_retrieval_entropy=0.05,
            generation_utility=0.4,
        ),
        ChunkStats(
            "d:2",
            "d",
            2,
            hit_rate=0.9,
            precision=0.9,
            co_retrieval_entropy=0.05,
            generation_utility=0.4,
        ),
        ChunkStats(
            "d:3",
            "d",
            3,
            hit_rate=0.7,
            precision=0.9,
            co_retrieval_entropy=0.9,
            generation_utility=0.0,
        ),
    ]
    ops = plan_operations(stats, RaacCfg())
    kinds = {(op.op_type, op.chunk_ids) for op in ops}
    assert (RaacOpType.SPLIT, ("d:0",)) in kinds  # high hit, low precision
    assert (RaacOpType.MERGE, ("d:1", "d:2")) in kinds  # adjacent, always co-retrieved
    assert (RaacOpType.RE_ANCHOR, ("d:3",)) in kinds  # retrieved but unused


# -- NLI entailment --------------------------------------------------------


def _chunk(cid: str, content: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id="d", chunk_index=int(cid[-1]), content=content)


def test_entailment_edges_and_chains():
    chunks = [_chunk("c0", "premise text"), _chunk("c1", "elaborated text"), _chunk("c2", "more")]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32)

    def classify(premise: str, hypothesis: str):
        return EntailmentLabel.ENTAILMENT, 0.9

    edges = build_entailment_edges(chunks, embeddings, classify, cos_gate=0.5)
    assert edges
    assert all(e.label is EntailmentLabel.ENTAILMENT for e in edges)

    chains = entailment_chains(edges, ["c0"], max_hops=3)
    assert chains
    # the best chain starts at the seed and has score = product of confidences
    best_path, best_score = chains[0]
    assert best_path[0] == "c0"
    assert 0.0 < best_score <= 1.0


def test_entailment_neutral_pairs_are_dropped():
    chunks = [_chunk("c0", "a"), _chunk("c1", "b")]
    embeddings = np.array([[1.0, 0.0], [0.99, 0.0]], dtype=np.float32)

    def classify(premise: str, hypothesis: str):
        return EntailmentLabel.NEUTRAL, 0.5

    assert build_entailment_edges(chunks, embeddings, classify, cos_gate=0.5) == []
