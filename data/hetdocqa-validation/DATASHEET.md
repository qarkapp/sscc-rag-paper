# Datasheet: HetDocQA

## Motivation
A heterogeneous, multi-format retrieval benchmark (PDF, code, markdown, tabular, prose) for evaluating retrieval over realistic mixed-document collections, where most public benchmarks are homogeneous Wikipedia prose.

## Composition
- Questions: 16
- Type distribution: {'factual': 4, 'code': 2, 'multi_hop': 3, 'thematic': 4, 'cross_document': 3}
- Split distribution: {'calibration': 4, 'dev': 3, 'test': 9}
- Multi-evidence questions: 10
- Average gold spans per question: 2.62
- Source modalities: ['code', 'markdown', 'pdf', 'prose', 'table']
- Source licenses: ['Apache-2.0', 'CC-BY-SA-4.0', 'arXiv-nonexclusive', 'validation-only']

## Collection process
Questions were drafted by a strong closed model (distinct from any model evaluated as a generator) over selected documents, then automatically filtered: a no-context answerability check removed questions answerable without retrieval, near-duplicates were removed by embedding similarity, and an LLM cross-validation pass checked evidence support, type label, and naturalness. A final human validation pass is applied before release.

## Labels
Gold evidence is annotated as character spans in the source documents and mapped to any system's chunks at evaluation time (>=50% span overlap), so retrieval metrics are independent of chunking.

## Splits
Calibration / dev / test splits are disjoint by collection, so thresholds tuned on dev cannot exploit corpus structure shared with test.

## Known limitations
Questions are LLM-drafted (then filtered and human-checked); English-only; domain coverage reflects the chosen sources.
