# Datasheet: HetDocQA

## Motivation
A heterogeneous, multi-format retrieval benchmark (PDF, code, markdown, tabular, prose) for evaluating retrieval over realistic mixed-document collections, where most public benchmarks are homogeneous Wikipedia prose.

## Composition
- Questions: 762
- Type distribution: {'factual': 129, 'code': 135, 'cross_document': 191, 'multi_hop': 156, 'thematic': 151}
- Split distribution: {'dev': 195, 'test': 363, 'calibration': 204}
- Multi-evidence questions: 527
- Average gold spans per question: 2.18
- Source modalities: ['code', 'markdown', 'pdf', 'prose', 'table']
- Source licenses: ['Apache-2.0', 'BSD-3-Clause', 'CC-BY-SA-4.0', 'MIT', 'arXiv (see source_ref)']

## Collection process
Questions were drafted by a strong closed model (distinct from any model evaluated as a generator) over selected documents, then automatically filtered: a no-context answerability check removed questions answerable without retrieval, near-duplicates were removed by embedding similarity, and an LLM cross-validation pass checked evidence support, type label, and naturalness. A final human validation pass is applied before release.

## Labels
Gold evidence is annotated as character spans in the source documents and mapped to any system's chunks at evaluation time (>=50% span overlap), so retrieval metrics are independent of chunking.

## Splits
Calibration / dev / test splits are disjoint by collection, so thresholds tuned on dev cannot exploit corpus structure shared with test.

## Known limitations
Questions are LLM-drafted (then filtered and human-checked); English-only; domain coverage reflects the chosen sources.
