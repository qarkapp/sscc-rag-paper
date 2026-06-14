# Paper outline

**Framing (locked):** analysis-led controlled study. HetDocQA (benchmark) and SSCC
(the one robust win) are major *supporting* contributions, not the headline. The
intellectual core is the reranker-dominance principle that explains a wall of negatives.

**Working title:** *Beyond the Reranker: Where Retrieval-Augmented Generation Gains
Actually Live on Heterogeneous Documents*

Alternatives: "What Actually Helps RAG on Heterogeneous Documents? A Controlled Study";
"When the Reranker Is (Almost) Enough."

**One-paragraph thesis (abstract sketch):**
A wave of retrieval-augmented-generation (RAG) enhancements -- hierarchical
summarization (RAPTOR), cross-document tiers, graph expansion, query routing,
hypothesis fusion, corrective re-retrieval -- are reported to help, but almost always
on homogeneous prose benchmarks. We ask whether they help on *realistic heterogeneous*
collections (code, tables, prose, PDFs). We build HetDocQA, a heterogeneous
multi-format QA benchmark with chunker-agnostic span labels, and run a controlled,
shared-backbone study of these enhancements across HetDocQA and two standard benchmarks
(MuSiQue, QASPER), with bootstrap CIs and multiple-comparison correction. Under
correction, **almost none of the enhancements provide a reliable benefit**; a strong
cross-encoder reranker dominates, and the only robust gains come from *outside*
re-ranking -- specifically a per-source calibrated corrector (SSCC) whose benefit is
itself heterogeneity-specific. We explain the pattern with a simple principle --
**a strong reranker already extracts the pool's signal, so headroom lives upstream
(query) or in the decision layer, not in re-ranking or hierarchical expansion** -- and
show several reported gains are implementation-fragile (they vanish under correct
reimplementation).

---

## Contributions (state explicitly in intro)
1. **HetDocQA** -- heterogeneous multi-format QA benchmark; chunker-agnostic
   span-overlap labels; collection-disjoint splits; datasheet. Released.
2. **A controlled, multi-benchmark study** of 8+ popular RAG enhancements with a shared
   embedder/reranker/generator, bootstrap CIs, and Holm-Bonferroni/BH-FDR correction.
3. **The reranker-dominance principle**: gains live upstream (query) or in the decision
   layer; re-ranking/expansion enhancements do not survive correction.
4. **SSCC** (per-source calibrated correction): the one robust, heterogeneity-specific
   positive -- the exception that proves the rule.
5. **Implementation-fragility finding**: several "gains" are valid negatives only after
   fixing faithful-reimplementation bugs (cross-doc tier level; PPR).

---

## Section flow

### 1. Introduction
- The enhancement proliferation + the homogeneity blind spot of current benchmarks.
- The question: do these help on heterogeneous real-world documents?
- Contributions list above. Headline finding up front (most don't help; reranker
  dominates; one decision-layer exception).

### 2. Related Work
- RAG enhancements: HyDE; RAPTOR/GraphRAG hierarchical+graph; CRAG/Self-RAG correction;
  Adaptive-RAG/routing; RRF fusion.
- Benchmarks (NQ/HotpotQA/MuSiQue/QASPER/QuALITY) and their prose homogeneity.
- Reproducibility & negative-results studies in IR/NLP.

### 3. HetDocQA
- Sources (permissive licenses), four+ modalities (pdf/prose/markdown/code/table),
  mixed-format collections; build-from-pointers reproducibility.
- **Span-based, chunker-agnostic relevance** (>=50% overlap) -- the fairness mechanism.
- Generation (strong distinct model) -> answerability/decontamination filter ->
  near-dup removal -> human validation. **Honestly state** current validation level
  (single-pass human; IAA/kappa as a limitation / future strengthening).
- Collection-disjoint calibration/dev/test (762 Q). Datasheet in appendix.

### 4. Experimental Setup
- Shared backbone: bge-m3 embedder, jina-reranker-v3, gpt-4.1-mini generator (fixed).
- Components under test + how each is isolated (one config diff per ablation).
- Fairness controls; metrics (Hit/Recall/nDCG@10, MRR; EM/token-F1); statistics
  (>=10k bootstrap, paired bootstrap, Holm-Bonferroni, BH-FDR); honesty firewall
  (no test tuning; generator distinct from HetDocQA generator + judge).

### 5. Results: what helps and what doesn't
- Main table: HetDocQA test ablation (full + each -component) with corrected p.
  Survivors of correction: rerank, HyDE, SSCC, floor. (RAPTOR survives on test but
  flips sign vs dev -> flagged unstable.) `results/hetdocqa_test_ablation.txt`
- Replication across benchmarks: MuSiQue + QASPER ablations; Holm survivors per set
  (MuSiQue {hyde,rerank,floor}; QASPER {rerank}). `results/{musique,qasper}_ablation.txt`
- Routing triad (keyword/EGR/compositional/oracle): no answer-F1 headroom anywhere;
  EGR worse than trivial. 
- SSCC heterogeneity-specificity: significant on HetDocQA (dev p=.011, test p=.005,
  survives Holm), inert on MuSiQue/QASPER -> the homogeneous controls are the evidence.

### 6. Analysis: why
- **Reranker-dominance principle.** wo_rerank collapses everything (nDCG -> ~0); given
  a strong reranker, re-ranking the same pool (routing, fusion, graph rescore,
  modality-calibration, facet-coverage) has no room, and expanding the pool with
  abstractions (RAPTOR, cross-doc) loses to specific chunks.
- Gains appear only upstream (HyDE = better query) or decision-layer (SSCC).
- **Query-side probe (honest):** modality-aware HyDE -- best point estimates on
  code/table questions but not separable from a prose ensemble at n=56
  (`results/hetdocqa_mahyde_control.txt`); reported as promising/inconclusive.
- **Implementation fragility:** cross-doc tier was inert by construction until the
  level bug was fixed (then a *valid* neutral); PPR was 40x too slow; RAPTOR sign-flips.

### 7. Limitations & Broader Impact
- Scale (762 Q); single-reranker dependence (would a weaker reranker change it?);
  generator choice; HetDocQA validation depth (IAA). What would overturn the claims.

### 8. Conclusion
- Stop bolting on re-ranking/expansion enhancements for heterogeneous RAG; invest
  query-side and decision-layer; evaluate on heterogeneous data with correction.

### Appendices
- Full per-benchmark tables; datasheet; capability matrix; reproduction; the seven
  faithfully-reimplemented components + the bugs found.

---

## Figures & tables (locked)

**Negatives placement:** core enhancement negatives (RAPTOR, cross-doc, graph, routing,
DPHF, CRAG) stay in the MAIN body -- they are the contribution. A dedicated appendix
holds the *exploratory* negatives (modality-calibration, facet-coverage, modality-HyDE
3-arm control) + exhaustive per-benchmark tables. Don't bury the core negatives.

**Figures (7 main body; each carries one distinct claim)**
- **F1 (HEADLINE):** principle diagram -- shared pipeline with each enhancement placed
  at its stage (query / pool-expansion / re-ranking / decision), color-coded by whether
  it survives correction. Leads with the *idea* so negatives read as explanatory.
- **F2:** HetDocQA construction + span->chunk labeling -- two-panel: build pipeline
  (sources -> collections -> generate -> filter -> validate -> splits) and the
  chunker-agnostic >=50%-overlap span->chunk mapping (the fairness mechanism).
- **F3:** forest plot of effect sizes (Δ-F1, Δ-nDCG ± 95% CI) on HetDocQA test, vertical
  zero line. Reranker off-scale; almost every other CI crosses zero; SSCC the lone
  exclusion. `results/hetdocqa_test_ablation.txt` (needs per-query deltas -> extract).
- **F4:** reranker dominance -- with/without-reranker collapse (nDCG ~0 without),
  contrasted with the tiny deltas of every other component.
- **F5:** heterogeneity gradient -- SSCC (and modality-HyDE) effect size across
  MuSiQue -> QASPER -> HetDocQA (homogeneous -> heterogeneous).
- **F6:** EGR entropy degeneracy -- histogram of per-query routing entropy (near-constant
  at the log K ceiling) = the signal routing relies on barely varies. Needs per-query
  entropies -> extend `scripts/diag_router.py` to dump them.
- **F7:** SSCC calibration mechanism -- per-source (bi- vs cross-encoder) relevance-score
  distributions + the fitted per-source thresholds, showing why one global threshold
  fails. Needs extraction from an SSCC calibration run.

Appendix figures: per-type/per-modality performance breakdown; implementation-fragility
(RAPTOR dev<->test sign-flip; PPR 6.4s->0.15s; cross-doc 0/30->retrieved post-fix);
modality-HyDE 3-arm control + prose-misses-code/table mechanism.

**Main-body tables**
- **T1:** HetDocQA composition (modality x type x split, span-label stats).
- **T2:** headline ablation -- HetDocQA test, full + each -component, nDCG@10/R@10/F1
  with corrected-significance markers.
- **T3 (PROMINENT, centerpiece):** cross-benchmark survival grid -- enhancement x
  {MuSiQue, QASPER, HetDocQA}, survives-Holm checkmarks. Almost all crosses; own it.

**Appendix tables:** full per-benchmark ablations (raw + corrected p, secondary
metrics); routing triad; exploratory negatives; datasheet; capability matrix.

---

## Honesty firewall for the write-up
- Report every negative prominently; never bury under correction.
- SSCC scope stated plainly (heterogeneity-specific, modest).
- Modality-HyDE = inconclusive, not a win.
- RAPTOR = unstable; do not claim.
- No metric without a real label; declare no-test-tuning.
