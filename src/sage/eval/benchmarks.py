"""Loaders for existing multi-hop and long-document QA benchmarks.

Each loader returns a :class:`~sage.eval.dataset.RetrievalDataset` with a pooled
corpus and passage-level ``qrels`` derived from the benchmark's own gold evidence, so
the same retrieval + answer harness runs across datasets unchanged.

* **MuSiQue-Ans** -- hard multi-hop; gold = the ``is_supporting`` paragraphs (the
  distractors make it shortcut-resistant). Stresses routing, DPHF, and NLI chains.
* **QASPER** -- question answering over full scientific papers; gold = the evidence
  paragraphs. A long-document corpus, so it exercises RAPTOR and the cross-doc tier.
  Loaded from the official CC-BY-4.0 release (the HF script loader was removed in
  ``datasets`` 3.x).
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from sage.eval.dataset import QAExample, RetrievalDataset

__all__ = ["load_musique", "load_qasper"]

_QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"


def load_musique(
    *, split: str = "validation", max_queries: int | None = 200, hf_name: str = "dgslibisey/MuSiQue"
) -> RetrievalDataset:
    """Load a MuSiQue-Ans sample with per-question pooled paragraphs as the corpus."""
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=split)
    if max_queries is not None:
        ds = ds.select(range(min(max_queries, len(ds))))

    corpus: dict[str, str] = {}
    examples: list[QAExample] = []
    qrels: dict[str, dict[str, int]] = {}
    for row in ds:
        qid = str(row["id"])
        rel: dict[str, int] = {}
        for para in row["paragraphs"]:
            pid = f"{qid}::{para['idx']}"
            title = para.get("title") or ""
            corpus[pid] = f"{title}\n{para['paragraph_text']}".strip()
            if para.get("is_supporting"):
                rel[pid] = 1
        if not rel:  # only keep answerable questions with gold support
            continue
        answers = (row["answer"], *(row.get("answer_aliases") or ()))
        examples.append(
            QAExample(
                qid=qid,
                question=row["question"],
                answers=tuple(a for a in answers if a),
                metadata={"hops": str(len(row.get("question_decomposition") or []))},
            )
        )
        qrels[qid] = rel
    return RetrievalDataset(name="musique", examples=examples, corpus=corpus, qrels=qrels)


def _qasper_json(cache_dir: Path, split: str) -> dict[str, Any]:
    """Download (once) and return the parsed QASPER split JSON."""
    member = f"qasper-{'dev' if split in ('validation', 'dev') else 'train'}-v0.3.json"
    local = cache_dir / member
    if not local.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_QASPER_URL, timeout=120) as resp:
            blob = resp.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"{member} missing from QASPER archive")
            local.write_bytes(extracted.read())
    parsed: dict[str, Any] = json.loads(local.read_text())
    return parsed


def _qasper_answer(ans: dict[str, Any]) -> str | None:
    """Reduce one QASPER answer annotation to a single gold string (or None to skip)."""
    if ans.get("unanswerable"):
        return None
    if ans.get("free_form_answer"):
        return str(ans["free_form_answer"])
    if ans.get("extractive_spans"):
        return "; ".join(ans["extractive_spans"])
    yn = ans.get("yes_no")
    if yn is not None:
        return "Yes" if yn else "No"
    return None


def load_qasper(
    *,
    split: str = "validation",
    max_papers: int | None = 80,
    cache_dir: str | Path = ".cache/qasper",
) -> RetrievalDataset:
    """Load QASPER over a pooled corpus of full-paper paragraphs.

    Gold evidence strings are matched within the question's own paper, while the
    corpus pools paragraphs across all sampled papers (cross-paper distractors).
    """
    papers = _qasper_json(Path(cache_dir), split)
    items = list(papers.items())
    if max_papers is not None:
        items = items[:max_papers]

    corpus: dict[str, str] = {}
    examples: list[QAExample] = []
    qrels: dict[str, dict[str, int]] = {}

    for paper_id, paper in items:
        # Build this paper's paragraph table: text -> pooled corpus id.
        para_ids: dict[str, str] = {}
        sections = [{"section_name": "Abstract", "paragraphs": [paper.get("abstract", "")]}]
        sections += paper.get("full_text", [])
        for s_idx, section in enumerate(sections):
            for p_idx, raw in enumerate(section.get("paragraphs", [])):
                body = (raw or "").strip()
                if not body:
                    continue
                pid = f"{paper_id}::{s_idx}::{p_idx}"
                corpus[pid] = body
                para_ids.setdefault(body, pid)

        for qa in paper.get("qas", []):
            qid = str(qa["question_id"])
            rel: dict[str, int] = {}
            golds: list[str] = []
            for annot in qa.get("answers", []):
                ans = annot.get("answer", {})
                for ev in ans.get("evidence", []):
                    ev_pid = para_ids.get((ev or "").strip())
                    if ev_pid is not None:
                        rel[ev_pid] = 1
                gold = _qasper_answer(ans)
                if gold:
                    golds.append(gold)
            if not rel or not golds:  # need both a retrieval target and a scorable answer
                continue
            examples.append(
                QAExample(
                    qid=qid,
                    question=qa["question"],
                    answers=tuple(dict.fromkeys(golds)),
                    metadata={"paper_id": paper_id},
                )
            )
            qrels[qid] = rel
    return RetrievalDataset(name="qasper", examples=examples, corpus=corpus, qrels=qrels)
