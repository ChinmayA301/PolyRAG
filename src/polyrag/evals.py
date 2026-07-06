"""Descriptive evaluation harness — reports, never overclaims.

Two layers:
1. Retrieval: hit@k — does a chunk from the expected source document appear in
   the top-k results for each labeled question? (No LLM involved.)
2. Generation: per-model answer rate, citation coverage, and latency on the
   same questions. Citation coverage measures grounding discipline, not
   factual correctness — the README is explicit about that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from polyrag import rag
from polyrag.index.store import VectorStore


@dataclass
class EvalReport:
    k: int
    retrieval_hits: int = 0
    retrieval_total: int = 0
    per_question: list[dict] = field(default_factory=list)
    per_model: dict = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.retrieval_hits / self.retrieval_total if self.retrieval_total else 0.0


def load_eval_set(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def evaluate(eval_set: list[dict], store: VectorStore, embedder, k: int,
             models: list[str] | None = None) -> EvalReport:
    report = EvalReport(k=k)
    for item in eval_set:
        hits = store.search(item["question"], embedder, k=k)
        retrieved_sources = [h.chunk.source_id for h in hits]
        hit = item["expected_source"] in retrieved_sources
        report.retrieval_hits += int(hit)
        report.retrieval_total += 1
        report.per_question.append({
            "question": item["question"],
            "expected_source": item["expected_source"],
            "hit": hit,
            "retrieved": retrieved_sources,
        })

    for model in models or []:
        stats = {"answered": 0, "errors": 0, "citation_coverage": [], "latency_s": []}
        for item in eval_set:
            result = rag.ask(item["question"], model, store, embedder, k=k)
            if result.completion.error:
                stats["errors"] += 1
                continue
            stats["answered"] += 1
            stats["citation_coverage"].append(result.citation_coverage)
            stats["latency_s"].append(result.completion.latency_s)
        n = max(stats["answered"], 1)
        report.per_model[model] = {
            "answered": stats["answered"],
            "errors": stats["errors"],
            "mean_citation_coverage": round(sum(stats["citation_coverage"]) / n, 3),
            "mean_latency_s": round(sum(stats["latency_s"]) / n, 2),
        }
    return report
