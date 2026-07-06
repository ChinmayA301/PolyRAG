"""RAG core: retrieve -> prompt with numbered context -> answer with citations.

Answers must cite context blocks as [n]; `citation_coverage` measures the share
of retrieved blocks the answer actually used, and `answer_citations` lets the
UI link each citation back to its source document and page. `compare` fans the
same retrieved context out to several models so differences in the answers are
attributable to the model, not to retrieval variance.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from polyrag.config import settings
from polyrag.index.store import SearchHit, VectorStore
from polyrag.llm.providers import Completion, complete

SYSTEM_PROMPT = """\
You answer questions about AI-governance documents using ONLY the provided context.
Rules:
- Cite every claim with the bracketed number of its context block, e.g. [2].
- If the context does not contain the answer, say so plainly. Never invent content.
- Be concise and precise; quote key phrases where wording matters (definitions, obligations).
"""


@dataclass
class RagResult:
    question: str
    completion: Completion
    hits: list[SearchHit]
    citation_coverage: float  # fraction of retrieved blocks cited in the answer
    cited_blocks: list[int]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.completion.text,
            "model": asdict(self.completion),
            "citation_coverage": self.citation_coverage,
            "cited_blocks": self.cited_blocks,
            "sources": [
                {
                    "n": i + 1,
                    "score": round(h.score, 4),
                    "source_id": h.chunk.source_id,
                    "title": h.chunk.title,
                    "url": h.chunk.url,
                    "page": h.chunk.page,
                    "extraction": h.chunk.extraction,
                    "preview": h.chunk.text[:240],
                }
                for i, h in enumerate(self.hits)
            ],
        }


def build_prompt(question: str, hits: list[SearchHit]) -> list[dict]:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        loc = f", page {hit.chunk.page}" if hit.chunk.page is not None else ""
        blocks.append(f"[{i}] ({hit.chunk.title}{loc})\n{hit.chunk.text}")
    context = "\n\n".join(blocks)
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _extract_citations(text: str, k: int) -> list[int]:
    """Models cite as [2], [1, 3], [1][4], or gpt-oss's native 【5†L1-L7】 —
    accept all bracket-number forms."""
    cited: set[int] = set()
    for group in re.findall(r"\[([\d,\s]+)\]", text):
        cited.update(int(m) for m in re.findall(r"\d{1,2}", group))
    cited.update(int(m) for m in re.findall(r"【(\d{1,2})†", text))
    return sorted(n for n in cited if 1 <= n <= k)


def ask(question: str, model: str, store: VectorStore, embedder,
        k: int | None = None) -> RagResult:
    hits = store.search(question, embedder, k=k or settings.top_k)
    messages = build_prompt(question, hits)
    completion = complete(model, messages)
    cited = _extract_citations(completion.text, len(hits)) if not completion.error else []
    coverage = len(cited) / len(hits) if hits else 0.0
    return RagResult(question, completion, hits, coverage, cited)


def compare(question: str, models: list[str], store: VectorStore, embedder,
            k: int | None = None) -> list[RagResult]:
    """Same question, same retrieved context, N models in parallel."""
    hits = store.search(question, embedder, k=k or settings.top_k)
    messages = build_prompt(question, hits)

    def run(model: str) -> RagResult:
        completion = complete(model, messages)
        cited = _extract_citations(completion.text, len(hits)) if not completion.error else []
        coverage = len(cited) / len(hits) if hits else 0.0
        return RagResult(question, completion, hits, coverage, cited)

    with ThreadPoolExecutor(max_workers=max(len(models), 1)) as pool:
        return list(pool.map(run, models))
