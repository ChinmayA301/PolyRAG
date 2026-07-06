"""FAISS vector store with a JSONL sidecar for chunk metadata.

Vectors are L2-normalized, so IndexFlatIP gives cosine similarity. Exact
(non-approximate) search is the right call at this corpus size — no recall
loss, and rebuilds are cheap. A manifest records the embedder used to build
the index so query-time embedding can never silently mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from polyrag.index.chunker import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], manifest: dict) -> None:
        self.index = index
        self.chunks = chunks
        self.manifest = manifest

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "VectorStore":
        if not chunks:
            raise ValueError("No chunks to index — run ingestion first")
        vectors = embedder.encode([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        manifest = {
            "embedder": embedder.name,
            "model": getattr(embedder, "model_name", None),
            "dim": int(vectors.shape[1]),
            "num_chunks": len(chunks),
            "num_sources": len({c.source_id for c in chunks}),
        }
        return cls(index, chunks, manifest)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / INDEX_FILE))
        with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (index_dir / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_path = index_dir / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run `polyrag ingest` then `polyrag index`.")
        index = faiss.read_index(str(index_path))
        chunks = [Chunk.from_dict(json.loads(line))
                  for line in (index_dir / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
        return cls(index, chunks, manifest)

    def search(self, query: str, embedder, k: int = 6) -> list[SearchHit]:
        qvec = embedder.encode([query])
        scores, ids = self.index.search(np.asarray(qvec, dtype="float32"), k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hits.append(SearchHit(chunk=self.chunks[int(idx)], score=float(score)))
        return hits
