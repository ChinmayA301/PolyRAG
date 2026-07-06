"""Embedding backends.

SentenceTransformerEmbedder is the real one: it auto-detects the best device
(CUDA GPU > Apple MPS > CPU), which is where the "runs accelerated when
hardware allows" claim lives. HashingEmbedder is a deterministic, dependency-free
fallback so tests and CI never download model weights.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Feature-hashing bag-of-words. Deterministic, no downloads, CPU-only.
    Good enough for exact-ish lexical matching in tests; not for production."""

    name = "hashing"
    model_name = None

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _l2_normalize(out)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False,
                                 normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


def get_embedder(kind: str, model_name: str, device: str = "auto", dim: int | None = None):
    """`dim` only matters for the hashing embedder — pass the index manifest's
    dim when loading, so query vectors always match the built index."""
    if kind == "hashing":
        return HashingEmbedder(dim=dim or 384)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name, device)
    if kind == "auto":
        try:
            return SentenceTransformerEmbedder(model_name, device)
        except ImportError:
            return HashingEmbedder()
    raise ValueError(f"Unknown embedder {kind!r}")
