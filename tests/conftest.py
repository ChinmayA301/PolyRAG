import pytest

from polyrag.index.chunker import chunk_records
from polyrag.index.embedder import HashingEmbedder
from polyrag.index.store import VectorStore

# Small fixture corpus with distinctive vocabulary per source so the hashing
# embedder retrieves deterministically.
RECORDS = [
    {
        "source_id": "doc-risk",
        "title": "Risk Framework",
        "url": "https://example.org/risk",
        "page": 1,
        "extraction": "text-layer",
        "text": ("The risk management framework defines four functions: govern, map, "
                 "measure, and manage. Trustworthy systems require documented risk "
                 "tolerance and continuous measurement of impacts across the lifecycle."),
    },
    {
        "source_id": "doc-rights",
        "title": "Rights Blueprint",
        "url": "https://example.org/rights",
        "page": 2,
        "extraction": "ocr",
        "text": ("Citizens deserve protection from algorithmic discrimination. Automated "
                 "systems should provide notice, explanation, and human alternatives. "
                 "Data privacy safeguards must be built in by default for everyone."),
    },
]


@pytest.fixture()
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture()
def store(embedder):
    chunks = chunk_records(RECORDS, max_chars=400, overlap=50)
    return VectorStore.build(chunks, embedder)
