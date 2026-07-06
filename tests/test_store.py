from polyrag.index.store import VectorStore


def test_search_returns_relevant_source(store, embedder):
    hits = store.search("algorithmic discrimination and data privacy", embedder, k=2)
    assert hits
    assert hits[0].chunk.source_id == "doc-rights"


def test_roundtrip_save_load(tmp_path, store, embedder):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.manifest["num_chunks"] == store.manifest["num_chunks"]
    assert loaded.manifest["embedder"] == "hashing"
    hits = loaded.search("govern map measure manage risk", embedder, k=1)
    assert hits[0].chunk.source_id == "doc-risk"


def test_load_missing_index_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope")
