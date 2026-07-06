from polyrag import rag
from polyrag.llm.providers import REGISTRY, complete


def test_mock_provider_cites_context_blocks():
    messages = rag.build_prompt("q?", [])
    completion = complete("mock", [{"role": "user", "content": "[1] a\n[2] b\nQuestion: q?"}])
    assert completion.error is None
    assert "[1]" in completion.text and "[2]" in completion.text
    assert messages[0]["role"] == "system"


def test_ask_with_mock_returns_full_citation_coverage(store, embedder):
    result = rag.ask("What are the four risk functions?", "mock", store, embedder, k=2)
    assert result.completion.error is None
    assert result.citation_coverage == 1.0  # mock cites every block
    assert len(result.hits) == 2
    assert result.to_dict()["sources"][0]["title"]


def test_compare_runs_all_models_and_shares_context(store, embedder):
    results = rag.compare("privacy protections", ["mock", "mock"], store, embedder, k=2)
    assert len(results) == 2
    ids_a = [h.chunk.chunk_id for h in results[0].hits]
    ids_b = [h.chunk.chunk_id for h in results[1].hits]
    assert ids_a == ids_b  # same retrieved context for every model


def test_unknown_alias_raises(store, embedder):
    import pytest

    with pytest.raises(KeyError):
        rag.ask("q?", "not-a-model", store, embedder)


def test_missing_key_surfaces_as_error_not_exception(store, embedder, monkeypatch):
    from polyrag.config import settings

    monkeypatch.setattr(settings, "openrouter_api_key", "")
    assert "deepseek" in REGISTRY
    result = rag.ask("q?", "deepseek", store, embedder, k=1)
    assert result.completion.error  # degraded, not crashed
    assert result.citation_coverage == 0.0


def test_citation_extraction_handles_grouped_brackets():
    from polyrag.rag import _extract_citations

    assert _extract_citations("Claims [1, 3] and [2][4], but not [12].", k=6) == [1, 2, 3, 4]
    assert _extract_citations("No citations here.", k=6) == []


def test_citation_extraction_handles_gpt_oss_style():
    from polyrag.rag import _extract_citations

    assert _extract_citations("Registered in the EU database【5†L1-L7】 and 【4†L4-L9】.", k=6) == [4, 5]
