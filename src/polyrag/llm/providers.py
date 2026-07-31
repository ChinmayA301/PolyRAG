"""Model registry over OpenAI-compatible endpoints.

Every provider here speaks the OpenAI chat-completions protocol, so one client
(the official `openai` SDK) covers Groq, GitHub Models, OpenRouter, and a local
Ollama server. Model aliases map to concrete provider model IDs. Retired models
stay in the registry as non-selectable historical records with named
replacements, so old aliases remain explainable without sending doomed calls.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from polyrag.config import settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str  # groq | openrouter | ollama | mock
    model_id: str
    description: str
    archived: bool = False
    replacement: str | None = None
    ui_selectable: bool = True


REGISTRY: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in [
        ModelSpec("gpt-oss", "groq", "openai/gpt-oss-120b", "OpenAI gpt-oss 120B open-weight (Groq free tier)"),
        ModelSpec("gpt-oss-20b", "groq", "openai/gpt-oss-20b", "OpenAI gpt-oss 20B open-weight (Groq free tier)"),
        ModelSpec("qwen-3.6", "groq", "qwen/qwen3.6-27b", "Alibaba Qwen 3.6 27B (Groq preview)"),
        ModelSpec("deepseek", "github", "deepseek/deepseek-v3-0324", "DeepSeek V3 (GitHub Models free tier; needs GITHUB_TOKEN)"),
        ModelSpec("deepseek-r1", "github", "deepseek/deepseek-r1-0528", "DeepSeek R1 reasoning (GitHub Models free tier; needs GITHUB_TOKEN)"),
        ModelSpec("gpt-4.1-mini", "github", "openai/gpt-4.1-mini", "OpenAI GPT-4.1 mini (GitHub Models; needs GITHUB_TOKEN)"),
        ModelSpec("nemotron", "openrouter", "nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA Nemotron 3 Super 120B (OpenRouter free tier; needs OPENROUTER_API_KEY)"),
        ModelSpec("ollama", "ollama", "llama3.2", "Local model via Ollama (set OLLAMA_ENABLED=true to show in the web UI)"),
        ModelSpec("mock", "mock", "mock", "Deterministic offline provider for tests and demos without keys", ui_selectable=False),
        # Historical records: retained for old commands/bookmarks, never offered
        # by the UI, and never sent to a provider after retirement.
        ModelSpec(
            "llama", "groq", "llama-3.3-70b-versatile",
            "Meta LLaMA 3.3 70B (Groq; archived ahead of 2026-08-16 shutdown)",
            archived=True, replacement="gpt-oss",
        ),
        ModelSpec(
            "qwen", "groq", "qwen/qwen3-32b",
            "Alibaba Qwen 3 32B (Groq; retired 2026-07-17)",
            archived=True, replacement="qwen-3.6",
        ),
        ModelSpec(
            "scout", "groq", "meta-llama/llama-4-scout-17b-16e-instruct",
            "Meta LLaMA 4 Scout (Groq; retired 2026-07-17)",
            archived=True, replacement="qwen-3.6",
        ),
        ModelSpec(
            "hermes", "openrouter", "nousresearch/hermes-3-llama-3.1-405b:free",
            "Hermes 3 LLaMA 405B free route (OpenRouter; archived)",
            archived=True, replacement="nemotron",
        ),
    ]
}

# Reasoning models wrap chain-of-thought in <think> tags; strip it from the answer.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class Completion:
    model_alias: str
    model_id: str
    provider: str
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    reasoning_stripped: bool = field(default=False)


class ProviderError(RuntimeError):
    pass


def _client_for(spec: ModelSpec) -> OpenAI:
    if spec.provider == "groq":
        if not settings.groq_api_key:
            raise ProviderError("GROQ_API_KEY is not set (free key: https://console.groq.com)")
        return OpenAI(base_url=GROQ_BASE_URL, api_key=settings.groq_api_key,
                      timeout=settings.request_timeout)
    if spec.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set (free key: https://openrouter.ai/keys)")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.openrouter_api_key,
                      timeout=settings.request_timeout)
    if spec.provider == "github":
        if not settings.github_token:
            raise ProviderError(
                "GITHUB_TOKEN is not set (use a dedicated PAT with models:read)")
        return OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=settings.github_token,
                      timeout=settings.request_timeout)
    if spec.provider == "ollama":
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama",
                      timeout=settings.request_timeout)
    raise ProviderError(f"No client for provider {spec.provider!r}")


def _mock_completion(spec: ModelSpec, messages: list[dict], started: float) -> Completion:
    """Deterministic answer that cites every context block. Used by tests/CI."""
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    refs = re.findall(r"^\[(\d+)\]", user, flags=re.MULTILINE)
    citations = "".join(f"[{r}]" for r in refs) or "[no context provided]"
    text = f"MOCK ANSWER — grounded in context blocks {citations}."
    return Completion(spec.alias, spec.model_id, spec.provider, text,
                      latency_s=time.perf_counter() - started,
                      prompt_tokens=len(user.split()), completion_tokens=len(text.split()))


def complete(alias: str, messages: list[dict], *, max_tokens: int | None = None,
             temperature: float | None = None) -> Completion:
    """Run one chat completion against a registered model. Never raises for
    provider-side failures: errors come back on `Completion.error` so a
    multi-model comparison degrades per-model instead of dying whole."""
    spec = REGISTRY.get(alias)
    if spec is None:
        raise KeyError(f"Unknown model alias {alias!r}. Known: {', '.join(REGISTRY)}")

    started = time.perf_counter()
    if spec.archived:
        replacement = f" Use `{spec.replacement}` instead." if spec.replacement else ""
        return Completion(
            spec.alias,
            spec.model_id,
            spec.provider,
            "",
            latency_s=time.perf_counter() - started,
            error=f"Model alias `{spec.alias}` is archived and no longer queried.{replacement}",
        )
    if spec.provider == "mock":
        return _mock_completion(spec, messages, started)

    try:
        client = _client_for(spec)
        resp = client.chat.completions.create(
            model=spec.model_id,
            messages=messages,
            max_tokens=max_tokens or settings.max_tokens,
            temperature=settings.temperature if temperature is None else temperature,
        )
        raw = resp.choices[0].message.content or ""
        text = _THINK_RE.sub("", raw).strip()
        usage = getattr(resp, "usage", None)
        return Completion(
            spec.alias, spec.model_id, spec.provider, text,
            latency_s=time.perf_counter() - started,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_stripped=(text != raw.strip()),
        )
    except Exception as exc:  # provider/network errors surface per-model
        return Completion(spec.alias, spec.model_id, spec.provider, "",
                          latency_s=time.perf_counter() - started, error=str(exc))


def available_aliases() -> list[dict]:
    """Registry plus deployment readiness and web-UI selectability metadata."""
    out = []
    for spec in REGISTRY.values():
        ready = not spec.archived and (
            spec.provider == "mock"
            or (spec.provider == "groq" and bool(settings.groq_api_key))
            or (spec.provider == "openrouter" and bool(settings.openrouter_api_key))
            or (spec.provider == "github" and bool(settings.github_token))
            or (spec.provider == "ollama" and settings.ollama_enabled)
        )
        out.append({
            "alias": spec.alias,
            "provider": spec.provider,
            "model_id": spec.model_id,
            "description": spec.description,
            "ready": ready,
            "archived": spec.archived,
            "replacement": spec.replacement,
            "selectable": ready and spec.ui_selectable,
        })
    return out
