"""Model registry over OpenAI-compatible endpoints.

Every provider here speaks the OpenAI chat-completions protocol, so one client
(the official `openai` SDK) covers Groq, OpenRouter, and a local Ollama server.
Model aliases are what users type (`--model llama`); each maps to a concrete
provider + model id. The registry is data, not code, so swapping a deprecated
hosted model is a one-line change.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from polyrag.config import settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str  # groq | openrouter | ollama | mock
    model_id: str
    description: str


REGISTRY: dict[str, ModelSpec] = {
    spec.alias: spec
    for spec in [
        ModelSpec("llama", "groq", "llama-3.3-70b-versatile", "Meta LLaMA 3.3 70B (Groq free tier)"),
        ModelSpec("gpt-oss", "groq", "openai/gpt-oss-120b", "OpenAI gpt-oss 120B open-weight (Groq free tier)"),
        ModelSpec("qwen", "groq", "qwen/qwen3-32b", "Alibaba Qwen3 32B (Groq free tier)"),
        ModelSpec("scout", "groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Meta LLaMA 4 Scout (Groq free tier)"),
        ModelSpec("deepseek", "openrouter", "deepseek/deepseek-r1:free", "DeepSeek R1 (OpenRouter free tier; needs OPENROUTER_API_KEY)"),
        ModelSpec("ollama", "ollama", "llama3.2", "Local model via Ollama (no API key; edit model_id to taste)"),
        ModelSpec("mock", "mock", "mock", "Deterministic offline provider for tests and demos without keys"),
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
    """Registry plus a 'ready' flag: does this machine have what the alias needs?"""
    out = []
    for spec in REGISTRY.values():
        ready = (
            spec.provider == "mock"
            or (spec.provider == "groq" and bool(settings.groq_api_key))
            or (spec.provider == "openrouter" and bool(settings.openrouter_api_key))
            or spec.provider == "ollama"  # can't know without probing; assume maybe
        )
        out.append({"alias": spec.alias, "provider": spec.provider, "model_id": spec.model_id,
                    "description": spec.description, "ready": ready})
    return out
