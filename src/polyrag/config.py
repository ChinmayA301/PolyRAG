"""Central settings. Everything is overridable via environment variables or .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Repo root in an editable/src checkout; the working directory when the
    package is pip-installed (e.g. in the Docker image, where WORKDIR /app
    holds data/ and static/)."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "data").is_dir() else Path.cwd()


PROJECT_ROOT = _project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider credentials. Use a dedicated, minimally scoped models:read PAT
    # for GitHub Models; do not deploy a general-purpose gh CLI token.
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    github_token: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_enabled: bool = False

    # Data layout
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "data" / "index"

    # Embeddings. "auto" resolves to sentence-transformers when installed,
    # otherwise the deterministic hashing embedder (tests/CI).
    embedder: str = "auto"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = "auto"  # auto -> cuda > mps > cpu

    # Chunking
    chunk_chars: int = 1400
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 4

    # Keep the public demo concise and inside free-tier token budgets.
    max_tokens: int = 600
    temperature: float = 0.2
    request_timeout: float = 60.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def sources_file(self) -> Path:
        return self.data_dir / "sources.yaml"


settings = Settings()
