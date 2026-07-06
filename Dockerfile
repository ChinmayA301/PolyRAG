# Slim CPU image for Cloud Run / any container host.
# The FAISS index is baked in at build time (data/index must exist — run
# `polyrag ingest && polyrag index` before `docker build`), so the container
# is stateless and cold-starts fast. GPU (CUDA) is auto-detected at runtime
# when deployed on GPU hosts; on Cloud Run CPU it falls back cleanly.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_DEVICE=cpu

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[embed]"

# Warm the embedding model into the image so Cloud Run cold starts don't
# download ~90MB from HuggingFace on first request.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY static ./static
COPY data/index ./data/index
COPY data/sources.yaml data/eval.yaml ./data/

EXPOSE 8080
CMD ["uvicorn", "polyrag.api:app", "--host", "0.0.0.0", "--port", "8080"]
