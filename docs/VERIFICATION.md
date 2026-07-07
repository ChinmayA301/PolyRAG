# Verification log

Real numbers from a real run — reproduce with the commands shown. Environment:
Apple Silicon Mac (embeddings on MPS), Python 3.12, Groq free tier, 2026-07-06.

## Corpus (all real, public documents)

`polyrag ingest` — 6/6 sources succeeded:

| source | type | pages | extraction |
|---|---|---|---|
| nist-ai-rmf | pdf | 48 | text-layer |
| nist-genai-profile | pdf | 64 | text-layer |
| eu-ai-act | pdf | 144 | text-layer |
| ai-bill-of-rights | pdf | 73 | text-layer |
| oecd-ai-principles | html | 1 | httpx+trafilatura (Crawl4AI fetch fell back; disclosed per record) |
| ec-ai-act-overview | html | 1 | crawl4ai |

No page in this corpus needed OCR (all PDFs ship text layers); the OCR fallback
is exercised by tests on synthetic blank-page PDFs instead. That is the honest
status of the OCR claim.

## Index

`polyrag index` — 1,097 chunks, 6 sources, MiniLM (384-dim), embedded on **mps**
(device auto-detected; CUDA path untested on NVIDIA hardware).

## Retrieval eval

`polyrag eval` — hit@6 = **89% (8/9)** on the hand-labeled set in `data/eval.yaml`.
The miss: "How does the European Commission categorize AI systems by risk level?"
retrieved the EU AI Act itself instead of the EC overview page — a reasonable
source for the answer, but not the labeled one. Kept as a miss.

## Live multi-model generation

`polyrag eval -m llama -m gpt-oss -m qwen` — 27 live Groq calls, 0 errors:

| model | answered | errors | mean citation coverage | mean latency (s) |
|---|---|---|---|---|
| llama (llama-3.3-70b-versatile) | 9/9 | 0 | 0.35 | 5.6 |
| gpt-oss (openai/gpt-oss-120b) | 9/9 | 0 | 0.30 | 13.3 |
| qwen (qwen/qwen3-32b) | 9/9 | 0 | 0.33 | 18.2 |

Latency includes free-tier queueing; single interactive queries ran 0.6–2.2s.
Citation coverage is the share of the 6 retrieved blocks each answer cited —
~0.3 means answers typically ground in 2 of 6 blocks, which is expected, not a
defect. It measures grounding discipline, not factual correctness.

Notable grounding behavior observed during verification: asked for the five
AI Bill of Rights principles with k=4, the retrieved chunks referenced but did
not enumerate the principles — both LLaMA and GPT-OSS answered "the context
does not list them" rather than fabricating the list.

## Tests

`pytest` — 20 passed (mock provider + hashing embedder; no keys or model
downloads needed, same configuration CI runs).

## DeepSeek verification (added 2026-07-06, later same day)

Free-tier churn, documented honestly: Groq deprecated its DeepSeek distill, and
OpenRouter retired `deepseek/deepseek-r1:free` (confirmed live: 404 "unavailable
for free"). The registry made the swap a one-line change: DeepSeek now runs on
the **GitHub Models free tier** (`models.github.ai`, OpenAI-compatible, any PAT
with `models:read`). Verified live on the prohibited-practices question,
same shared context as LLaMA:

| model | latency | behavior |
|---|---|---|
| deepseek/deepseek-v3-0324 | 5.7s | cited [1][2], flagged that the context doesn't enumerate the practices |
| deepseek/deepseek-r1-0528 | 5.4s | `<think>` trace stripped cleanly; most explicit refusal to overclaim of any model tested |

## Not verified
- OpenRouter (`hermes` alias): implemented; the key on this machine returned
  401 at verification time, so not verified live.
- Local Ollama provider: implemented, not run (Ollama not installed here).
- CUDA embedding: code path is standard torch/sentence-transformers; verified
  on MPS only.
- Cloud Run deployment: scripts complete and reviewed; not executed. Cloud Run
  requires a billing-linked account (a card, even inside the free tier), and this
  project runs card-free — the live Docker deployment is the Hugging Face Space
  below instead.

## Live deployment (added 2026-07-06)

`deploy/hf_space.py` → https://chinmaya301-polyrag.hf.space — the repo's
Dockerfile running on Hugging Face Spaces' free Docker tier (no card), FAISS
index and embedding model baked into the image, GROQ_API_KEY as a Space secret.
Verified after build: `/healthz` reports the baked index; a live `/compare`
returned grounded answers. The hosted demo serves the Groq models only —
the GitHub Models token is deliberately not deployed (a `gh` CLI token has repo
access; a scoped models:read PAT could be added later).
