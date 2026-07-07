"""Deploy polyrag to a Hugging Face Space (Docker, free tier, no card required).

Usage:
    HF_TOKEN=... GROQ_API_KEY=... python deploy/hf_space.py [owner/space-name]

Requires a built index (`polyrag ingest && polyrag index`) — the Space image
bakes it in, same as the Cloud Run deployment. Idempotent: re-running updates
the Space in place.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """\
---
title: polyrag
emoji: 🏛️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 8080
pinned: false
license: mit
short_description: Multi-model RAG over real AI-governance documents
---

# polyrag — live demo

Side-by-side, citation-grounded answers from multiple LLMs over real
AI-governance documents (NIST AI RMF, EU AI Act, AI Bill of Rights, OECD).
Same question, same retrieved FAISS context — differences are the models'.

Source, evaluation methodology, and honest-claims table:
**https://github.com/ChinmayA301/PolyRAG**
"""

FILES = ["Dockerfile", "pyproject.toml", "data/sources.yaml", "data/eval.yaml"]
DIRS = ["src", "static", "data/index"]


def main() -> None:
    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN not set")
    groq_key = os.environ.get("GROQ_API_KEY") or sys.exit("GROQ_API_KEY not set")
    if not (ROOT / "data/index/index.faiss").exists():
        sys.exit("No index at data/index — run `polyrag ingest && polyrag index` first")

    api = HfApi(token=token)
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{api.whoami()['name']}/polyrag"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    # Secret is stored on the Space, never in its git history.
    api.add_space_secret(repo_id, "GROQ_API_KEY", groq_key)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in FILES:
            (stage / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, stage / f)
        for d in DIRS:
            shutil.copytree(ROOT / d, stage / d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=stage,
                          commit_message="Deploy polyrag")

    owner, name = repo_id.split("/")
    print(f"Pushed. Build status: https://huggingface.co/spaces/{repo_id}")
    print(f"App URL (once built): https://{owner.lower()}-{name.lower()}.hf.space")


if __name__ == "__main__":
    main()
