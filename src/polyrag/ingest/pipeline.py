"""Ingestion pipeline: sources.yaml -> data/raw (downloads) -> data/extracted (JSONL).

The ingest report is honest by design: it counts pages by extraction method and
lists anything that failed or needs OCR, so the corpus never silently pretends
to be more complete than it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from polyrag.ingest.pdf import extract_pdf
from polyrag.ingest.web import fetch_page


def load_sources(sources_file: Path) -> list[dict]:
    spec = yaml.safe_load(sources_file.read_text(encoding="utf-8"))
    return spec["sources"]


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return  # cached
    with httpx.stream("GET", url, follow_redirects=True, timeout=120,
                      headers={"User-Agent": "polyrag/0.1"}) as resp:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)


def ingest_source(source: dict, raw_dir: Path, extracted_dir: Path) -> dict:
    """Returns a per-source report; writes records to extracted/{id}.jsonl."""
    sid, title, url, kind = source["id"], source["title"], source["url"], source["type"]
    report = {"source_id": sid, "type": kind, "status": "ok", "pages": 0,
              "by_extraction": {}, "error": None}
    try:
        if kind == "pdf":
            dest = raw_dir / f"{sid}.pdf"
            _download(url, dest)
            records = extract_pdf(dest, sid, title, url)
        elif kind == "html":
            records = [fetch_page(url, sid, title, js=source.get("js", False))]
        else:
            raise ValueError(f"Unknown source type {kind!r}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        return report

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with open(extracted_dir / f"{sid}.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report["pages"] = len(records)
    for rec in records:
        key = rec["extraction"]
        report["by_extraction"][key] = report["by_extraction"].get(key, 0) + 1
    return report


def run_ingest(sources_file: Path, raw_dir: Path, extracted_dir: Path,
               only: list[str] | None = None) -> list[dict]:
    reports = []
    for source in load_sources(sources_file):
        if only and source["id"] not in only:
            continue
        reports.append(ingest_source(source, raw_dir, extracted_dir))
    return reports


def load_extracted(extracted_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(extracted_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("text", "").strip():
                records.append(rec)
    return records
