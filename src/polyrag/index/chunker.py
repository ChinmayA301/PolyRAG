"""Paragraph-aware chunking with overlap. Chunks carry full provenance
(source id, title, url, page) so answers can cite exactly where text came from."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    url: str
    page: int | None
    extraction: str  # text-layer | ocr | crawl4ai | httpx
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into ~max_chars chunks; hard-split oversized paragraphs
    with `overlap` chars of carry-over so no boundary loses context."""
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            start = 0
            while start < len(para):
                chunks.append(para[start:start + max_chars].strip())
                start += max_chars - overlap
            continue
        if len(buf) + len(para) + 2 > max_chars:
            flush()
        buf = f"{buf}\n\n{para}" if buf else para
    flush()
    return [c for c in chunks if len(c) > 40]  # drop fragments too short to retrieve


def chunk_records(records: list[dict], *, max_chars: int, overlap: int) -> list[Chunk]:
    """records: output of the ingestion pipeline (one per page / web article)."""
    chunks: list[Chunk] = []
    for rec in records:
        for i, piece in enumerate(chunk_text(rec["text"], max_chars=max_chars, overlap=overlap)):
            page = rec.get("page")
            loc = f"p{page}" if page is not None else "web"
            chunks.append(Chunk(
                chunk_id=f"{rec['source_id']}:{loc}:{i}",
                source_id=rec["source_id"],
                title=rec["title"],
                url=rec["url"],
                page=page,
                extraction=rec.get("extraction", "text-layer"),
                text=piece,
            ))
    return chunks
