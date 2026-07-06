"""PDF extraction: PyMuPDF text layer first, OCR fallback for scanned pages.

Each page becomes one record tagged with how its text was obtained
(`text-layer` or `ocr`). Pages that need OCR when rapidocr isn't installed are
recorded with `extraction: "ocr-missing"` and empty text — the gap is disclosed
in the ingest report rather than silently dropped.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

# Below this many characters we assume the text layer is absent/garbage and
# the page is likely scanned.
MIN_TEXT_CHARS = 40

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR  # optional extra [ocr]

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_page(page: fitz.Page) -> str:
    pix = page.get_pixmap(dpi=200)
    import numpy as np

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    result, _ = _get_ocr()(img)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def extract_pdf(path: Path, source_id: str, title: str, url: str) -> list[dict]:
    records: list[dict] = []
    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            extraction = "text-layer"
            if len(text) < MIN_TEXT_CHARS:
                try:
                    text = _ocr_page(page).strip()
                    extraction = "ocr"
                except ImportError:
                    extraction = "ocr-missing"
                    text = ""
            records.append({
                "source_id": source_id,
                "title": title,
                "url": url,
                "page": page_num,
                "extraction": extraction,
                "text": text,
            })
    return records
