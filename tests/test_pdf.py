import fitz

from polyrag.ingest.pdf import extract_pdf


def _make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_text_layer_extraction(tmp_path):
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, ["Hello governance world, this is page one with a real text layer."])
    records = extract_pdf(pdf, "src", "Title", "https://example.org")
    assert len(records) == 1
    assert records[0]["extraction"] == "text-layer"
    assert "governance" in records[0]["text"]
    assert records[0]["page"] == 1


def test_blank_page_flagged_for_ocr(tmp_path):
    pdf = tmp_path / "b.pdf"
    _make_pdf(pdf, ["A normal page with enough text to pass the threshold easily.", ""])
    records = extract_pdf(pdf, "src", "Title", "https://example.org")
    assert records[0]["extraction"] == "text-layer"
    # page 2 has no text layer -> ocr if rapidocr installed, else disclosed as missing
    assert records[1]["extraction"] in ("ocr", "ocr-missing")
