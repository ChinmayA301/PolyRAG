from polyrag.index.chunker import chunk_records, chunk_text


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraph {i} with some meaningful sentence content here." for i in range(40))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert chunks
    assert all(len(c) <= 300 for c in chunks)


def test_oversized_paragraph_is_split_with_overlap():
    text = "word " * 500  # one giant paragraph, no \n\n breaks
    chunks = chunk_text(text.strip(), max_chars=400, overlap=100)
    assert len(chunks) > 1
    # consecutive chunks share carried-over text
    assert chunks[0][-50:].strip() in chunks[0]


def test_short_fragments_dropped():
    assert chunk_text("tiny", max_chars=400, overlap=50) == []


def test_chunk_records_carry_provenance():
    records = [{"source_id": "s1", "title": "T", "url": "u", "page": 3,
                "extraction": "ocr", "text": "A sentence long enough to survive the fragment filter."}]
    chunks = chunk_records(records, max_chars=400, overlap=50)
    assert chunks[0].source_id == "s1"
    assert chunks[0].page == 3
    assert chunks[0].extraction == "ocr"
    assert chunks[0].chunk_id == "s1:p3:0"
