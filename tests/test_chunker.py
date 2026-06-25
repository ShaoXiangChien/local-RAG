from rag_demo.ingestion.chunker import ChunkingConfig, RecursiveChunker


def test_chunker_preserves_markdown_headings_and_text_order() -> None:
    text = "# Intro\n\n" + "alpha beta gamma. " * 80 + "\n\n## Details\n\n" + "delta epsilon. " * 80
    chunker = RecursiveChunker(ChunkingConfig(chunk_token_size=45, chunk_token_overlap=8))

    chunks = chunker.split(text, doc_id="doc123", filename="notes.md")

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "doc123:0"
    assert chunks[0].filename == "notes.md"
    assert chunks[0].text.startswith("# Intro")
    assert chunks[0].start_char < chunks[0].end_char
    assert all(chunk.token_count_estimate <= 55 for chunk in chunks)
    assert "alpha" in " ".join(chunk.text for chunk in chunks)
    assert "delta" in " ".join(chunk.text for chunk in chunks)


def test_chunker_applies_overlap_between_adjacent_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(90))
    chunker = RecursiveChunker(ChunkingConfig(chunk_token_size=30, chunk_token_overlap=5))

    chunks = chunker.split(text, doc_id="doc123", filename="notes.txt")

    assert len(chunks) >= 3
    first_tail = chunks[0].text.split()[-5:]
    second_head = chunks[1].text.split()[:5]
    assert first_tail == second_head
