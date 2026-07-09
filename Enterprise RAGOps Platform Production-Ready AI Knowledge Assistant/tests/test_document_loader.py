from app.document_loader import chunk_text, normalize_text


def test_normalize_text_removes_extra_whitespace() -> None:
    assert normalize_text("hello   world\n\n\nagain") == "hello world\n\nagain"


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert len(chunks) == 2
    assert chunks[0].token_count == 6
    assert chunks[1].text.startswith("token4 token5")

