from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    content_type: str
    sha256: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    token_count: int


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_supported_file(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {allowed}.")


def load_document(path: Path, content_type: str = "application/octet-stream") -> LoadedDocument:
    validate_supported_file(path.name)
    content = path.read_bytes()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = extract_pdf_text(path)
    else:
        text = content.decode("utf-8", errors="replace")

    text = normalize_text(text)
    if not text:
        raise ValueError("Document did not contain extractable text.")

    return LoadedDocument(
        filename=path.name,
        content_type=content_type,
        sha256=sha256_bytes(content),
        text=text,
    )


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    page_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_text.append(f"[page {page_number}]\n{text}")
    return "\n\n".join(page_text)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            TextChunk(
                chunk_index=index,
                text=" ".join(chunk_words),
                token_count=len(chunk_words),
            )
        )
        if end == len(words):
            break
        start = end - overlap
        index += 1

    return chunks

