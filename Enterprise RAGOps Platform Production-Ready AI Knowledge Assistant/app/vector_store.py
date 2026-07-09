from __future__ import annotations

import json
import math
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.document_loader import LoadedDocument, TextChunk


EMBEDDING_MODEL_NAME = "local-hash-embedding"
EMBEDDING_MODEL_VERSION = "2026-07-09.v1"
EMBEDDING_DIMENSIONS = 384


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    filename: str
    content_type: str
    sha256: str
    document_version: int
    chunk_count: int
    created_at: str
    updated_at: str
    active: bool = True


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    token_count: int
    chunk_version: int
    embedding_model: str
    embedding_model_version: str
    score: float = 0.0


class LocalHashEmbeddingModel:
    name = EMBEDDING_MODEL_NAME
    version = EMBEDDING_MODEL_VERSION
    dimensions = EMBEDDING_DIMENSIONS

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for raw_token in text.lower().split():
            token = "".join(ch for ch in raw_token if ch.isalnum())
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimensions
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SQLiteVectorStore:
    def __init__(self, db_path: Path | str = "data/processed/ragops.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = LocalHashEmbeddingModel()
        self._init_db()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    document_version INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_documents_filename
                ON documents(filename);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    chunk_version INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "active" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(active)"
            )

    def upsert_document(self, document: LoadedDocument, chunks: list[TextChunk]) -> tuple[StoredDocument, list[StoredChunk]]:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT document_id, sha256, document_version, created_at
                FROM documents
                WHERE filename = ?
                ORDER BY document_version DESC
                LIMIT 1
                """,
                (document.filename,),
            ).fetchone()

            if existing and existing["sha256"] == document.sha256:
                stored_document = self._row_to_document(
                    connection.execute(
                        "SELECT * FROM documents WHERE document_id = ?",
                        (existing["document_id"],),
                    ).fetchone()
                )
                stored_chunks = self.list_chunks_for_document(connection, stored_document.document_id)
                return stored_document, stored_chunks

            document_id = str(uuid.uuid4())
            document_version = int(existing["document_version"]) + 1 if existing else 1
            created_at = existing["created_at"] if existing else now
            connection.execute("UPDATE documents SET active = 0 WHERE filename = ?", (document.filename,))

            connection.execute(
                """
                INSERT INTO documents (
                    document_id, filename, content_type, sha256,
                    document_version, chunk_count, created_at, updated_at, active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    document.filename,
                    document.content_type,
                    document.sha256,
                    document_version,
                    len(chunks),
                    created_at,
                    now,
                    1,
                ),
            )

            stored_chunks: list[StoredChunk] = []
            for chunk in chunks:
                chunk_id = f"{document_id}:chunk-{chunk.chunk_index}"
                embedding = self.embedding_model.embed(chunk.text)
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, chunk_index, text, token_count,
                        chunk_version, embedding, embedding_model,
                        embedding_model_version, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_count,
                        1,
                        json.dumps(embedding),
                        self.embedding_model.name,
                        self.embedding_model.version,
                        now,
                    ),
                )
                stored_chunks.append(
                    StoredChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=document.filename,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        chunk_version=1,
                        embedding_model=self.embedding_model.name,
                        embedding_model_version=self.embedding_model.version,
                    )
                )

            stored_document = StoredDocument(
                document_id=document_id,
                filename=document.filename,
                content_type=document.content_type,
                sha256=document.sha256,
                document_version=document_version,
                chunk_count=len(chunks),
                created_at=created_at,
                updated_at=now,
            )
            return stored_document, stored_chunks

    def list_documents(self) -> list[StoredDocument]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE active = 1 ORDER BY updated_at DESC, filename ASC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def list_all_documents(self) -> list[StoredDocument]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY filename ASC, document_version DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_latest_document_by_filename(self, filename: str) -> StoredDocument | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE filename = ?
                ORDER BY document_version DESC
                LIMIT 1
                """,
                (filename,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def delete_documents_not_in(self, filenames: set[str]) -> int:
        placeholders = ",".join("?" for _ in filenames)
        with self._connect() as connection:
            if filenames:
                rows = connection.execute(
                    f"SELECT document_id FROM documents WHERE filename NOT IN ({placeholders})",
                    tuple(filenames),
                ).fetchall()
            else:
                rows = connection.execute("SELECT document_id FROM documents").fetchall()

            document_ids = [row["document_id"] for row in rows]
            for document_id in document_ids:
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return len(document_ids)

    def delete_inactive_document_versions(self) -> int:
        with self._connect() as connection:
            rows = connection.execute("SELECT document_id FROM documents WHERE active = 0").fetchall()
            document_ids = [row["document_id"] for row in rows]
            for document_id in document_ids:
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return len(document_ids)

    def search(self, query: str, top_k: int = 4) -> list[StoredChunk]:
        query_embedding = self.embedding_model.embed(query)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, d.filename
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE d.active = 1
                """
            ).fetchall()

        scored_chunks: list[StoredChunk] = []
        for row in rows:
            embedding = json.loads(row["embedding"])
            score = cosine_similarity(query_embedding, embedding)
            scored_chunks.append(
                StoredChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    filename=row["filename"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    token_count=row["token_count"],
                    chunk_version=row["chunk_version"],
                    embedding_model=row["embedding_model"],
                    embedding_model_version=row["embedding_model_version"],
                    score=score,
                )
            )

        return sorted(scored_chunks, key=lambda chunk: chunk.score, reverse=True)[:top_k]

    def list_chunks_for_document(self, connection: sqlite3.Connection, document_id: str) -> list[StoredChunk]:
        rows = connection.execute(
            """
            SELECT c.*, d.filename
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.document_id = ?
            ORDER BY c.chunk_index ASC
            """,
            (document_id,),
        ).fetchall()
        return [
            StoredChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                filename=row["filename"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                token_count=row["token_count"],
                chunk_version=row["chunk_version"],
                embedding_model=row["embedding_model"],
                embedding_model_version=row["embedding_model_version"],
            )
            for row in rows
        ]

    def _row_to_document(self, row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            document_id=row["document_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            sha256=row["sha256"],
            document_version=row["document_version"],
            chunk_count=row["chunk_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            active=bool(row["active"]),
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions.")
    return sum(a * b for a, b in zip(left, right, strict=True))
