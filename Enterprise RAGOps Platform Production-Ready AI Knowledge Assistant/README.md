# Enterprise RAGOps Platform

Production-style AI knowledge assistant for learning RAG, LLMOps, MLOps, backend engineering, Docker, CI/CD, monitoring, and DigitalOcean deployment.

This repository is being built in phases. Phase 1 gives you the ingestion foundation: FastAPI, document upload, PDF/TXT/Markdown extraction, text chunking, local embeddings, SQLite metadata, and local vector search.

## Architecture

```text
User
  -> Streamlit frontend
  -> FastAPI backend
  -> Document loader
  -> Text chunker
  -> Local embedding model
  -> SQLite metadata + local vector storage
  -> Retrieval API
```

Later phases will add LLM answer generation, source-cited responses, tracing, evaluation gates, Prometheus/Grafana, Docker Compose, GitHub Actions, and DigitalOcean deployment.

## Tech Stack

- FastAPI for the backend API
- Pydantic for request and response validation
- SQLite for Phase 1 metadata and local vector storage
- `pypdf` for PDF text extraction
- Streamlit for the first frontend
- Pytest for tests

## Phase 1 Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open API docs:

```text
http://localhost:8000/docs
```

Start the Streamlit frontend in a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

## Phase 1 API

### `GET /health`

Checks whether the backend is running.

### `POST /documents/upload`

Uploads a PDF, TXT, Markdown, or `.markdown` file. The backend extracts text, chunks it, embeds each chunk, stores document metadata, and stores chunk vectors locally.

### `GET /documents`

Lists uploaded documents and version metadata.

### `POST /ask`

Phase 1 retrieval-only endpoint. It returns the top matching chunks as citations. LLM answer generation starts in Phase 2.

## Why Phase 1 Matters

RAG quality starts before the LLM. If parsing, chunking, metadata, or embeddings are weak, the assistant retrieves poor context and the generated answer becomes unreliable. Production AI teams treat ingestion as a versioned data pipeline, not as a one-off upload script.

In this phase, every document gets:

- `sha256` content hash
- document version
- chunk version
- embedding model name
- embedding model version

That metadata becomes essential later when debugging bad answers, rolling back bad indexes, comparing embedding models, and proving which sources influenced an answer.

## Common Mistakes

- Uploading scanned PDFs with no OCR layer. `pypdf` can only extract embedded text.
- Chunking by characters without thinking about retrieval quality.
- Forgetting to store embedding model versions.
- Re-ingesting changed documents without tracking document versions.
- Jumping into LLM prompts before validating retrieval.

## Verification Checklist

- `pytest -q` passes.
- `GET /health` returns `{"status":"ok", ...}`.
- Uploading a `.txt`, `.md`, or text-based `.pdf` returns chunks.
- `GET /documents` shows the uploaded document.
- `POST /ask` returns relevant citation chunks.
- `data/processed/ragops.db` is created after ingestion.

## Screenshots

Add screenshots here as the frontend evolves:

- `docs/screenshots/upload.png`
- `docs/screenshots/answer-with-citations.png`
- `docs/screenshots/monitoring-dashboard.png`

## Future Improvements

- Phase 2: LLM-backed answer generation with citations
- Phase 3: Streamlit feedback and trace viewing
- Phase 4: tracing, JSON logs, and optional Langfuse/OpenTelemetry
- Phase 5: RAG evaluation pipeline and deployment quality gate
- Phase 6: prompt/model version comparison
- Phase 7: Prometheus metrics and Grafana dashboard
- Phase 8: RAGOps automation and rollback strategy
- Phase 9+: Docker, GitHub Actions, and DigitalOcean deployment

