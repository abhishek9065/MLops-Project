# Enterprise RAGOps Platform

Production-style AI knowledge assistant for learning RAG, LLMOps, MLOps, backend engineering, Docker, CI/CD, monitoring, and DigitalOcean deployment.

This repository is being built in phases. It now covers ingestion, retrieval, RAG answer generation, citations, user feedback, trace persistence, prompt/model versioning, RAG evaluation gates, Prometheus metrics, Grafana, RAGOps automation, Docker Compose, GitHub Actions, and DigitalOcean deployment guidance.

## Architecture

```text
User
  -> Streamlit frontend
  -> FastAPI backend
  -> Document loader
  -> Text chunker
  -> Local embedding model
  -> SQLite metadata + local vector storage
  -> Prompt renderer
  -> LLM client
  -> Trace store
  -> Feedback loop
  -> Evaluation gate
  -> Prometheus metrics
  -> Grafana dashboard
  -> Docker Compose deployment
  -> GitHub Actions CI/CD
  -> DigitalOcean Droplet
```

The current implementation defaults to SQLite and a local vector store for low-cost learning. Docker Compose also starts Postgres and Qdrant so the repository is ready for a later storage migration without changing the deployment topology.

## Tech Stack

- FastAPI for the backend API
- Pydantic for request and response validation
- SQLite for Phase 1 metadata and local vector storage
- `pypdf` for PDF text extraction
- Streamlit for the first frontend
- Pytest for tests
- JSON structured logs for Phase 4 tracing
- Prometheus metrics for Phase 7 monitoring
- Grafana dashboard JSON starter config
- Docker Compose for API, frontend, Postgres, Qdrant, Prometheus, Grafana, and optional MLflow
- Optional OpenAI, Gemini, Langfuse, and OpenTelemetry integration points

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
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

## Phase 1 API

### `GET /health`

Checks whether the backend is running.

### `POST /documents/upload`

Uploads a PDF, TXT, Markdown, or `.markdown` file. The backend extracts text, chunks it, embeds each chunk, stores document metadata, and stores chunk vectors locally.

### `GET /documents`

Lists uploaded documents and version metadata.

### `POST /ask`

Runs retrieval, builds a versioned prompt, calls the configured LLM provider, returns an answer with source citations, and stores a trace.

Request:

```json
{
  "question": "What does the incident runbook require?",
  "top_k": 4,
  "prompt_version": "v1"
}
```

### `GET /traces/{trace_id}`

Loads the saved LLMOps trace for one question. This includes the question, retrieved chunks, prompt version, rendered prompt, model name, token usage, estimated cost, latency, answer, citations, feedback, and errors.

### `POST /traces/{trace_id}/feedback`

Stores thumbs-up/thumbs-down feedback for a generated answer.

Request:

```json
{
  "feedback": "up",
  "comment": "Answer was grounded in the handbook."
}
```

### `GET /metrics`

Returns Prometheus metrics for API requests, latency, retrieval, LLM calls, token usage, estimated cost, errors, and feedback.

## Phase 2: RAG Answer Generation

We now build a real RAG flow:

```text
question -> retrieve chunks -> render prompt -> call LLM -> answer with citations -> save trace
```

The default provider is local and free:

```env
LLM_PROVIDER=local
LLM_MODEL=local-extractive-rag-v1
```

The local model is intentionally simple. It forms an extractive answer from retrieved chunks so you can test the RAG system without paying for API calls. Hosted models can be enabled later:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
```

or:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=your_key_here
```

Why this matters: production RAG systems should let you test retrieval, prompts, traces, and feedback without depending on a paid provider for every local run.

## Phase 3: Frontend Feedback and Trace Viewing

The Streamlit app now has three tabs:

- `Upload`: upload PDF, TXT, or Markdown files.
- `Ask`: ask questions, view answer, citations, latency, model, cost, and feedback buttons.
- `Trace`: inspect the exact prompt, answer, citations, model, latency, cost, and feedback for a `trace_id`.

Why this matters: thumbs-up/down feedback is the first practical signal for answer quality. In production, this feedback becomes the seed for evaluation datasets, prompt comparisons, fine-tuning data, and regression tests.

## Phase 4: LLMOps Tracing and Structured Logs

Each `/ask` call stores:

- question
- retrieved chunks
- rendered prompt
- prompt version
- LLM model name
- embedding model name
- answer
- token usage
- estimated cost
- retrieval latency
- LLM latency
- total latency
- user feedback
- errors

Traces are stored in the SQLite `traces` table inside:

```text
data/processed/ragops.db
```

Logs are JSON-formatted so they are easy to ship later to systems such as Grafana Loki, Datadog, OpenSearch, or DigitalOcean logging.

Optional observability environment variables are already reserved:

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
OTEL_EXPORTER_OTLP_ENDPOINT=
```

The current implementation keeps those integrations optional so the project remains lightweight. Later phases can add full SDK-based remote trace export.

## Phase 5: RAG Evaluation Pipeline

The evaluation dataset lives at:

```text
data/eval/evaluation_dataset.jsonl
```

Each row contains:

- `question`
- `expected_answer`
- `expected_source_document`
- `expected_source_chunk`

The source documents for evaluation live in:

```text
data/eval/source_docs/
```

Run the evaluation gate:

```bash
python -m scripts.run_evaluation --threshold 0.65
```

or:

```bash
make evaluate
```

The script measures:

- `answer_relevance`: overlap between generated answer and expected answer.
- `faithfulness`: how much of the answer is supported by retrieved context.
- `context_precision`: how many retrieved chunks come from the expected source.
- `context_recall`: whether the retrieved context contains expected-answer facts.
- `citation_correctness`: whether the expected source document and chunk were cited.
- `hallucination_risk`: inverse of faithfulness.
- `overall_score`: weighted deployment gate score.

The CI workflow runs:

```bash
python -m scripts.run_evaluation --threshold 0.65
```

If the score drops below the threshold, CI fails. In a production AI system, this prevents a bad prompt, model, chunking strategy, or vector index update from being deployed silently.

## Phase 6: Prompt and Model Versioning

Prompt templates are stored separately:

```text
prompts/rag_prompt_v1.txt
prompts/rag_prompt_v2.txt
```

Every trace stores:

- prompt version
- rendered prompt
- LLM model name
- embedding model name
- embedding model version on chunks
- token usage
- cost
- latency

Compare prompt versions:

```bash
python -m scripts.run_evaluation --compare-prompts v1 v2 --threshold 0.65
```

or:

```bash
make compare-prompts
```

The report is written to:

```text
data/eval/evaluation_report.json
```

Why this matters: prompt changes are production changes. If `v2` improves answer relevance but increases latency or breaks citation correctness, you need evidence before rollout.

## Phase 7: Monitoring

Prometheus metrics are exposed at:

```text
http://localhost:8000/metrics
```

Implemented metrics:

- `ragops_requests_total`: traffic volume by endpoint and status.
- `ragops_request_latency_seconds`: API latency by endpoint.
- `ragops_errors_total`: server-side error count.
- `ragops_retrieval_latency_seconds`: vector search latency.
- `ragops_llm_latency_seconds`: LLM latency by model and prompt version.
- `ragops_token_usage_total`: prompt and completion token usage.
- `ragops_estimated_cost_usd_total`: estimated provider cost.
- `ragops_feedback_score_total`: thumbs-up/down feedback volume.
- `ragops_last_feedback_score`: latest feedback signal.

In real AI products, these metrics answer operational questions:

- Is the assistant getting slower?
- Did a prompt update increase token usage?
- Are users downvoting answers after a model change?
- Are retrieval calls slow because the index is unhealthy?
- Are LLM costs rising faster than user traffic?

Prometheus config:

```text
monitoring/prometheus.yml
```

Grafana dashboard starter:

```text
monitoring/grafana/dashboards/ragops_dashboard.json
```

## Phase 8: RAGOps Automation

RAGOps is the operational discipline around keeping retrieval systems healthy as documents, prompts, models, and indexes change.

Commands:

```bash
python -m scripts.reindex_changed_docs
python -m scripts.reindex_changed_docs --delete-stale
python -m scripts.delete_stale_chunks
python -m scripts.rebuild_index
python -m scripts.test_retrieval_quality --threshold 0.75
```

Make targets:

```bash
make reindex-changed
make delete-stale
make rebuild-index
make retrieval-quality
```

What each command does:

- `reindex_changed_docs`: hashes source files and only indexes changed documents.
- `reindex_changed_docs --delete-stale`: reindexes changed files, removes deleted source documents, and prunes inactive old versions.
- `delete_stale_chunks`: removes indexed documents whose source files no longer exist and deletes inactive old document versions.
- `rebuild_index`: deletes and rebuilds the local SQLite index.
- `test_retrieval_quality`: checks whether expected documents appear in top-k retrieval.

Retrieval quality reports are written to:

```text
data/eval/retrieval_quality_report.json
```

Why active versions matter: when a document changes, old chunks should not keep answering user questions. The vector store now marks older document versions inactive and retrieval searches only active versions. Cleanup can then prune inactive versions once you are confident the new index is good.

Rollback strategy:

- Prompt rollback: restore the previous prompt file and use the previous `prompt_version`.
- Model rollback: restore `LLM_PROVIDER` and `LLM_MODEL` in `.env`.
- Index rollback: restore `data/processed/ragops.db` from backup or rebuild from known-good raw docs.
- Deployment rollback: check out a known-good commit and run `docker compose up -d --build`.

Detailed rollback notes:

```text
docs/retraining_and_rollback.md
```

## Phase 9: Docker

Build and run the full local stack:

```bash
cp .env.example .env
docker compose config
docker compose up -d --build
```

Services:

- API: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`
- Postgres: `127.0.0.1:5432`
- Qdrant: `http://127.0.0.1:6333`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

Optional MLflow:

```bash
docker compose --profile mlflow up -d --build
```

Stop:

```bash
docker compose down
```

Common Docker mistakes:

- Exposing Postgres, Qdrant, Prometheus, or Grafana publicly without auth.
- Forgetting that `.env.example` is not a secret store.
- Rebuilding containers while expecting local SQLite data to persist outside mounted `./data`.
- Running paid LLM providers in unattended demos without API usage limits.
- Forgetting that Compose overlays `.env` on top of `.env.example`; real secrets belong in `.env` only.
- Binding service ports to `0.0.0.0` on a public Droplet instead of keeping them on `127.0.0.1` behind Nginx.

## Phase 10: GitHub Actions CI/CD

Workflow:

```text
.github/workflows/ci-cd.yml
```

The workflow runs:

- install dependencies
- `pytest -q`
- RAG evaluation gate
- retrieval quality gate
- Docker image build
- optional SSH deploy to a DigitalOcean Droplet

Required GitHub secrets for deployment:

- `DO_HOST`
- `DO_USER`
- `SSH_PRIVATE_KEY`
- `OPENAI_API_KEY` or `GEMINI_API_KEY` if using hosted LLMs

Deployment is guarded by repository variable:

```text
ENABLE_DROPLET_DEPLOY=true
```

Details:

```text
docs/github_secrets.md
```

## Phase 11: DigitalOcean Deployment

Recommended first deployment:

- Ubuntu CPU Droplet
- Basic shared CPU plan
- no GPU
- Docker Compose
- Nginx reverse proxy
- HTTPS with Certbot

Setup guide:

```text
docs/digitalocean_deploy.md
```

Install server dependencies:

```bash
bash deploy/setup_droplet.sh
```

Run app:

```bash
docker compose config
docker compose up -d --build
```

Nginx config:

```text
deploy/nginx/ragops.conf
```

Cost control:

- destroy unused Droplets instead of only powering them off
- avoid GPUs
- avoid unnecessary snapshots, backups, and volumes
- keep hosted LLM usage limits low
- expose only ports `80`, `443`, and `22`

## Testing

Tests cover:

- document parsing and chunking
- vector store insert/search
- API health
- document upload validation
- ask endpoint and trace creation
- feedback capture
- Prometheus metrics endpoint
- evaluation pipeline
- reindex changed documents
- stale document deletion
- retrieval quality gate

Run:

```bash
pytest -q
```

## Repository Structure

```text
app/
frontend/
data/
  raw/
  processed/
  eval/
prompts/
scripts/
tests/
monitoring/
  prometheus.yml
  grafana/
docker/
deploy/
docs/
.github/workflows/
Dockerfile
docker-compose.yml
requirements.txt
.env.example
Makefile
README.md
```

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
- Treating the final answer as the only thing worth logging.
- Logging prompts without trace IDs, which makes debugging production failures painful.
- Using paid LLM calls in every local test.
- Comparing prompts without tracking prompt versions in each trace.
- Shipping prompt changes without running an evaluation gate.
- Monitoring only uptime while ignoring latency, cost, feedback, and citation quality.
- Treating hallucination risk as a vague feeling instead of a measurable regression signal.
- Rebuilding every document on every deployment instead of hashing and reindexing changed files.
- Leaving old chunks in the index after source documents are deleted.
- Publishing DigitalOcean service ports directly instead of routing through Nginx and firewall rules.

## Verification Checklist

- `pytest -q` passes.
- `GET /health` returns `{"status":"ok", ...}`.
- Uploading a `.txt`, `.md`, or text-based `.pdf` returns chunks.
- `GET /documents` shows the uploaded document.
- `POST /ask` returns relevant citation chunks.
- `POST /ask` returns an answer, model name, latency, token usage, cost, and `trace_id`.
- `GET /traces/{trace_id}` returns the rendered prompt and saved citations.
- `POST /traces/{trace_id}/feedback` records thumbs-up/down feedback.
- `python -m scripts.run_evaluation --threshold 0.65` passes.
- `python -m scripts.run_evaluation --compare-prompts v1 v2 --threshold 0.65` produces `data/eval/evaluation_report.json`.
- `GET /metrics` returns `ragops_requests_total`.
- `python -m scripts.test_retrieval_quality --threshold 0.75` passes.
- `docker compose build api frontend` succeeds.
- `docker compose config` validates the full stack.
- `data/processed/ragops.db` is created after ingestion.

## Screenshots

Add screenshots here as the frontend evolves:

- `docs/screenshots/upload.png`
- `docs/screenshots/answer-with-citations.png`
- `docs/screenshots/monitoring-dashboard.png`

## Future Improvements

- Replace local hash embeddings with OpenAI, Gemini, or sentence-transformers embeddings.
- Move metadata from SQLite to Postgres.
- Move vectors from SQLite to Qdrant.
- Add authenticated users and per-tenant document permissions.
- Add Langfuse SDK trace export.
- Add Grafana provisioning for automatic dashboard import.
- Add immutable image publishing to DigitalOcean Container Registry.
