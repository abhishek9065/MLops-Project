.PHONY: install api frontend test ingest rebuild-index reindex-changed delete-stale retrieval-quality evaluate compare-prompts docker-build docker-up docker-down

install:
	python -m pip install -r requirements.txt

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

test:
	pytest -q

evaluate:
	python -m scripts.run_evaluation --threshold 0.65

compare-prompts:
	python -m scripts.run_evaluation --compare-prompts v1 v2 --threshold 0.65

ingest:
	python -m scripts.ingest_documents data/raw

rebuild-index:
	python -m scripts.rebuild_index

reindex-changed:
	python -m scripts.reindex_changed_docs --delete-stale

delete-stale:
	python -m scripts.delete_stale_chunks

retrieval-quality:
	python -m scripts.test_retrieval_quality --threshold 0.75

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
