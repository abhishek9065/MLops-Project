.PHONY: install api frontend test ingest rebuild-index

install:
	python -m pip install -r requirements.txt

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/streamlit_app.py

test:
	pytest -q

ingest:
	python -m scripts.ingest_documents data/raw

rebuild-index:
	python -m scripts.rebuild_index

