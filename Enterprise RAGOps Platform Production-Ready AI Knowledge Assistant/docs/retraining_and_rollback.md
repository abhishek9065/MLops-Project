# RAGOps Rollback Strategy

RAG systems can regress even when the API still works. The most common causes are bad document parsing, poor chunking, prompt changes, embedding model changes, vector index corruption, and LLM provider changes.

## Before Rollout

Run:

```bash
pytest -q
python -m scripts.test_retrieval_quality --threshold 0.75
python -m scripts.run_evaluation --threshold 0.65
python -m scripts.run_evaluation --compare-prompts v1 v2 --threshold 0.65
```

## Rollback Targets

Prompt rollback:

- Restore the previous file in `prompts/`.
- Set requests back to the previous `prompt_version`.
- Confirm traces show the expected prompt version.

Model rollback:

- Restore `LLM_PROVIDER` and `LLM_MODEL` in `.env`.
- Restart the API container.
- Compare traces before and after rollback.

Index rollback:

- Restore a known-good `data/processed/ragops.db` backup.
- Or rebuild from known-good raw documents:

```bash
python -m scripts.rebuild_index --source-dir data/raw
```

Document rollback:

- Revert the changed raw document.
- Run:

```bash
python -m scripts.reindex_changed_docs --delete-stale
python -m scripts.delete_stale_chunks
```

Deployment rollback:

```bash
git log --oneline
git checkout <known_good_commit>
docker compose up -d --build
```

For production teams, prefer a tagged release and immutable Docker image rather than deploying directly from a mutable branch.

## What to Monitor After Rollback

- `ragops_llm_latency_seconds`
- `ragops_token_usage_total`
- `ragops_estimated_cost_usd_total`
- `ragops_feedback_score_total`
- evaluation `overall_score`
- citation correctness
- trace errors
