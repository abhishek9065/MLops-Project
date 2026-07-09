# RAGOps Release Guide

Every RAG release needs a rollback plan, health check, prompt version, embedding model version, and deployment owner.
Teams should compare prompt versions before rollout using answer relevance, faithfulness, citation correctness, and latency.
Bad prompt, model, or index updates should be rolled back by restoring the previous prompt file, model configuration, or vector index snapshot.

