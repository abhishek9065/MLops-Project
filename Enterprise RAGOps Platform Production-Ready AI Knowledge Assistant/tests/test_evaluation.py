from pathlib import Path

from scripts.run_evaluation import DEFAULT_DATASET, DEFAULT_SOURCE_DIR, run_evaluation, summarize


def test_evaluation_pipeline_scores_above_threshold() -> None:
    results = run_evaluation(DEFAULT_DATASET, DEFAULT_SOURCE_DIR, prompt_version="v1")
    summary = summarize(results)

    assert len(results) == 4
    assert summary["overall_score"] >= 0.65
    assert Path("data/eval/evaluation_dataset.jsonl").exists()
