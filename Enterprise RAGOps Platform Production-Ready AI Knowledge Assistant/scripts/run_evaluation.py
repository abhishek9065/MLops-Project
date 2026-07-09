from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from app.document_loader import chunk_text, load_document
from app.rag_pipeline import RagPipeline
from app.tracing import SQLiteTraceStore
from app.vector_store import SQLiteVectorStore


DEFAULT_DATASET = Path("data/eval/evaluation_dataset.jsonl")
DEFAULT_SOURCE_DIR = Path("data/eval/source_docs")
DEFAULT_REPORT = Path("data/eval/evaluation_report.json")


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    expected_answer: str
    expected_source_document: str
    expected_source_chunk: int


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    prompt_version: str
    answer: str
    expected_answer: str
    expected_source_document: str
    retrieved_documents: list[str]
    answer_relevance: float
    faithfulness: float
    context_precision: float
    context_recall: float
    citation_correctness: float
    hallucination_risk: float
    overall_score: float
    latency_ms: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic RAG evaluation gate.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--compare-prompts", nargs="*", default=None)
    args = parser.parse_args()

    prompt_versions = args.compare_prompts or [args.prompt_version]
    report = {
        "threshold": args.threshold,
        "prompt_versions": {},
    }
    exit_code = 0

    for prompt_version in prompt_versions:
        results = run_evaluation(
            dataset_path=args.dataset,
            source_dir=args.source_dir,
            prompt_version=prompt_version,
        )
        summary = summarize(results)
        report["prompt_versions"][prompt_version] = {
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        if summary["overall_score"] < args.threshold:
            exit_code = 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["prompt_versions"], indent=2))
    if exit_code:
        print(f"Evaluation failed. One or more prompt versions scored below {args.threshold}.", file=sys.stderr)
    raise SystemExit(exit_code)


def run_evaluation(dataset_path: Path, source_dir: Path, prompt_version: str) -> list[EvaluationResult]:
    examples = load_examples(dataset_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "eval.db"
        vector_store = SQLiteVectorStore(db_path)
        trace_store = SQLiteTraceStore(db_path)
        ingest_eval_sources(source_dir, vector_store)
        pipeline = RagPipeline(vector_store=vector_store, trace_store=trace_store)

        results = []
        for example in examples:
            response = pipeline.answer(example.question, top_k=4, prompt_version=prompt_version)
            retrieved_text = " ".join(citation.text for citation in response.citations)
            retrieved_documents = [citation.filename for citation in response.citations]
            answer_relevance = overlap_score(response.answer, example.expected_answer)
            faithfulness = overlap_score(response.answer, retrieved_text)
            context_precision = citation_precision(response.citations, example.expected_source_document)
            context_recall = overlap_score(example.expected_answer, retrieved_text)
            citation_correctness = citation_correctness_score(
                response.citations,
                expected_source_document=example.expected_source_document,
                expected_source_chunk=example.expected_source_chunk,
            )
            hallucination_risk = round(1 - faithfulness, 4)
            overall_score = round(
                (
                    answer_relevance * 0.25
                    + faithfulness * 0.25
                    + context_precision * 0.15
                    + context_recall * 0.15
                    + citation_correctness * 0.20
                ),
                4,
            )
            results.append(
                EvaluationResult(
                    question=example.question,
                    prompt_version=prompt_version,
                    answer=response.answer,
                    expected_answer=example.expected_answer,
                    expected_source_document=example.expected_source_document,
                    retrieved_documents=retrieved_documents,
                    answer_relevance=answer_relevance,
                    faithfulness=faithfulness,
                    context_precision=context_precision,
                    context_recall=context_recall,
                    citation_correctness=citation_correctness,
                    hallucination_risk=hallucination_risk,
                    overall_score=overall_score,
                    latency_ms=response.total_latency_ms,
                )
            )
    return results


def ingest_eval_sources(source_dir: Path, vector_store: SQLiteVectorStore) -> None:
    for path in sorted(source_dir.glob("*")):
        if not path.is_file():
            continue
        document = load_document(path, content_type="text/markdown")
        chunks = chunk_text(document.text)
        vector_store.upsert_document(document, chunks)


def load_examples(dataset_path: Path) -> list[EvaluationExample]:
    examples = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        examples.append(EvaluationExample(**payload))
    return examples


def summarize(results: list[EvaluationResult]) -> dict[str, float]:
    metric_names = [
        "answer_relevance",
        "faithfulness",
        "context_precision",
        "context_recall",
        "citation_correctness",
        "hallucination_risk",
        "overall_score",
        "latency_ms",
    ]
    return {
        metric_name: round(sum(getattr(result, metric_name) for result in results) / len(results), 4)
        for metric_name in metric_names
    }


def overlap_score(left: str, right: str) -> float:
    left_terms = normalize_terms(left)
    right_terms = normalize_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return round(len(left_terms & right_terms) / len(left_terms), 4)


def citation_precision(citations, expected_source_document: str) -> float:
    if not citations:
        return 0.0
    relevant = sum(1 for citation in citations if citation.filename == expected_source_document)
    return round(relevant / len(citations), 4)


def citation_correctness_score(citations, expected_source_document: str, expected_source_chunk: int) -> float:
    for citation in citations:
        if citation.filename == expected_source_document and citation.chunk_index == expected_source_chunk:
            return 1.0
    return 0.0


def normalize_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "is",
        "are",
        "what",
        "how",
        "why",
        "should",
        "must",
        "with",
        "using",
        "every",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


if __name__ == "__main__":
    main()
