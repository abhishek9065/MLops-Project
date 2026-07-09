from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.document_loader import chunk_text, load_document
from app.vector_store import SQLiteVectorStore
from scripts.run_evaluation import DEFAULT_DATASET, DEFAULT_SOURCE_DIR


__test__ = False
DEFAULT_REPORT = Path("data/eval/retrieval_quality_report.json")


def run_retrieval_quality_test(dataset_path: Path, source_dir: Path, threshold: float, top_k: int) -> dict:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        store = SQLiteVectorStore(Path(temp_dir) / "retrieval.db")
        for path in sorted(source_dir.glob("*")):
            if path.is_file():
                document = load_document(path)
                store.upsert_document(document, chunk_text(document.text))

        rows = []
        hits = 0
        total = 0
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            example = json.loads(line)
            results = store.search(example["question"], top_k=top_k)
            retrieved = [result.filename for result in results]
            hit = example["expected_source_document"] in retrieved
            hits += int(hit)
            rows.append(
                {
                    "question": example["question"],
                    "expected_source_document": example["expected_source_document"],
                    "retrieved_documents": retrieved,
                    "hit": hit,
                }
            )

    score = hits / total if total else 0
    return {"score": round(score, 4), "threshold": threshold, "passed": score >= threshold, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether retrieval returns expected source documents.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = run_retrieval_quality_test(args.dataset, args.source_dir, args.threshold, args.top_k)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
