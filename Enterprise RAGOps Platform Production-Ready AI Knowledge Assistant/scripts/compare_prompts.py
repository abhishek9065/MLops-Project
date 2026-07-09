from __future__ import annotations

import json

from scripts.run_evaluation import DEFAULT_DATASET, DEFAULT_REPORT, DEFAULT_SOURCE_DIR, run_evaluation, summarize


def main() -> None:
    report = {}
    for prompt_version in ["v1", "v2"]:
        results = run_evaluation(DEFAULT_DATASET, DEFAULT_SOURCE_DIR, prompt_version)
        report[prompt_version] = summarize(results)

    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for prompt_version, summary in report.items():
        print(f"{prompt_version}: {summary}")


if __name__ == "__main__":
    main()
