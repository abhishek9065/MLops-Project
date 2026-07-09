"""
Roll the Production model back to a previous version.

Usage:
    python -m src.models.rollback              # list all versions + stages
    python -m src.models.rollback --to 1       # make version 1 Production again

WHEN YOU'D USE THIS: a freshly promoted model starts misbehaving in production
(bad predictions, latency, a data bug). Instead of a panicked retrain, you
reinstate the last known-good version in seconds. This is only possible because
promotion ARCHIVES old versions rather than deleting them.
"""
from __future__ import annotations

import argparse

from src.models.registry import list_versions, rollback_to, setup_mlflow
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback the Production model.")
    parser.add_argument("--to", help="Version number to promote back to Production")
    args = parser.parse_args()

    setup_mlflow()
    versions = list_versions()
    if not versions:
        raise SystemExit("No registered versions found. Train a model first.")

    logger.info("Registered versions:")
    for v in versions:
        marker = "  <== PRODUCTION" if v["stage"] == "Production" else ""
        logger.info("  v%s [%s]%s", v["version"], v["stage"], marker)

    if args.to:
        rollback_to(args.to)


if __name__ == "__main__":
    main()
