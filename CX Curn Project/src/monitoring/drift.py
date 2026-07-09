"""
Data drift detection using PSI (Population Stability Index) + KS test.

WHY PSI?
--------
PSI is the industry-standard, model-agnostic drift metric (born in credit
scoring). It compares the distribution of a feature in a REFERENCE set (training
data) against a CURRENT set (recent production data) and returns a single number:
  PSI < 0.10  -> no significant shift
  0.10-0.25   -> moderate shift, investigate
  PSI >= 0.25 -> significant shift, likely retrain

For numeric features we also run a Kolmogorov-Smirnov test (a p-value for "are
these two samples from the same distribution?"). PSI tells you *how much*; KS
gives a statistical yes/no.

WHY DRIFT != BAD MODEL (important nuance)
-----------------------------------------
Drift is a WARNING, not a verdict. The model might still perform fine. That's
why drift here TRIGGERS a retraining evaluation (retrain.py), which only
promotes a new model if it's genuinely better on fresh data. We never blindly
replace a model just because a distribution moved.

Alternative: the `evidently` library produces richer HTML reports. We use a
self-contained statistical method so the project has zero heavy dependencies and
you can see exactly how the numbers are computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_DATA_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
NO_DRIFT, MODERATE, SIGNIFICANT = "none", "moderate", "significant"


def _severity(psi: float) -> str:
    if psi < 0.10:
        return NO_DRIFT
    if psi < 0.25:
        return MODERATE
    return SIGNIFICANT


def _psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    # Bin edges are the quantiles of the REFERENCE data (so each ref bin ~= equal mass).
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0].astype(float)
    a = np.histogram(actual, bins=edges)[0].astype(float)
    e_perc = np.clip(e / e.sum(), eps, None)
    a_perc = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def _psi_categorical(expected: pd.Series, actual: pd.Series, eps: float = 1e-6) -> float:
    categories = set(expected.unique()) | set(actual.unique())
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    psi = 0.0
    for c in categories:
        e_perc = max(float(e.get(c, 0.0)), eps)
        a_perc = max(float(a.get(c, 0.0)), eps)
        psi += (a_perc - e_perc) * np.log(a_perc / e_perc)
    return float(psi)


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    features: dict[str, dict] = {}

    for col in NUMERIC_FEATURES:
        psi = _psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
        ks_p = float(ks_2samp(reference[col], current[col]).pvalue)
        features[col] = {
            "type": "numeric",
            "psi": round(psi, 4),
            "ks_pvalue": round(ks_p, 4),
            "severity": _severity(psi),
            "drifted": psi >= 0.10,
        }

    for col in CATEGORICAL_FEATURES:
        psi = _psi_categorical(reference[col], current[col])
        features[col] = {
            "type": "categorical",
            "psi": round(psi, 4),
            "severity": _severity(psi),
            "drifted": psi >= 0.10,
        }

    n_drifted = sum(1 for f in features.values() if f["drifted"])
    n_significant = sum(1 for f in features.values() if f["severity"] == SIGNIFICANT)
    dataset_drift = n_significant >= 1 or (n_drifted / len(features)) > 0.3

    return {
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "n_features": len(features),
        "n_drifted": n_drifted,
        "n_significant": n_significant,
        "dataset_drift": bool(dataset_drift),
        "features": features,
    }


def _write_html(report: dict, path: Path) -> None:
    rows = ""
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        color = {"none": "#2e7d32", "moderate": "#f9a825", "significant": "#c62828"}[f["severity"]]
        ks = f.get("ks_pvalue", "-")
        rows += (
            f"<tr><td>{name}</td><td>{f['type']}</td><td>{f['psi']}</td>"
            f"<td>{ks}</td><td style='color:{color};font-weight:bold'>{f['severity']}</td></tr>"
        )
    verdict = "DATASET DRIFT DETECTED" if report["dataset_drift"] else "No significant dataset drift"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Drift report</title>
<style>body{{font-family:system-ui,Arial;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}</style></head>
<body><h1>Data Drift Report</h1>
<p><b>{verdict}</b> &mdash; {report['n_drifted']}/{report['n_features']} features drifted,
{report['n_significant']} significant. Reference n={report['n_reference']}, current n={report['n_current']}.</p>
<table><tr><th>Feature</th><th>Type</th><th>PSI</th><th>KS p-value</th><th>Severity</th></tr>
{rows}</table>
<p style="color:#555">PSI &lt; 0.10 none &middot; 0.10&ndash;0.25 moderate &middot; &ge; 0.25 significant</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect data drift (PSI + KS).")
    parser.add_argument("--reference", default=str(RAW_DATA_FILE), help="Reference CSV (training data)")
    parser.add_argument("--current", required=True, help="Current CSV (new/incoming data)")
    parser.add_argument("--outdir", default=str(REPORTS_DIR), help="Where to write reports")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    report = compute_drift(reference, current)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "drift_report.json").write_text(json.dumps(report, indent=2))
    _write_html(report, outdir / "drift_report.html")

    logger.info(
        "Drift: dataset_drift=%s | drifted=%d/%d | significant=%d",
        report["dataset_drift"],
        report["n_drifted"],
        report["n_features"],
        report["n_significant"],
    )
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"])[:6]:
        logger.info("  %-20s psi=%.3f (%s)", name, f["psi"], f["severity"])
    logger.info("Reports written to %s", outdir)

    # Exit code 1 signals "drift detected" -- handy for CI/cron to trigger retraining.
    raise SystemExit(1 if report["dataset_drift"] else 0)


if __name__ == "__main__":
    main()
