"""Evaluation reporting.

Writes what was measured and nothing else. Where a case could not be scored,
the report says so rather than quietly treating it as a pass, because an
accuracy figure that silently excludes the hard cases is worse than no figure.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nl2sql.evaluation.runner import EvaluationReport


def to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Return the report as a serialisable mapping."""
    return {
        "dataset": report.dataset,
        "dialect": report.dialect,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "aggregates": report.aggregates,
        "cases": [
            {
                key: (asdict(value) if hasattr(value, "__dataclass_fields__") else value)
                for key, value in asdict(case).items()
            }
            for case in report.cases
        ],
    }


def to_markdown(report: EvaluationReport) -> str:
    """Render the report as Markdown."""
    aggregates = report.aggregates
    lines = [
        "# Evaluation report",
        "",
        f"Dataset: `{report.dataset or 'unnamed'}`",
        f"Dialect: `{report.dialect}`",
        f"Started: {report.started_at}",
        f"Finished: {report.finished_at}",
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "| --- | --- |",
    ]
    for key, value in aggregates.items():
        label = key.replace("_", " ")
        lines.append(f"| {label} | {'not measured' if value is None else value} |")

    lines += [
        "",
        "Rates are over the cases that could be scored on that axis. A case whose",
        "reference query could not be run is not counted as either correct or",
        "incorrect.",
        "",
        "## Cases",
        "",
        "| Case | Valid | Executed | Table F1 | Column F1 | Results match | Correct | Latency ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.cases:
        cells = [
            case.case_id,
            "yes" if case.sql_valid else "no",
            "yes" if case.executed else "no",
            f"{case.table_score.f1:.2f}" if case.table_score else "n/a",
            f"{case.column_score.f1:.2f}" if case.column_score else "n/a",
            _tri(case.results_match),
            _tri(case.correct),
            f"{case.latency_ms:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    failures = [case for case in report.cases if case.correct is False or case.error_code]
    if failures:
        lines += ["", "## Cases needing attention", ""]
        for case in failures:
            detail = case.error or ", ".join(case.characteristic_failures) or "result mismatch"
            lines.append(f"- **{case.case_id}**: {detail}")
    return "\n".join(lines) + "\n"


def _tri(value: bool | None) -> str:
    """Render a three state value."""
    if value is None:
        return "not scored"
    return "yes" if value else "no"


def write_report(report: EvaluationReport, directory: Path | str) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports and return their paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = target / f"eval-{stamp}.json"
    markdown_path = target / f"eval-{stamp}.md"
    json_path.write_text(json.dumps(to_dict(report), indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, markdown_path
