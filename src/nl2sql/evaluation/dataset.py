"""Evaluation datasets.

A case states what a correct answer looks like from several angles, because no
single angle is sufficient. The reference SQL allows a comparison of meaning
and of results; the expected tables and columns allow retrieval to be scored
even when the SQL differs; and the expected result characteristics catch a
query that returns plausible but wrong shaped output.

Reference SQL carries the dialect it was written in, so a dataset written for
Azure SQL can be evaluated against a development database of another kind by
transpiling it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from nl2sql.core.exceptions import ConfigurationError


class ExpectedResult(BaseModel):
    """What the result set should look like."""

    model_config = ConfigDict(extra="forbid")

    non_empty: bool | None = None
    row_count: int | None = None
    min_rows: int | None = None
    max_rows: int | None = None
    columns: list[str] = Field(default_factory=list)
    ordered_by: str | None = None
    descending: bool = True


class EvaluationCase(BaseModel):
    """One question with everything needed to judge the answer."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_tables: list[str] = Field(default_factory=list)
    expected_columns: list[str] = Field(default_factory=list)
    reference_sql: str = ""
    reference_dialect: str = "tsql"
    expected_result: ExpectedResult = Field(default_factory=ExpectedResult)
    tags: list[str] = Field(default_factory=list)


def load_dataset(path: Path | str) -> list[EvaluationCase]:
    """Load cases from a JSON Lines, JSON or YAML file."""
    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(f"The evaluation dataset {source} was not found.")

    text = source.read_text(encoding="utf-8")
    raw: list[Any]
    if source.suffix == ".jsonl":
        raw = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("//")
        ]
    elif source.suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text) or []
        raw = loaded if isinstance(loaded, list) else loaded.get("cases", [])
    else:
        loaded = json.loads(text)
        raw = loaded if isinstance(loaded, list) else loaded.get("cases", [])

    cases = [EvaluationCase.model_validate(item) for item in raw]
    identifiers = [case.id for case in cases]
    duplicates = {item for item in identifiers if identifiers.count(item) > 1}
    if duplicates:
        raise ConfigurationError(
            f"The dataset {source} has duplicate case ids: {sorted(duplicates)}"
        )
    return cases
