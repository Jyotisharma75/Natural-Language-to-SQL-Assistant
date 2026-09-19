"""The evaluation runner.

Runs a dataset through the real pipeline and measures what happened. Every
number in the report is computed from an actual run: nothing is assumed, and
a case that could not be scored on some axis is reported as not scored rather
than counted as a pass or a failure.

The reference query is executed through the same validator and executor as a
generated one. A dataset is a file on disk and is not automatically more
trustworthy than model output.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import sqlglot
from sqlglot.errors import SqlglotError

from nl2sql.config.settings import EvaluationSettings
from nl2sql.core.exceptions import AppError
from nl2sql.evaluation.dataset import EvaluationCase
from nl2sql.evaluation.metrics import (
    SetScore,
    ast_equivalent,
    check_characteristics,
    percentile,
    rate,
    results_match,
    set_score,
)
from nl2sql.metadata.service import MetadataService
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.orchestrator import QueryCommand, QueryPipeline
from nl2sql.pipeline.sql_executor import SQLExecutor
from nl2sql.pipeline.sql_validator import SQLValidator
from nl2sql.security.auth import Principal

logger = get_logger(__name__)


@dataclass(slots=True)
class CaseOutcome:
    """What one evaluation case produced."""

    case_id: str
    question: str
    sql: str | None = None
    sql_valid: bool = False
    executed: bool = False
    table_score: SetScore | None = None
    column_score: SetScore | None = None
    semantically_equivalent: bool | None = None
    results_match: bool | None = None
    characteristic_failures: list[str] = field(default_factory=list)
    row_count: int | None = None
    latency_ms: float = 0.0
    error_code: str | None = None
    error: str | None = None

    @property
    def correct(self) -> bool | None:
        """Return whether the case is correct, or None when it could not be judged.

        Matching results is the strongest evidence available. When the
        reference could not be run, equivalence of the statements is used
        instead, and when neither is available the case is not scored.
        """
        if self.results_match is not None:
            return self.results_match and not self.characteristic_failures
        if self.semantically_equivalent is not None:
            return self.semantically_equivalent and not self.characteristic_failures
        return None


@dataclass(slots=True)
class EvaluationReport:
    """Every case outcome plus the aggregates computed from them."""

    cases: list[CaseOutcome] = field(default_factory=list)
    aggregates: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    dataset: str = ""
    dialect: str = ""

    def compute_aggregates(self) -> dict[str, Any]:
        """Compute the summary from the case outcomes."""
        total = len(self.cases)
        latencies = [case.latency_ms for case in self.cases]
        judged = [case for case in self.cases if case.correct is not None]
        table_scores = [c.table_score.f1 for c in self.cases if c.table_score is not None]
        column_scores = [c.column_score.f1 for c in self.cases if c.column_score is not None]
        compared = [c for c in self.cases if c.results_match is not None]
        equivalent = [c for c in self.cases if c.semantically_equivalent is not None]

        self.aggregates = {
            "cases": total,
            "sql_valid": sum(1 for c in self.cases if c.sql_valid),
            "sql_validity_rate": rate(sum(1 for c in self.cases if c.sql_valid), total),
            "executed": sum(1 for c in self.cases if c.executed),
            "execution_success_rate": rate(sum(1 for c in self.cases if c.executed), total),
            "cases_scored_for_correctness": len(judged),
            "correct": sum(1 for c in judged if c.correct),
            "correctness_rate_of_scored": rate(sum(1 for c in judged if c.correct), len(judged)),
            "result_comparisons": len(compared),
            "result_match_rate_of_compared": rate(
                sum(1 for c in compared if c.results_match), len(compared)
            ),
            "semantic_comparisons": len(equivalent),
            "semantic_equivalence_rate_of_compared": rate(
                sum(1 for c in equivalent if c.semantically_equivalent), len(equivalent)
            ),
            "table_f1_mean": round(sum(table_scores) / len(table_scores), 4)
            if table_scores
            else None,
            "column_f1_mean": round(sum(column_scores) / len(column_scores), 4)
            if column_scores
            else None,
            "characteristic_failures": sum(1 for c in self.cases if c.characteristic_failures),
            "errors": sum(1 for c in self.cases if c.error_code),
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p95": percentile(latencies, 0.95),
            "latency_ms_mean": round(sum(latencies) / total, 2) if total else 0.0,
        }
        return self.aggregates


class EvaluationRunner:
    """Runs evaluation cases through the pipeline and scores the results."""

    def __init__(
        self,
        *,
        pipeline: QueryPipeline,
        validator: SQLValidator,
        executor: SQLExecutor,
        metadata: MetadataService,
        settings: EvaluationSettings,
        principal: Principal | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._validator = validator
        self._executor = executor
        self._metadata = metadata
        self._settings = settings
        self._principal = principal or Principal(id="evaluation", tenant_id=None, roles=("reader",))

    async def run(
        self, cases: Sequence[EvaluationCase], *, dataset_name: str = ""
    ) -> EvaluationReport:
        """Run every case and return the report."""
        report = EvaluationReport(
            started_at=datetime.now(UTC).isoformat(),
            dataset=dataset_name,
            dialect=self._executor.dialect,
        )
        for case in cases:
            report.cases.append(await self._run_case(case))
        report.finished_at = datetime.now(UTC).isoformat()
        report.compute_aggregates()
        logger.info("evaluation_completed", dataset=dataset_name, **report.aggregates)
        return report

    async def _run_case(self, case: EvaluationCase) -> CaseOutcome:
        """Run one case and score every axis that can be scored."""
        outcome = CaseOutcome(case_id=case.id, question=case.question)
        started = time.perf_counter()
        try:
            result = await self._pipeline.run(
                QueryCommand(question=case.question, principal=self._principal)
            )
        except AppError as exc:
            outcome.latency_ms = (time.perf_counter() - started) * 1000
            outcome.error_code = exc.code
            outcome.error = exc.message[:300]
            logger.info("evaluation_case_failed", case=case.id, code=exc.code)
            return outcome

        outcome.latency_ms = (time.perf_counter() - started) * 1000
        outcome.sql = result.sql
        outcome.sql_valid = result.valid
        outcome.executed = result.executed
        outcome.row_count = result.row_count
        outcome.table_score = set_score(case.expected_tables, result.tables_used)
        outcome.column_score = set_score(case.expected_columns, result.columns_used, tails=True)

        columns = [column["name"] for column in result.columns]
        outcome.characteristic_failures = check_characteristics(
            case.expected_result, columns, result.rows
        )

        if case.reference_sql:
            reference = self._transpile(case)
            if reference:
                outcome.semantically_equivalent = ast_equivalent(
                    result.sql or "", reference, dialect=self._executor.sqlglot_dialect
                )
                reference_rows = await self._run_reference(reference)
                if reference_rows is not None:
                    outcome.results_match = results_match(
                        reference_rows,
                        [list(row) for row in result.rows],
                        tolerance=self._settings.float_tolerance,
                        column_order_sensitive=self._settings.column_order_sensitive,
                    )
        return outcome

    def _transpile(self, case: EvaluationCase) -> str | None:
        """Render the reference query in the dialect of the database under test."""
        target = self._executor.sqlglot_dialect
        if case.reference_dialect == target:
            return case.reference_sql
        try:
            converted = sqlglot.transpile(
                case.reference_sql, read=case.reference_dialect, write=target
            )
        except SqlglotError as exc:
            logger.warning(
                "reference_sql_not_transpilable", case=case.id, error_type=type(exc).__name__
            )
            return None
        return converted[0] if converted else None

    async def _run_reference(self, sql: str) -> list[list[Any]] | None:
        """Run the reference query through the same safety path as a generated one."""
        catalog = await self._metadata.catalog_async()
        report = self._validator.validate(sql, catalog, dialect=self._executor.sqlglot_dialect)
        if not report.valid:
            logger.warning(
                "reference_sql_invalid",
                issues=[issue.code for issue in report.errors],
            )
            return None
        try:
            executed = await self._executor.execute(
                report.execution_sql,
                report.parameters,
                max_rows=report.row_limit_applied,
            )
        except AppError as exc:
            logger.warning("reference_sql_failed", code=exc.code)
            return None
        return [list(row) for row in executed.rows]
