"""Audit trail.

One record per question, whether it succeeded, was refused by validation or
failed in the database. It answers the questions an operator actually asks
after the fact: what was asked, what SQL that produced, which model wrote it,
whether validation passed, what ran, how long it took and how much it cost.

What it deliberately does not hold: the caller's identity in the clear, the
question in the clear, or any row of data. Identities are hashed, the question
is masked, and results are never recorded.

A failure to write the audit row never fails the request. The same event is
already on the structured log, and returning an error for a query that
succeeded would be the worse outcome.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from nl2sql.core.masking import Masker, hash_identifier, mask_sql_literals
from nl2sql.db.repositories import AuditRepository
from nl2sql.observability.logging import get_logger
from nl2sql.observability.metrics import MetricsRegistry

logger = get_logger(__name__)


@dataclass(slots=True)
class AuditEntry:
    """Everything recorded about one question."""

    query_id: str
    request_id: str | None = None
    principal_id: str | None = None
    tenant_id: str | None = None
    question: str = ""
    intent: str | None = None
    complexity: float | None = None
    primary_provider: str | None = None
    primary_model: str | None = None
    verifier_provider: str | None = None
    verifier_model: str | None = None
    prompt_versions: dict[str, str] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generated_sql: str | None = None
    tables_used: tuple[str, ...] = ()
    validation_passed: bool = False
    validation_issues: list[dict[str, str]] = field(default_factory=list)
    execution_status: str = "not_executed"
    row_count: int | None = None
    truncated: bool = False
    confidence: float | None = None
    latency_ms: float | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    error_code: str | None = None


class AuditLogger:
    """Writes audit entries to the log, the metrics and the database."""

    def __init__(
        self,
        *,
        repository: AuditRepository | None,
        masker: Masker,
        metrics: MetricsRegistry | None = None,
        enabled: bool = True,
        log_sql: bool = True,
        mask_sql: bool = True,
        dialect: str | None = None,
    ) -> None:
        self._repository = repository
        self._masker = masker
        self._metrics = metrics
        self._enabled = enabled
        self._log_sql = log_sql
        self._mask_sql = mask_sql
        self._dialect = dialect

    async def record(self, entry: AuditEntry) -> None:
        """Record one entry everywhere it belongs."""
        sql = self._prepare_sql(entry.generated_sql)
        question = self._masker.mask_text(entry.question)

        logger.info(
            "query_audit",
            query_id=entry.query_id,
            intent=entry.intent,
            complexity=entry.complexity,
            primary_model=entry.primary_model,
            verifier_model=entry.verifier_model,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            validation_passed=entry.validation_passed,
            validation_issues=[issue.get("code") for issue in entry.validation_issues],
            execution_status=entry.execution_status,
            row_count=entry.row_count,
            truncated=entry.truncated,
            confidence=entry.confidence,
            latency_ms=round(entry.latency_ms or 0.0, 1),
            stage_timings={k: round(v, 1) for k, v in entry.stage_timings.items()},
            error_code=entry.error_code,
            tables_used=list(entry.tables_used),
            sql=sql if self._log_sql else None,
        )
        self._record_metrics(entry)

        if not self._enabled or self._repository is None:
            return

        values: dict[str, Any] = {
            "query_id": entry.query_id,
            "request_id": entry.request_id,
            "principal_hash": hash_identifier(entry.principal_id),
            "tenant_hash": hash_identifier(entry.tenant_id),
            "question_masked": question,
            "question_sha256": hashlib.sha256(entry.question.encode("utf-8")).hexdigest(),
            "intent": entry.intent,
            "complexity": entry.complexity,
            "primary_provider": entry.primary_provider,
            "primary_model": entry.primary_model,
            "verifier_provider": entry.verifier_provider,
            "verifier_model": entry.verifier_model,
            "prompt_versions": json.dumps(entry.prompt_versions) if entry.prompt_versions else None,
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "generated_sql": sql,
            "tables_used": json.dumps(list(entry.tables_used)) if entry.tables_used else None,
            "validation_passed": entry.validation_passed,
            "validation_issues": (
                json.dumps(entry.validation_issues) if entry.validation_issues else None
            ),
            "execution_status": entry.execution_status,
            "row_count": entry.row_count,
            "truncated": entry.truncated,
            "confidence": entry.confidence,
            "latency_ms": entry.latency_ms,
            "stage_timings": json.dumps(entry.stage_timings) if entry.stage_timings else None,
            "error_code": entry.error_code,
        }
        try:
            await asyncio.to_thread(self._repository.record, values)
        except Exception as exc:
            logger.warning(
                "audit_write_failed", query_id=entry.query_id, error_type=type(exc).__name__
            )

    def _prepare_sql(self, sql: str | None) -> str | None:
        """Mask literals in SQL before it is stored or logged."""
        if not sql:
            return None
        if not self._mask_sql:
            return sql
        return mask_sql_literals(sql, dialect=self._dialect)

    def _record_metrics(self, entry: AuditEntry) -> None:
        if self._metrics is None:
            return
        self._metrics.increment(
            "nl2sql_queries_total",
            labels={
                "status": entry.execution_status,
                "validated": str(entry.validation_passed).lower(),
            },
        )
        if entry.latency_ms is not None:
            self._metrics.observe("nl2sql_query_latency_ms", entry.latency_ms)
        if entry.error_code:
            self._metrics.increment("nl2sql_query_errors_total", labels={"code": entry.error_code})
        for stage, milliseconds in entry.stage_timings.items():
            self._metrics.observe("nl2sql_stage_latency_ms", milliseconds, labels={"stage": stage})

    @staticmethod
    def as_dict(entry: AuditEntry) -> dict[str, Any]:
        """Return the entry as a mapping, for tests and support tooling."""
        return asdict(entry)
