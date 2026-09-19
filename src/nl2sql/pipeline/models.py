"""Data passed between pipeline stages.

Two kinds of type live here. The pydantic models are contracts with a language
model: they define exactly what a model must return, and they are the schema
the provider enforces. The dataclasses are internal results passed from one
stage to the next.

The generation contract deliberately asks for a short description of what the
finished query does, never for the reasoning that produced it. Internal chain
of thought is not requested, not stored and not returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nl2sql.metadata.models import TableInfo

IntentType = Literal["lookup", "aggregation", "ranking", "comparison", "trend", "anomaly"]

FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class IntentAnalysis(BaseModel):
    """What kind of answer a question is asking for."""

    model_config = ConfigDict(extra="ignore")

    intent_type: IntentType = "lookup"
    time_expressions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    ambiguity: float = 0.3
    is_data_question: bool = True

    @field_validator("ambiguity")
    @classmethod
    def _clamp_ambiguity(cls, value: float) -> float:
        return _clamp(value)


class GeneratedSQL(BaseModel):
    """The structured output a model must return when asked for SQL."""

    model_config = ConfigDict(extra="ignore")

    sql: str = ""
    reasoning_summary: str = ""
    tables_used: list[str] = Field(default_factory=list)
    columns_used: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)

    @field_validator("sql")
    @classmethod
    def _clean_sql(cls, value: str) -> str:
        """Strip the code fence and trailing semicolon models add out of habit."""
        text = (value or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().lower().startswith("sql"):
                    text = text.lstrip()[3:]
            text = text.strip()
        while text.endswith(";"):
            text = text[:-1].rstrip()
        return text

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return _clamp(value)


class VerificationVerdict(BaseModel):
    """A second model's judgement of whether a query answers the question."""

    model_config = ConfigDict(extra="ignore")

    agrees: bool = True
    issues: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return _clamp(value)


class TableSelection(BaseModel):
    """The tables a model believes are needed."""

    model_config = ConfigDict(extra="ignore")

    tables: list[str] = Field(default_factory=list)


class FollowupRewrite(BaseModel):
    """A follow up question rewritten to stand on its own."""

    model_config = ConfigDict(extra="ignore")

    standalone_question: str = ""


class AnswerOutput(BaseModel):
    """The written answer."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""


# ---------------------------------------------------------------------------
# Internal stage results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QueryFilter:
    """A caller supplied filter applied to the result of the generated query."""

    column: str
    operator: FilterOperator
    value: Any


@dataclass(slots=True)
class NormalizedQuestion:
    """The question after cleaning, screening and follow up resolution."""

    original: str
    text: str
    is_followup: bool = False
    injection_findings: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedSchema:
    """The tables chosen for the prompt and the context text describing them."""

    tables: tuple[TableInfo, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    context: str = ""
    candidate_count: int = 0
    join_edges: int = 0
    llm_selected: bool = False

    @property
    def table_names(self) -> tuple[str, ...]:
        """Return the qualified names of the chosen tables."""
        return tuple(table.qualified_name for table in self.tables)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Which model generates, which one verifies, and why."""

    primary: str
    verifier: str | None
    complexity: float
    features: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found in generated SQL."""

    code: str
    message: str
    severity: str = "error"


@dataclass(slots=True)
class ValidationReport:
    """The result of validating one candidate query."""

    valid: bool = False
    issues: list[ValidationIssue] = field(default_factory=list)
    display_sql: str = ""
    execution_sql: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    statement_kind: str | None = None
    tables: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()
    masked_outputs: frozenset[str] = frozenset()
    row_limit_applied: int | None = None
    scoped_tables: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return only the issues that block execution."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return the issues that are advisory."""
        return tuple(issue for issue in self.issues if issue.severity != "error")


@dataclass(frozen=True, slots=True)
class ColumnMeta:
    """One column of a result set."""

    name: str
    type_name: str = "unknown"


@dataclass(slots=True)
class ExecutionResult:
    """Rows returned by the database, already capped."""

    columns: tuple[ColumnMeta, ...] = ()
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    truncated: bool = False
    size_limited: bool = False
    execution_ms: float = 0.0

    @property
    def row_count(self) -> int:
        """Return how many rows are being returned."""
        return len(self.rows)


@dataclass(slots=True)
class FormattedResult:
    """The result set converted to JSON safe values."""

    columns: list[dict[str, str]] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)


@dataclass(slots=True)
class CandidateSQL:
    """One generated query and everything known about its quality."""

    generated: GeneratedSQL
    provider: str
    model: str
    report: ValidationReport
    origin: str = "primary"
    verdict: VerificationVerdict | None = None
    dry_run_ok: bool | None = None
    score: float = 0.0
    generation_error: str | None = None

    @property
    def valid(self) -> bool:
        """Return whether this candidate passed validation."""
        return self.report.valid


@dataclass(slots=True)
class SelectionOutcome:
    """The winning candidate and how the choice was made."""

    winner: CandidateSQL
    candidates: list[CandidateSQL] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    verified: bool = False
    escalated: bool = False
    repairs: int = 0


@dataclass(slots=True)
class QueryOutcome:
    """Everything one question produced."""

    query_id: str
    answer: str | None = None
    sql: str | None = None
    columns: list[dict[str, str]] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    valid: bool = False
    issues: list[ValidationIssue] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    intent: str | None = None
    routing: RoutingDecision | None = None
    tables_used: tuple[str, ...] = ()
    columns_used: tuple[str, ...] = ()
    reasoning_summary: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    executed: bool = False
