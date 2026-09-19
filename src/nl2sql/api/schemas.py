"""Request and response models.

The response carries what a caller needs to use and to trust the answer: the
answer itself, the SQL that produced it, the rows, how long it took, how
confident the system is and what it is unsure about.

It deliberately leaves out everything internal: which model was used, the
prompt versions, the schema context, token counts and stage timings. Those go
to the log and the audit trail, where operators can see them, rather than to
anyone who can call the API.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Identifiers a caller may supply. Anything else is rejected before it can
#: reach a log line, a prompt or a storage key.
_IDENTIFIER = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"]

FilterValue = str | int | float | bool | None | list[str | int | float]


class FilterModel(BaseModel):
    """A filter applied to the columns the generated query returns."""

    model_config = ConfigDict(extra="forbid")

    column: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[\w ]+$")]
    operator: FilterOperator = "eq"
    value: FilterValue = None


class UserContextModel(BaseModel):
    """Optional context about the caller's session."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: _IDENTIFIER | None = None
    locale: Annotated[str, Field(max_length=16)] | None = None
    timezone: Annotated[str, Field(max_length=64)] | None = None


class QueryRequest(BaseModel):
    """A question to answer."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(min_length=1, max_length=4000)]
    user_context: UserContextModel | None = None
    filters: list[FilterModel] = Field(default_factory=list, max_length=20)


class ColumnModel(BaseModel):
    """One column of a result set."""

    name: str
    type: str = "unknown"


class IssueModel(BaseModel):
    """One validation finding."""

    code: str
    message: str
    severity: str = "error"


class QueryResponse(BaseModel):
    """The answer to a question."""

    query_id: str
    answer: str | None = None
    sql: str | None = None
    columns: list[ColumnModel] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    execution_time: float = Field(default=0.0, description="Query execution time in seconds.")
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ValidateResponse(BaseModel):
    """The result of validating a question without running anything."""

    query_id: str
    valid: bool
    sql: str | None = None
    explanation: str = ""
    tables_used: list[str] = Field(default_factory=list)
    issues: list[IssueModel] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SchemaSummary(BaseModel):
    """One schema and the tables it holds."""

    name: str
    table_count: int
    tables: list[str] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    """Every schema the caller may query."""

    schemas: list[SchemaSummary] = Field(default_factory=list)
    table_count: int = 0
    discovered_at: str | None = None


class TableColumnModel(BaseModel):
    """One column of a table."""

    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    description: str | None = None


class ForeignKeyModel(BaseModel):
    """One discovered relationship."""

    columns: list[str]
    references: str
    referenced_columns: list[str] = Field(default_factory=list)


class TableModel(BaseModel):
    """One table or view with its columns and relationships."""

    schema_name: str
    name: str
    kind: str = "table"
    description: str | None = None
    columns: list[TableColumnModel] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyModel] = Field(default_factory=list)


class TablesResponse(BaseModel):
    """The tables the caller may query."""

    tables: list[TableModel] = Field(default_factory=list)
    count: int = 0


class HealthResponse(BaseModel):
    """Liveness."""

    status: str = "ok"
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness, as a set of named checks."""

    status: str
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    """The error envelope."""

    code: str
    message: str
    request_id: str | None = None
    query_id: str | None = None
    issues: list[IssueModel] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Every failure is returned in this shape."""

    error: ErrorBody
