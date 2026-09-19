"""The query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from nl2sql.api.deps import ContainerDep, PrincipalDep
from nl2sql.api.schemas import (
    ColumnModel,
    ErrorResponse,
    IssueModel,
    QueryRequest,
    QueryResponse,
    ValidateResponse,
)
from nl2sql.core.exceptions import InputValidationError
from nl2sql.pipeline.models import QueryFilter
from nl2sql.pipeline.orchestrator import QueryCommand

router = APIRouter(prefix="/api/v1", tags=["query"])

_ERRORS: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "The question was rejected."},
    401: {"model": ErrorResponse, "description": "Authentication failed."},
    422: {"model": ErrorResponse, "description": "No safe query could be produced."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
    503: {"model": ErrorResponse, "description": "A dependency is unavailable."},
}


def _command(
    request: QueryRequest, container: ContainerDep, principal: PrincipalDep, *, validate_only: bool
) -> QueryCommand:
    """Build the pipeline command from a validated request."""
    maximum = container.settings.api.max_filters
    if len(request.filters) > maximum:
        raise InputValidationError(f"At most {maximum} filters may be supplied.")
    return QueryCommand(
        question=request.question,
        principal=principal,
        conversation_id=(request.user_context.conversation_id if request.user_context else None),
        filters=tuple(
            QueryFilter(column=f.column, operator=f.operator, value=f.value)
            for f in request.filters
        ),
        validate_only=validate_only,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses=_ERRORS,
    summary="Answer a question from the database",
)
async def run_query(
    request: QueryRequest, container: ContainerDep, principal: PrincipalDep
) -> QueryResponse:
    """Turn a question into SQL, run it, and explain the result."""
    outcome = await container.pipeline.run(
        _command(request, container, principal, validate_only=False)
    )
    expose_sql = container.settings.api.expose_sql
    return QueryResponse(
        query_id=outcome.query_id,
        answer=outcome.answer,
        sql=outcome.sql if expose_sql else None,
        columns=[
            ColumnModel(name=c["name"], type=c.get("type", "unknown")) for c in outcome.columns
        ],
        rows=outcome.rows,
        row_count=outcome.row_count,
        truncated=outcome.truncated,
        execution_time=round(outcome.execution_time_ms / 1000, 4),
        confidence=outcome.confidence,
        warnings=outcome.warnings,
    )


@router.post(
    "/query/validate",
    response_model=ValidateResponse,
    status_code=status.HTTP_200_OK,
    responses=_ERRORS,
    summary="Produce and check a query without running it",
)
async def validate_query(
    request: QueryRequest, container: ContainerDep, principal: PrincipalDep
) -> ValidateResponse:
    """Run every stage up to execution and report what would run."""
    outcome = await container.pipeline.run(
        _command(request, container, principal, validate_only=True)
    )
    expose_sql = container.settings.api.expose_sql
    return ValidateResponse(
        query_id=outcome.query_id,
        valid=outcome.valid,
        sql=outcome.sql if expose_sql else None,
        explanation=outcome.reasoning_summary,
        tables_used=list(outcome.tables_used),
        issues=[
            IssueModel(code=issue.code, message=issue.message, severity=issue.severity)
            for issue in outcome.issues
        ],
        confidence=outcome.confidence,
        warnings=outcome.warnings,
    )
