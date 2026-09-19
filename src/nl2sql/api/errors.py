"""Error handling.

Two rules. Every failure comes back in the same envelope, so a client can
branch on ``error.code`` without parsing prose. And nothing internal crosses
the boundary: no stack trace, no driver message, no SQL, no table name that
the caller was not already allowed to see. The detail stays in the log, keyed
by the request and query identifiers that are in the response.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from nl2sql.api.schemas import ErrorBody, ErrorResponse, IssueModel
from nl2sql.core.context import get_query_id, get_request_id
from nl2sql.core.exceptions import AppError
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)


def _envelope(
    code: str, message: str, *, status: int, issues: list[IssueModel] | None = None
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=get_request_id(),
            query_id=get_query_id(),
            issues=issues or [],
        )
    )
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    """Install the handlers that produce the error envelope."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        issues = [
            IssueModel(code=str(issue.get("code", "issue")), message=str(issue.get("message", "")))
            for issue in exc.details.get("issues", [])
            if isinstance(issue, dict)
        ]
        log = logger.warning if exc.http_status < 500 else logger.error
        log(
            "request_failed",
            code=exc.code,
            status=exc.http_status,
            path=request.url.path,
            detail=exc.message,
        )
        return _envelope(exc.code, exc.public_message, status=exc.http_status, issues=issues)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # The field paths are returned, the submitted values are not.
        fields = [
            IssueModel(
                code="invalid_field",
                message=f"{'.'.join(str(part) for part in error.get('loc', ()))}: "
                f"{error.get('msg', 'is not valid')}",
            )
            for error in exc.errors()
        ]
        logger.warning("request_invalid", path=request.url.path, field_count=len(fields))
        return _envelope(
            "invalid_request",
            "The request body is not valid.",
            status=422,
            issues=fields[:10],
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "request_unhandled_error", path=request.url.path, error_type=type(exc).__name__
        )
        return _envelope(
            "internal_error",
            "An unexpected error occurred.",
            status=500,
        )

    def _unused(*_: Any) -> None:
        """Keep the handlers referenced for linters that miss decorators."""

    _unused(handle_app_error, handle_validation_error, handle_unexpected)
