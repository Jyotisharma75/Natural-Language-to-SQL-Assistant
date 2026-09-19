"""Domain exception hierarchy.

Every deliberate failure derives from :class:`AppError`. Each error carries a
stable machine readable ``code``, an HTTP status the API layer maps it onto,
a ``retryable`` flag the retry policies consult, and a ``public_message`` that
is safe to return to a caller. Internal detail goes in ``details`` and is only
ever logged, after masking.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every deliberate failure in the application."""

    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        public_message: str | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details: dict[str, Any] = details or {}
        self._public_message = public_message
        super().__init__(self.message)

    @property
    def public_message(self) -> str:
        """Return the message that is safe to show to an API caller."""
        return self._public_message or self.default_message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# Configuration -------------------------------------------------------------
class ConfigurationError(AppError):
    """Configuration is missing, malformed or internally inconsistent."""

    code = "configuration_error"
    default_message = "The service is not configured correctly."


class SecretNotFoundError(ConfigurationError):
    """A required secret could not be resolved."""

    code = "secret_not_found"


# Request input -------------------------------------------------------------
class InputValidationError(AppError):
    """Caller supplied input breaks a rule."""

    code = "invalid_input"
    default_message = "The request is not valid."
    http_status = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details, public_message=message)


class PromptInjectionError(AppError):
    """The question looks like an attempt to override system instructions."""

    code = "prompt_injection_detected"
    default_message = (
        "The question contains instructions or SQL fragments that are not allowed. "
        "Rephrase it as a plain question about the data."
    )
    http_status = 400


class AuthenticationError(AppError):
    """The caller could not be authenticated."""

    code = "unauthenticated"
    default_message = "Authentication is required."
    http_status = 401


class AuthorizationError(AppError):
    """The caller is authenticated but not permitted to do this."""

    code = "forbidden"
    default_message = "You are not allowed to perform this operation."
    http_status = 403


class TenantContextError(AuthorizationError):
    """Tenant isolation is enabled but the caller has no tenant."""

    code = "tenant_required"
    default_message = "A tenant context is required for this request."


class RateLimitExceededError(AppError):
    """The caller exceeded the configured request rate."""

    code = "rate_limited"
    default_message = "Too many requests. Try again shortly."
    http_status = 429


# Schema and metadata -------------------------------------------------------
class MetadataError(AppError):
    """Schema discovery failed."""

    code = "metadata_unavailable"
    default_message = "Database metadata is not available right now."
    http_status = 503
    retryable = True


class SchemaNotFoundError(AppError):
    """A requested schema or table is not available to the caller."""

    code = "schema_not_found"
    default_message = "The requested schema is not available."
    http_status = 404


class NoRelevantSchemaError(AppError):
    """No accessible table is relevant to the question."""

    code = "no_relevant_tables"
    default_message = (
        "No accessible tables appear relevant to this question. "
        "Try naming the subject area or metric more explicitly."
    )
    http_status = 422


# SQL validation ------------------------------------------------------------
class SQLValidationError(AppError):
    """Generated SQL failed safety or correctness validation."""

    code = "sql_validation_failed"
    default_message = "A safe query could not be produced for this question."
    http_status = 422


# Language models -----------------------------------------------------------
class LLMError(AppError):
    """A language model call failed and repeating it will not help."""

    code = "llm_error"
    default_message = "The language model could not complete the request."
    http_status = 502


class LLMUnavailableError(LLMError):
    """No usable language model provider is configured or reachable."""

    code = "llm_unavailable"
    default_message = "No language model is available right now."
    http_status = 503


class LLMRateLimitError(LLMError):
    """The provider throttled the request."""

    code = "llm_rate_limited"
    default_message = "The language model is busy. Try again shortly."
    http_status = 503
    retryable = True


class LLMTimeoutError(LLMError):
    """The provider did not respond in time."""

    code = "llm_timeout"
    default_message = "The language model timed out."
    http_status = 504
    retryable = True


class LLMServiceError(LLMError):
    """The provider failed on its side."""

    code = "llm_service_error"
    retryable = True


class LLMResponseError(LLMError):
    """The provider answered with output that does not match the contract."""

    code = "llm_invalid_response"
    default_message = "The language model returned an unusable response."


# Execution -----------------------------------------------------------------
class ExecutionError(AppError):
    """The database rejected or failed the query."""

    code = "query_execution_failed"
    default_message = "The query could not be executed."
    http_status = 500


class InvalidQueryReferenceError(ExecutionError):
    """The database reported an unknown object or column."""

    code = "invalid_query_reference"
    default_message = "The generated query referenced something the database does not recognise."
    http_status = 422


class QueryTimeoutError(ExecutionError):
    """The query exceeded the configured execution time."""

    code = "query_timeout"
    default_message = "The query took too long and was cancelled."
    http_status = 504


class TransientDatabaseError(ExecutionError):
    """A transient database fault that is safe to retry for read only work."""

    code = "database_unavailable"
    default_message = "The database is temporarily unavailable."
    http_status = 503
    retryable = True


class DatabasePermissionError(ExecutionError):
    """The database login lacks permission for the query."""

    code = "database_permission_denied"
    default_message = "The service account is not permitted to read that data."
    http_status = 403


class QueryCostExceededError(SQLValidationError):
    """The estimated cost of the query exceeds the configured ceiling."""

    code = "query_cost_exceeded"
    default_message = "The query is estimated to be too expensive to run."
