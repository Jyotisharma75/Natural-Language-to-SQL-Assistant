"""Retry policies built from configuration.

Only failures that declare themselves retryable are retried. A validation
failure, an authorisation failure or a syntax error is never repeated, because
repeating it cannot change the outcome and, for anything that is not read only,
could change data twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from nl2sql.core.exceptions import AppError
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How an operation is retried.

    Attributes:
        name: Label used in log events.
        max_attempts: Total attempts including the first. ``1`` disables retry.
        initial_backoff_seconds: Delay before the second attempt.
        max_backoff_seconds: Ceiling for the exponential growth.
        jitter_seconds: Upper bound of random jitter added to each delay.
    """

    name: str
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff values must not be negative")


def is_retryable(exc: BaseException) -> bool:
    """Return whether ``exc`` is a transient failure worth retrying."""
    if isinstance(exc, AppError):
        return exc.retryable
    return False


def _log_before_sleep(policy: RetryPolicy) -> Callable[[RetryCallState], None]:
    def _callback(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        logger.warning(
            "retrying_operation",
            policy=policy.name,
            attempt=state.attempt_number,
            max_attempts=policy.max_attempts,
            error_type=type(exc).__name__ if exc else None,
        )

    return _callback


def build_async_retrying(
    policy: RetryPolicy,
    *,
    predicate: Callable[[BaseException], bool] | None = None,
) -> AsyncRetrying:
    """Return an asynchronous tenacity controller for ``policy``."""
    return AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_exponential_jitter(
            initial=policy.initial_backoff_seconds,
            max=policy.max_backoff_seconds,
            jitter=policy.jitter_seconds,
        ),
        retry=retry_if_exception(predicate or is_retryable),
        before_sleep=_log_before_sleep(policy),
        reraise=True,
    )
