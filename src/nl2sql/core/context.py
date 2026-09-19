"""Request scoped identifiers carried in context variables.

Context variables follow an ``await`` chain and are copied into threads
started with :func:`asyncio.to_thread`, so a log line written deep inside the
executor still carries the request and query identifiers without threading
them through every signature.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("nl2sql_request_id", default=None)
_query_id: ContextVar[str | None] = ContextVar("nl2sql_query_id", default=None)

#: Identifiers accepted from an inbound header. Anything else is replaced, so a
#: caller cannot inject log formatting or oversized values through the header.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def new_id() -> str:
    """Return a new random identifier."""
    return uuid.uuid4().hex


def sanitize_inbound_id(value: str | None) -> str:
    """Return ``value`` when it is a safe identifier, otherwise a new one."""
    if value and _SAFE_ID.match(value):
        return value
    return new_id()


def get_request_id() -> str | None:
    """Return the current request identifier."""
    return _request_id.get()


def set_request_id(value: str) -> Token[str | None]:
    """Bind the request identifier for the current context."""
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the request identifier bound before ``token`` was issued."""
    _request_id.reset(token)


def get_query_id() -> str | None:
    """Return the current query identifier."""
    return _query_id.get()


def set_query_id(value: str) -> Token[str | None]:
    """Bind the query identifier for the current context."""
    return _query_id.set(value)


def reset_query_id(token: Token[str | None]) -> None:
    """Restore the query identifier bound before ``token`` was issued."""
    _query_id.reset(token)
