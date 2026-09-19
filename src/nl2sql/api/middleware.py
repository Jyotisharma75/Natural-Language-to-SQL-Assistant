"""Request middleware.

Written as plain ASGI rather than as a Starlette ``BaseHTTPMiddleware`` so
that the context variables it sets are visible to the endpoint, to every
thread the endpoint hands work to, and to the logger, without the extra task
hop that class introduces.

It does three things: gives every request an identifier, rejects a body that
is too large before it is read, and logs the outcome with its duration.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nl2sql.core.context import reset_request_id, sanitize_inbound_id, set_request_id
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = b"x-request-id"


class RequestContextMiddleware:
    """Binds a request identifier, enforces the body limit and logs the result."""

    def __init__(self, app: ASGIApp, *, max_request_bytes: int = 32_768) -> None:
        self._app = app
        self._max_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        inbound = headers.get(REQUEST_ID_HEADER)
        request_id = sanitize_inbound_id(inbound.decode("latin-1") if inbound else None)
        token = set_request_id(request_id)
        started = time.perf_counter()
        status_holder: dict[str, int] = {}

        content_length = headers.get(b"content-length")
        if content_length is not None and int(content_length or 0) > self._max_bytes:
            await _send_too_large(send, request_id, self._max_bytes)
            reset_request_id(token)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (REQUEST_ID_HEADER, request_id.encode("latin-1")),
                ]
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "request_completed",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status_holder.get("status"),
                duration_ms=round(duration_ms, 1),
            )
            reset_request_id(token)


async def _send_too_large(send: Send, request_id: str, limit: int) -> None:
    """Refuse an oversized body without reading it."""
    body = (
        '{"error":{"code":"request_too_large","message":'
        f'"The request body exceeds the {limit} byte limit.",'
        f'"request_id":"{request_id}"}}}}'
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (REQUEST_ID_HEADER, request_id.encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


AnyHandler = Callable[..., Awaitable[Any]]
