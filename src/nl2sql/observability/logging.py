"""Structured logging with identifier injection and masking.

Every event is a dictionary rendered as JSON in deployed environments. Three
processors always run: one attaches the request and query identifiers from
context variables, one masks sensitive keys and credential shaped text, and a
renderer produces the final line. Standard library logging, including
SQLAlchemy and uvicorn, is routed through the same processors.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from nl2sql.core.context import get_query_id, get_request_id
from nl2sql.core.masking import Masker

_NOISY_LOGGERS = (
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "httpx",
    "httpcore",
    "urllib3",
    "openai._base_client",
    "sqlalchemy.engine.Engine",
    "asyncio",
    "transformers",
)


def _add_identifiers(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    request_id = get_request_id()
    query_id = get_query_id()
    if request_id is not None:
        event_dict.setdefault("request_id", request_id)
    if query_id is not None:
        event_dict.setdefault("query_id", query_id)
    return event_dict


class MaskingProcessor:
    """Structlog processor that masks every value in an event."""

    def __init__(self, masker: Masker) -> None:
        self._masker = masker

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        for key in list(event_dict.keys()):
            value = event_dict[key]
            if self._masker.is_sensitive_key(key):
                event_dict[key] = "***"
            else:
                event_dict[key] = self._masker.mask(value)
        return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    log_format: str = "json",
    service_name: str = "nl2sql-assistant",
    environment: str = "development",
    masker: Masker | None = None,
) -> None:
    """Configure structlog and standard library logging. Safe to call repeatedly."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    active_masker = masker or Masker()

    def _service(
        logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event_dict.setdefault("service", service_name)
        event_dict.setdefault("env", environment)
        return event_dict

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_identifiers,
        _service,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
        MaskingProcessor(active_masker),
    ]
    renderer: Any
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=processors,
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)
    if numeric_level > logging.DEBUG:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger bound to ``name``."""
    logger = structlog.get_logger()
    return logger.bind(logger=name) if name else logger
