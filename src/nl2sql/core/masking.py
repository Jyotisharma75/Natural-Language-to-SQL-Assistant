"""Masking of sensitive values before they reach logs, audit rows or storage.

Two mechanisms work together:

* key based masking replaces the whole value of any mapping key whose name
  matches a sensitive key, such as ``password`` or ``api_key``
* pattern based masking rewrites substrings of free text that look like
  credentials, connection string secrets, bearer tokens or email addresses

Both lists come from configuration. The defaults here are a floor, not the
whole policy.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

MASK = "***"

DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "api_key",
        "apikey",
        "x-api-key",
        "authorization",
        "access_token",
        "refresh_token",
        "token",
        "client_secret",
        "connection_string",
        "odbc_connect",
        "account_key",
    }
)

#: Patterns applied to free text. Each replaces only the secret portion, so a
#: masked connection string still shows which server it pointed at.
DEFAULT_PATTERNS: tuple[str, ...] = (
    r"(?i)((?:password|pwd|secret|accountkey|sharedaccesskey)\s*=\s*)([^;\s]+)",
    r"(?i)(bearer\s+)([A-Za-z0-9\-._~+/]+=*)",
    r"(?i)((?:api[_-]?key|x-api-key)\s*[:=]\s*)([A-Za-z0-9\-._~+/]{8,})",
    r"()([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    r"()(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})",
)


def hash_identifier(value: str | None, *, length: int = 16) -> str | None:
    """Return a short stable hash of an identifier, for correlation without disclosure."""
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


class Masker:
    """Mask sensitive keys and patterns in strings and nested structures."""

    def __init__(
        self,
        *,
        sensitive_keys: Iterable[str] = (),
        patterns: Iterable[str] = (),
        include_defaults: bool = True,
    ) -> None:
        keys = {key.casefold() for key in sensitive_keys}
        if include_defaults:
            keys |= DEFAULT_SENSITIVE_KEYS
        self._keys = frozenset(keys)
        compiled_sources = list(patterns)
        if include_defaults:
            compiled_sources = [*DEFAULT_PATTERNS, *compiled_sources]
        self._patterns = [re.compile(source) for source in compiled_sources]

    def is_sensitive_key(self, key: str) -> bool:
        """Return whether a mapping key names a sensitive value."""
        folded = key.casefold()
        return folded in self._keys or any(k in folded for k in self._keys if len(k) > 4)

    def mask_text(self, text: str) -> str:
        """Mask credential shaped substrings in free text."""
        masked = text
        for pattern in self._patterns:
            if pattern.groups >= 2:
                masked = pattern.sub(lambda m: f"{m.group(1)}{MASK}", masked)
            else:
                masked = pattern.sub(MASK, masked)
        return masked

    def mask(self, value: Any) -> Any:
        """Return ``value`` with sensitive content masked, recursing into containers."""
        if isinstance(value, str):
            return self.mask_text(value)
        if isinstance(value, Mapping):
            return {
                key: (MASK if isinstance(key, str) and self.is_sensitive_key(key) else self.mask(v))
                for key, v in value.items()
            }
        if isinstance(value, list | tuple):
            return type(value)(self.mask(item) for item in value)
        return value


def mask_sql_literals(sql: str, *, dialect: str | None = None) -> str:
    """Replace the literals in ``sql`` with placeholders.

    Used when logging or storing generated SQL, so a value copied from a
    question, such as a customer name or an account number, does not land in a
    log aggregator.

    Row limits and offsets keep their values. They disclose nothing, and a
    logged statement whose limit reads as zero would mislead whoever is
    reading the log to work out what actually ran. Falls back to a regular
    expression when the statement cannot be parsed.
    """
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except SqlglotError:
        return re.sub(r"'(?:[^']|'')*'", "'?'", sql)

    def _is_row_count(node: exp.Expression) -> bool:
        parent = node.parent
        return isinstance(parent, exp.Limit | exp.Offset | exp.Fetch)

    def _replace(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Literal) and not _is_row_count(node):
            return exp.Literal.string("?") if node.is_string else exp.Literal.number(0)
        return node

    return tree.transform(_replace).sql(dialect=dialect)
