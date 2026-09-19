"""Parameter marker conversion.

Rewriting a query with sqlglot produces bare markers such as
``__NL2SQL_PTENANT__`` where a value belongs. This module turns those markers
into whatever the driver in use expects and returns the values in the matching
order. Doing it as a separate step, rather than formatting values into the SQL,
is what keeps the promise that no caller supplied value is ever concatenated
into a statement.

Markers are matched by a reserved pattern that model output is forbidden to
contain, so there is no way for generated text to be mistaken for one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from nl2sql.core.exceptions import ConfigurationError

MARKER_PATTERN = re.compile(r"__NL2SQL_P[A-Z0-9_]*?__")


def to_driver_sql(
    sql: str, parameters: Mapping[str, Any], paramstyle: str
) -> tuple[str, tuple[Any, ...] | dict[str, Any]]:
    """Return ``sql`` in the driver's parameter style with values in order.

    Args:
        sql: Statement containing reserved markers.
        parameters: Value for each marker name.
        paramstyle: The DBAPI paramstyle, as reported by the dialect.

    Raises:
        ConfigurationError: When the statement contains a marker with no value,
            or the driver uses a parameter style this function cannot emit.
    """
    ordered: list[Any] = []
    named: dict[str, Any] = {}
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        marker = match.group(0)
        if marker not in parameters:
            missing.append(marker)
            return marker
        value = parameters[marker]
        if paramstyle == "qmark":
            ordered.append(value)
            return "?"
        if paramstyle == "numeric":
            ordered.append(value)
            return f":{len(ordered)}"
        if paramstyle == "format":
            ordered.append(value)
            return "%s"
        key = marker.strip("_").lower()
        named[key] = value
        if paramstyle == "named":
            return f":{key}"
        if paramstyle == "pyformat":
            return f"%({key})s"
        raise ConfigurationError(f"Unsupported driver parameter style: {paramstyle}")

    if paramstyle in {"format", "pyformat"}:
        # Escape literal percent signs before markers are introduced, because
        # these styles read percent as the start of a placeholder.
        sql = sql.replace("%", "%%")

    converted = MARKER_PATTERN.sub(replace, sql)
    if missing:
        raise ConfigurationError(
            f"The statement contains parameter markers with no bound value: {sorted(set(missing))}"
        )
    if paramstyle in {"qmark", "numeric", "format"}:
        return converted, tuple(ordered)
    return converted, named


def has_markers(sql: str) -> bool:
    """Return whether ``sql`` still contains reserved parameter markers."""
    return bool(MARKER_PATTERN.search(sql))
