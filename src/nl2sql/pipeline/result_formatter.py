"""Result formatting.

Database drivers return values a JSON encoder cannot serialise: decimals,
dates, times, UUIDs and byte strings. This converts them once, in one place,
so neither the API layer nor the answer prompt has to think about it.

Decimals are the interesting case. Rendering them as floats is convenient for
a caller doing arithmetic and wrong for money, where the decimal places are
part of the value, so which one happens is configuration.
"""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import uuid
from typing import Any

from nl2sql.config.settings import AnswerSettings
from nl2sql.pipeline.models import ExecutionResult, FormattedResult


class ResultFormatter:
    """Converts a result set into JSON safe values."""

    def __init__(self, settings: AnswerSettings) -> None:
        self._decimal_as = settings.decimal_as

    def format(self, result: ExecutionResult) -> FormattedResult:
        """Return the result set with JSON safe values and column metadata."""
        columns = [
            {"name": column.name, "type": self._column_type(result, index, column.type_name)}
            for index, column in enumerate(result.columns)
        ]
        rows = [[self.convert(value) for value in row] for row in result.rows]
        return FormattedResult(columns=columns, rows=rows)

    def convert(self, value: Any) -> Any:
        """Convert one value into something JSON can carry."""
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, decimal.Decimal):
            return float(value) if self._decimal_as == "float" else str(value)
        if isinstance(value, dt.datetime | dt.date | dt.time):
            return value.isoformat()
        if isinstance(value, dt.timedelta):
            return value.total_seconds()
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, bytes | bytearray | memoryview):
            return base64.b64encode(bytes(value)).decode("ascii")
        return str(value)

    def _column_type(self, result: ExecutionResult, index: int, declared: str) -> str:
        """Return the declared type, or infer one from the first value present."""
        if declared != "unknown":
            return declared
        for row in result.rows:
            if index < len(row) and row[index] is not None:
                return self._infer(row[index])
        return "unknown"

    @staticmethod
    def _infer(value: Any) -> str:
        """Infer a coarse type label from a value."""
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int | float | decimal.Decimal):
            return "number"
        if isinstance(value, dt.datetime | dt.date | dt.time):
            return "datetime"
        if isinstance(value, bytes | bytearray | memoryview):
            return "binary"
        return "string"
