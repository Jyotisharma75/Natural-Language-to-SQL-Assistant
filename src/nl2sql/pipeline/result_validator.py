"""Result validation and masking.

Runs between execution and the answer. It is the last point at which a value
can be stopped, so masking happens here rather than anywhere earlier: the
validator decided which output columns derive from a masked column, and this
stage replaces those values before they reach the answer prompt, the API
response or a log.

It also turns the facts about the result set into warnings a reader needs:
that nothing matched, that the rows were cut short, or that a value was too
large to return whole.
"""

from __future__ import annotations

from typing import Any

from nl2sql.config.settings import LimitsSettings, SecuritySettings
from nl2sql.pipeline.models import ExecutionResult, ValidationReport


class ResultValidator:
    """Checks a result set and masks what must not be shown."""

    def __init__(self, limits: LimitsSettings, security: SecuritySettings) -> None:
        self._limits = limits
        self._security = security

    def check(
        self, result: ExecutionResult, report: ValidationReport
    ) -> tuple[ExecutionResult, list[str]]:
        """Return the result with masking applied, plus warnings for the caller."""
        warnings: list[str] = []
        masked_indexes = [
            index
            for index, column in enumerate(result.columns)
            if column.name in report.masked_outputs
        ]

        rows = result.rows
        if masked_indexes or self._limits.max_cell_chars:
            rows = [self._process_row(row, masked_indexes, warnings) for row in result.rows]

        if not rows:
            warnings.append("The query ran successfully but no rows matched.")
        if result.truncated:
            warnings.append(
                f"Only the first {len(rows)} rows are shown. Narrow the question to see "
                "a complete answer."
            )
        if result.size_limited:
            warnings.append(
                "The result was cut short because it exceeded the maximum response size."
            )
        if masked_indexes:
            warnings.append(
                f"{len(masked_indexes)} column(s) contain restricted values and are masked."
            )
        if report.output_columns and len(result.columns) != len(report.output_columns):
            warnings.append(
                "The columns returned differ from the columns the validated query declared."
            )

        return (
            ExecutionResult(
                columns=result.columns,
                rows=rows,
                truncated=result.truncated,
                size_limited=result.size_limited,
                execution_ms=result.execution_ms,
            ),
            warnings,
        )

    def _process_row(
        self, row: tuple[Any, ...], masked_indexes: list[int], warnings: list[str]
    ) -> tuple[Any, ...]:
        """Mask restricted values and cut oversized ones down to the cell limit."""
        values = list(row)
        for index in masked_indexes:
            if index < len(values) and values[index] is not None:
                values[index] = self._security.mask_token
        limit = self._limits.max_cell_chars
        for index, value in enumerate(values):
            if isinstance(value, str) and len(value) > limit:
                values[index] = value[:limit]
                message = f"Some values were longer than {limit} characters and were cut short."
                if message not in warnings:
                    warnings.append(message)
        return tuple(values)
