"""Execution, result validation and formatting."""

from __future__ import annotations

import datetime as dt
import decimal
from typing import Any

import pytest

from nl2sql.core.exceptions import (
    DatabasePermissionError,
    ExecutionError,
    InvalidQueryReferenceError,
    QueryTimeoutError,
    TransientDatabaseError,
)
from nl2sql.pipeline.models import ColumnMeta, ExecutionResult, ValidationReport
from nl2sql.pipeline.result_formatter import ResultFormatter
from nl2sql.pipeline.result_validator import ResultValidator
from nl2sql.pipeline.sql_executor import SQLExecutor

pytestmark = pytest.mark.unit


@pytest.fixture
def executor(container) -> SQLExecutor:
    """The executor wired to the sample database."""
    return container.executor


async def test_rows_are_returned_with_column_metadata(executor):
    """A query returns rows and the names of the columns."""
    result = await executor.execute(
        "SELECT facility_name, facility_id FROM facilities ORDER BY facility_id"
    )
    assert result.row_count == 3
    assert [column.name for column in result.columns] == ["facility_name", "facility_id"]
    assert result.truncated is False


async def test_parameters_are_bound(executor):
    """Values travel as parameters, not inside the statement."""
    result = await executor.execute(
        "SELECT facility_name FROM facilities WHERE facility_name = __NL2SQL_PF0__",
        {"__NL2SQL_PF0__": "Lyon Plant"},
    )
    assert [row[0] for row in result.rows] == ["Lyon Plant"]


async def test_truncation_is_detected(executor):
    """Fetching one more row than the ceiling is how truncation is known."""
    result = await executor.execute("SELECT facility_name FROM facilities", max_rows=2)
    assert result.row_count == 2
    assert result.truncated is True


async def test_byte_budget_stops_a_large_result(build_container):
    """A result that exceeds the byte budget is cut short and says so."""
    container = build_container(limits={"max_result_bytes": 32})
    result = await container.executor.execute("SELECT facility_name FROM facilities")
    assert result.size_limited is True
    assert result.truncated is True


async def test_unknown_object_is_reported_as_an_invalid_reference(executor):
    """A name the database does not know produces a typed error."""
    with pytest.raises(InvalidQueryReferenceError):
        await executor.execute("SELECT a FROM no_such_table")


async def test_dry_run_binds_without_returning_rows(executor):
    """The dry run proves a query resolves and reads nothing."""
    assert await executor.dry_run("SELECT facility_name FROM facilities") is True
    with pytest.raises(InvalidQueryReferenceError):
        await executor.dry_run("SELECT missing_column FROM facilities")


async def test_ping_reports_the_database_is_reachable(executor):
    """Readiness depends on this answering."""
    assert await executor.ping() is True


async def test_cost_estimation_is_skipped_where_unsupported(executor):
    """SQLite offers no plan cost, so estimation returns nothing rather than failing."""
    assert await executor.estimate_cost("SELECT facility_name FROM facilities") is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HYT00 timeout expired", QueryTimeoutError),
        ("[08S01] transport-level error", TransientDatabaseError),
        ("Database '40613' is not currently available", TransientDatabaseError),
        ("The SELECT permission was denied on the object", DatabasePermissionError),
        ("[42S02] Invalid object name 'dbo.x'", InvalidQueryReferenceError),
        ("something else entirely", ExecutionError),
    ],
)
def test_driver_errors_are_classified(executor, message, expected):
    """Only genuinely transient faults are marked retryable."""
    classified = executor._classify(RuntimeError(message))
    assert isinstance(classified, expected)
    assert classified.retryable is (expected is TransientDatabaseError)


async def test_transient_failures_are_retried(build_container, monkeypatch):
    """A transient fault is retried, because the statement is read only."""
    container = build_container(
        execution={"retry": {"max_attempts": 3, "initial_backoff_seconds": 0, "jitter_seconds": 0}}
    )
    executor = container.executor
    attempts = {"count": 0}
    original = executor._execute_sync

    def flaky(*args: Any, **kwargs: Any) -> Any:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TransientDatabaseError("40613 is not currently available")
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "_execute_sync", flaky)
    result = await executor.execute("SELECT facility_name FROM facilities")
    assert attempts["count"] == 2
    assert result.row_count == 3


async def test_validation_failures_are_not_retried(build_container, monkeypatch):
    """A failure that cannot change on a retry is raised immediately."""
    container = build_container(
        execution={"retry": {"max_attempts": 3, "initial_backoff_seconds": 0}}
    )
    executor = container.executor
    attempts = {"count": 0}

    def always_invalid(*args: Any, **kwargs: Any) -> Any:
        attempts["count"] += 1
        raise InvalidQueryReferenceError("no such column")

    monkeypatch.setattr(executor, "_execute_sync", always_invalid)
    with pytest.raises(InvalidQueryReferenceError):
        await executor.execute("SELECT x FROM facilities")
    assert attempts["count"] == 1


# -- result validation ------------------------------------------------------
def _result(rows: list[tuple[Any, ...]], names: tuple[str, ...]) -> ExecutionResult:
    return ExecutionResult(columns=tuple(ColumnMeta(name=name) for name in names), rows=rows)


def test_empty_results_produce_a_warning(settings):
    """No rows is an answer, and the caller is told plainly."""
    validator = ResultValidator(settings.limits, settings.security)
    _, warnings = validator.check(_result([], ("facility_name",)), ValidationReport())
    assert any("no rows matched" in warning for warning in warnings)


def test_masked_columns_are_replaced(settings):
    """A masked output column never carries its real value."""
    validator = ResultValidator(settings.limits, settings.security)
    report = ValidationReport(masked_outputs=frozenset({"contact_email"}))
    checked, warnings = validator.check(
        _result([("Rotterdam", "ops@example.invalid")], ("facility_name", "contact_email")),
        report,
    )
    assert checked.rows[0][1] == settings.security.mask_token
    assert checked.rows[0][0] == "Rotterdam"
    assert any("masked" in warning for warning in warnings)


def test_oversized_cells_are_trimmed(build_container, settings):
    """A very large value is cut to the configured ceiling."""
    limits = settings.limits.model_copy(update={"max_cell_chars": 10})
    validator = ResultValidator(limits, settings.security)
    checked, warnings = validator.check(_result([("x" * 500,)], ("notes",)), ValidationReport())
    assert len(checked.rows[0][0]) == 10
    assert any("cut short" in warning for warning in warnings)


def test_truncation_is_reported(settings):
    """Truncation is surfaced so an answer is not read as complete."""
    validator = ResultValidator(settings.limits, settings.security)
    result = _result([("a",)], ("facility_name",))
    result.truncated = True
    _, warnings = validator.check(result, ValidationReport())
    assert any("Only the first" in warning for warning in warnings)


# -- formatting -------------------------------------------------------------
def test_values_are_converted_for_json(settings):
    """Decimals, dates and bytes become values JSON can carry."""
    formatter = ResultFormatter(settings.answer)
    result = _result(
        [
            (
                decimal.Decimal("10.5"),
                dt.date(2026, 2, 1),
                dt.datetime(2026, 2, 1, 9, 30),
                b"ab",
                None,
            )
        ],
        ("amount", "day", "moment", "blob", "missing"),
    )
    formatted = formatter.format(result)
    assert formatted.rows[0][0] == 10.5
    assert formatted.rows[0][1] == "2026-02-01"
    assert formatted.rows[0][2].startswith("2026-02-01T09:30")
    assert isinstance(formatted.rows[0][3], str)
    assert formatted.rows[0][4] is None


def test_decimals_can_be_kept_exact(settings):
    """Money keeps its decimal places when configuration asks for strings."""
    formatter = ResultFormatter(settings.answer.model_copy(update={"decimal_as": "string"}))
    formatted = formatter.format(_result([(decimal.Decimal("10.50"),)], ("amount",)))
    assert formatted.rows[0][0] == "10.50"


def test_column_types_are_inferred_when_the_driver_does_not_say(settings):
    """A driver that reports no type does not leave the caller guessing."""
    formatter = ResultFormatter(settings.answer)
    formatted = formatter.format(_result([("text", 3)], ("name", "count")))
    assert formatted.columns[0]["type"] == "string"
    assert formatted.columns[1]["type"] == "number"
