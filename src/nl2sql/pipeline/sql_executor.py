"""Query execution.

Everything here exists to make a read only query bounded and honest:

* parameters are bound by the driver. The statement text is built from a
  syntax tree and never by concatenating a value into it
* the timeout is set on the driver, so the database stops the work. A timeout
  enforced only in Python would return control while the server kept running
  the query
* one row more than the ceiling is fetched, which is how truncation is
  detected without a second count query, and a byte budget stops a query
  returning a few enormous rows
* the connection is rolled back on the way out. The statement is read only, so
  there is nothing to commit, and an open transaction on a pooled connection
  is a leak waiting to happen
* only transient faults are retried, and only because the statement cannot
  change anything. Azure SQL fails over often enough that a service without
  this looks unreliable
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import sqlglot
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import exp
from sqlglot.errors import SqlglotError

from nl2sql.config.settings import (
    CostEstimationSettings,
    ExecutionSettings,
    LimitsSettings,
    TenancySettings,
)
from nl2sql.core.exceptions import (
    DatabasePermissionError,
    ExecutionError,
    InvalidQueryReferenceError,
    QueryTimeoutError,
    TransientDatabaseError,
)
from nl2sql.core.retry import RetryPolicy, build_async_retrying
from nl2sql.db.params import to_driver_sql
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import ColumnMeta, ExecutionResult

logger = get_logger(__name__)

#: How a Python value maps onto the coarse type reported to API callers.
_TYPE_LABELS: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (int, "number"),
    (float, "number"),
    (str, "string"),
    (bytes, "binary"),
)


class SQLExecutor:
    """Runs validated SQL against the read only engine."""

    def __init__(
        self,
        engine: Engine,
        *,
        settings: ExecutionSettings,
        limits: LimitsSettings,
        tenancy: TenancySettings,
        cost: CostEstimationSettings | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        self._limits = limits
        self._tenancy = tenancy
        self._cost = cost
        self._retry = RetryPolicy(
            name="database",
            max_attempts=settings.retry.max_attempts,
            initial_backoff_seconds=settings.retry.initial_backoff_seconds,
            max_backoff_seconds=settings.retry.max_backoff_seconds,
            jitter_seconds=settings.retry.jitter_seconds,
        )

    @property
    def dialect(self) -> str:
        """Return the SQLAlchemy dialect name."""
        return self._engine.dialect.name

    @property
    def paramstyle(self) -> str:
        """Return the parameter style the driver expects."""
        return str(self._engine.dialect.paramstyle)

    # -- public surface ----------------------------------------------------
    async def execute(
        self,
        sql: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        max_rows: int | None = None,
        timeout_seconds: float | None = None,
        tenant_id: str | None = None,
    ) -> ExecutionResult:
        """Run a validated statement and return the rows it produced."""
        limit = max_rows if max_rows is not None else self._limits.max_rows
        timeout = timeout_seconds or self._limits.max_execution_seconds

        async for attempt in build_async_retrying(self._retry):
            with attempt:
                return await asyncio.to_thread(
                    self._execute_sync,
                    sql,
                    dict(parameters or {}),
                    limit,
                    timeout,
                    tenant_id,
                )
        raise AssertionError("unreachable: tenacity either returns or reraises")

    async def dry_run(
        self,
        sql: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        """Bind a statement without reading rows, to prove it resolves.

        The database checks every name, type and aggregate in the statement
        while returning nothing, which catches a plausible looking query that
        would fail at the point a user is waiting for the answer.
        """
        probe = self._as_empty_probe(sql)
        if probe is None:
            return False
        await asyncio.to_thread(
            self._execute_sync,
            probe,
            dict(parameters or {}),
            0,
            self._settings.dry_run_timeout_seconds,
            tenant_id,
        )
        return True

    async def estimate_cost(
        self, sql: str, parameters: Mapping[str, Any] | None = None
    ) -> float | None:
        """Return the optimiser's estimated cost, where the database offers one."""
        if self._cost is None or not self._cost.enabled or self.dialect != "mssql":
            return None
        try:
            return await asyncio.to_thread(self._estimate_sync, sql, dict(parameters or {}))
        except (SQLAlchemyError, ExecutionError) as exc:
            # Estimation needs SHOWPLAN permission, which a read only login may
            # not have. That is a reason to skip the check, not to fail.
            logger.warning("cost_estimation_unavailable", error_type=type(exc).__name__)
            return None

    async def ping(self) -> bool:
        """Return whether the database answers."""
        try:
            await asyncio.to_thread(self._ping_sync)
        except SQLAlchemyError as exc:
            logger.warning("database_ping_failed", error_type=type(exc).__name__)
            return False
        return True

    # -- synchronous work --------------------------------------------------
    def _ping_sync(self) -> None:
        with self._engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1").fetchone()

    def _execute_sync(
        self,
        sql: str,
        parameters: dict[str, Any],
        max_rows: int,
        timeout: float,
        tenant_id: str | None,
    ) -> ExecutionResult:
        """Run one statement. Executed in a worker thread."""
        started = time.perf_counter()
        driver_sql, bound = to_driver_sql(sql, parameters, self.paramstyle)

        with self._engine.connect() as connection:
            reset = self._apply_timeout(connection, timeout)
            try:
                self._set_session_context(connection, tenant_id)
                try:
                    cursor = (
                        connection.exec_driver_sql(driver_sql, bound)
                        if bound
                        else connection.exec_driver_sql(driver_sql)
                    )
                except Exception as exc:
                    raise self._classify(exc) from exc

                columns = self._columns(cursor)
                rows, truncated, size_limited = self._fetch(cursor, max_rows)
                cursor.close()
            finally:
                reset()
                connection.rollback()

        elapsed_ms = (time.perf_counter() - started) * 1000
        result = ExecutionResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            size_limited=size_limited,
            execution_ms=elapsed_ms,
        )
        logger.info(
            "query_executed",
            row_count=result.row_count,
            truncated=truncated,
            size_limited=size_limited,
            execution_ms=round(elapsed_ms, 1),
        )
        return result

    def _fetch(self, cursor: Any, max_rows: int) -> tuple[list[tuple[Any, ...]], bool, bool]:
        """Read rows up to the row and byte ceilings."""
        rows: list[tuple[Any, ...]] = []
        truncated = False
        size_limited = False
        total_bytes = 0
        batch_size = max(1, self._settings.fetch_batch_size)

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            for raw in batch:
                if len(rows) >= max_rows:
                    truncated = True
                    break
                row = tuple(raw)
                total_bytes += self._row_bytes(row)
                if total_bytes > self._limits.max_result_bytes:
                    truncated = True
                    size_limited = True
                    break
                rows.append(row)
            if truncated:
                break
        return rows, truncated, size_limited

    @staticmethod
    def _row_bytes(row: tuple[Any, ...]) -> int:
        """Estimate the size of one row, cheaply."""
        total = 0
        for value in row:
            if value is None:
                continue
            if isinstance(value, str):
                total += len(value) * 2
            elif isinstance(value, bytes):
                total += len(value)
            else:
                total += 16
        return total

    @staticmethod
    def _columns(cursor: Any) -> tuple[ColumnMeta, ...]:
        """Describe the result columns from the driver's own metadata."""
        description = getattr(cursor, "cursor", None)
        description = getattr(description, "description", None)
        if not description:
            names = list(cursor.keys())
            return tuple(ColumnMeta(name=str(name)) for name in names)
        columns: list[ColumnMeta] = []
        for entry in description:
            name = str(entry[0])
            type_code = entry[1] if len(entry) > 1 else None
            columns.append(ColumnMeta(name=name, type_name=_label_for(type_code)))
        return tuple(columns)

    def _apply_timeout(self, connection: Any, timeout: float) -> Any:
        """Set a driver level timeout and return a callable that undoes it."""
        raw = getattr(connection.connection, "driver_connection", None)
        if raw is None:
            return lambda: None

        if self.dialect == "mssql" and hasattr(raw, "timeout"):
            previous = getattr(raw, "timeout", 0)
            raw.timeout = int(max(1, round(timeout)))

            def reset_mssql() -> None:
                raw.timeout = previous

            return reset_mssql

        if hasattr(raw, "set_progress_handler"):
            # SQLite has no query timeout, but it calls a handler every so many
            # virtual machine instructions and aborts when the handler returns
            # a non zero value.
            deadline = time.monotonic() + timeout

            def handler() -> int:
                return 1 if time.monotonic() > deadline else 0

            raw.set_progress_handler(handler, 10000)

            def reset_sqlite() -> None:
                raw.set_progress_handler(None, 0)

            return reset_sqlite

        return lambda: None

    def _set_session_context(self, connection: Any, tenant_id: str | None) -> None:
        """Publish the tenant to the session, for row level security to read."""
        key = self._tenancy.session_context_key
        if not (self._tenancy.enabled and key and tenant_id and self.dialect == "mssql"):
            return
        connection.exec_driver_sql("EXEC sp_set_session_context ?, ?", (key, tenant_id))

    def _estimate_sync(self, sql: str, parameters: dict[str, Any]) -> float | None:
        """Ask SQL Server for the estimated plan cost without running the query."""
        from defusedxml import ElementTree

        driver_sql, bound = to_driver_sql(sql, parameters, self.paramstyle)
        with self._engine.connect() as connection:
            reset = self._apply_timeout(
                connection, self._cost.timeout_seconds if self._cost else 10.0
            )
            try:
                connection.exec_driver_sql("SET SHOWPLAN_XML ON")
                try:
                    cursor = (
                        connection.exec_driver_sql(driver_sql, bound)
                        if bound
                        else connection.exec_driver_sql(driver_sql)
                    )
                    rows = cursor.fetchall()
                finally:
                    connection.exec_driver_sql("SET SHOWPLAN_XML OFF")
            finally:
                reset()
                connection.rollback()

        if not rows or not rows[0] or not rows[0][0]:
            return None
        root = ElementTree.fromstring(str(rows[0][0]))
        costs = [
            float(element.get("StatementSubTreeCost", 0) or 0)
            for element in root.iter()
            if element.get("StatementSubTreeCost")
        ]
        return max(costs) if costs else None

    # -- helpers -----------------------------------------------------------
    def _as_empty_probe(self, sql: str) -> str | None:
        """Wrap a validated statement so it binds every name but returns nothing."""
        try:
            parsed = sqlglot.parse_one(sql, read=self.sqlglot_dialect)
            if not isinstance(parsed, exp.Query):
                return None
            probe = (
                exp.select(exp.Star())
                .from_(parsed.subquery(alias="nl2sql_dry"))
                .where(exp.EQ(this=exp.Literal.number(1), expression=exp.Literal.number(0)))
            )
            return str(probe.sql(dialect=self.sqlglot_dialect))
        except SqlglotError as exc:
            logger.warning("dry_run_wrap_failed", error_type=type(exc).__name__)
            return None

    @property
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect matching this engine."""
        from nl2sql.db.engine import sqlglot_dialect

        return sqlglot_dialect(self.dialect)

    def _classify(self, exc: Exception) -> ExecutionError:
        """Turn a driver failure into a typed error, deciding retryability."""
        text = str(exc).lower()

        def matches(markers: list[str]) -> bool:
            return any(marker.lower() in text for marker in markers)

        if matches(self._settings.timeout_error_markers):
            return QueryTimeoutError(
                "The query exceeded the allowed execution time.",
                details={"error": str(exc)[:300]},
            )
        if matches(self._settings.transient_error_markers):
            return TransientDatabaseError(
                "The database reported a transient fault.",
                details={"error": str(exc)[:300]},
            )
        if matches(self._settings.permission_error_markers):
            return DatabasePermissionError(
                "The database refused the query for lack of permission.",
                details={"error": str(exc)[:300]},
            )
        if matches(self._settings.invalid_reference_markers):
            return InvalidQueryReferenceError(
                "The database does not recognise something the query referenced.",
                details={"error": str(exc)[:300]},
            )
        return ExecutionError(
            f"The query failed: {type(exc).__name__}",
            details={"error": str(exc)[:300]},
        )


def _label_for(type_code: Any) -> str:
    """Return a coarse type label for a driver type code."""
    if type_code is None:
        return "unknown"
    if isinstance(type_code, type):
        for python_type, label in _TYPE_LABELS:
            if issubclass(type_code, python_type):
                return label
        name = type_code.__name__.lower()
    else:
        name = str(type_code).lower()
    for marker, label in (
        ("date", "datetime"),
        ("time", "datetime"),
        ("decimal", "number"),
        ("int", "number"),
        ("float", "number"),
        ("double", "number"),
        ("num", "number"),
        ("char", "string"),
        ("text", "string"),
        ("str", "string"),
        ("bool", "boolean"),
        ("byte", "binary"),
        ("binary", "binary"),
    ):
        if marker in name:
            return label
    return "unknown"
