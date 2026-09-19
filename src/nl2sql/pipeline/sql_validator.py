"""SQL validation.

The guard has already decided the statement is a permitted kind and contains
nothing dangerous in structure. This stage decides whether it may touch the
data it asks for, and rewrites it so that what finally executes is bounded.

It does six things:

* resolves every table and every column against the filtered catalogue, so a
  table or column the policy hides is rejected even if the model knew its name
  from somewhere. Aliases, CTEs and derived tables are resolved through scopes
  rather than by string matching, so a column hidden behind two levels of
  subquery is still checked
* refuses SELECT star unless configuration allows it, because a star silently
  returns whatever columns exist, including ones added to the table later
* applies the cost ceilings: join count, cartesian joins, subquery depth and
  total tables
* adds the tenant predicate, as a bound parameter
* applies caller supplied filters by wrapping the query, never by editing the
  text the model produced
* clamps the row limit, and produces a second rendering that fetches one extra
  row so truncation can be detected honestly

Two statements come out. ``display_sql`` is what the caller is shown, and
``execution_sql`` is what runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope

from nl2sql.config.settings import LimitsSettings, SecuritySettings
from nl2sql.metadata.models import SchemaCatalog, TableInfo
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import QueryFilter, ValidationIssue, ValidationReport
from nl2sql.security.policy import AccessPolicy
from nl2sql.security.sql_guard import SQLGuard
from nl2sql.security.tenancy import TenantScoper

logger = get_logger(__name__)

#: Alias given to the generated query when it is wrapped to apply filters.
RESULT_ALIAS = "nl2sql_result"

#: Prefix of the markers that carry filter values.
FILTER_MARKER = "__NL2SQL_PF{index}__"

_OPERATORS: dict[str, type[exp.Binary]] = {
    "eq": exp.EQ,
    "ne": exp.NEQ,
    "gt": exp.GT,
    "gte": exp.GTE,
    "lt": exp.LT,
    "lte": exp.LTE,
}


def _named_selects(expression: exp.Expr) -> tuple[str, ...]:
    """Return the output column names of anything that returns rows.

    Only a query has output columns. Anything else, including a construct this
    version of the parser models differently, yields nothing rather than
    raising.
    """
    if not isinstance(expression, exp.Query):
        return ()
    try:
        return tuple(expression.named_selects)
    except SqlglotError:
        return ()


class SQLValidator:
    """Checks generated SQL against the policy and rewrites it to be safe."""

    def __init__(
        self,
        *,
        policy: AccessPolicy,
        guard: SQLGuard,
        scoper: TenantScoper,
        limits: LimitsSettings,
        security: SecuritySettings,
    ) -> None:
        self._policy = policy
        self._guard = guard
        self._scoper = scoper
        self._limits = limits
        self._security = security

    def validate(
        self,
        sql: str,
        catalog: SchemaCatalog,
        *,
        dialect: str,
        tenant_id: str | None = None,
        filters: Sequence[QueryFilter] = (),
        declared_tables: Sequence[str] = (),
    ) -> ValidationReport:
        """Validate one candidate query and return what may be executed."""
        report = ValidationReport()

        guarded = self._guard.check(sql, dialect=dialect)
        report.statement_kind = guarded.statement_kind
        report.issues.extend(
            ValidationIssue(code=issue.code, message=issue.message, severity=issue.severity)
            for issue in guarded.issues
        )
        if not guarded.ok or guarded.expression is None:
            report.valid = False
            return report

        expression = guarded.expression
        bindings: dict[int, tuple[str, str, str]] = {}
        tables = self._resolve(expression, catalog, report, bindings)
        self._check_stars(expression, report)
        self._check_cost(expression, report, len(tables))
        self._check_declared(tables, declared_tables, report)

        report.tables = tuple(sorted(tables))
        # Bindings also hold entries for columns that resolved to a CTE or a
        # derived table rather than to a base table. Those carry no schema and
        # are not reported as columns read from the database.
        report.columns = tuple(
            sorted(
                f"{schema}.{table}.{column}"
                for schema, table, column in bindings.values()
                if schema and table
            )
        )

        if report.errors:
            report.valid = False
            return report

        parameters: dict[str, Any] = {}
        scoped = self._scoper.apply(expression, catalog, tenant_id=tenant_id)
        working = scoped.expression
        parameters.update(scoped.parameters)
        report.scoped_tables = scoped.scoped_tables

        output_columns = self._output_columns(expression)
        report.output_columns = output_columns
        report.masked_outputs = self._masked_outputs(expression, bindings, catalog)

        if filters:
            working = self._apply_filters(
                working, filters, output_columns, parameters, report, dialect
            )
            if report.errors:
                report.valid = False
                return report

        display, execution, applied = self._apply_row_limit(working)
        report.row_limit_applied = applied
        report.parameters = parameters

        try:
            report.display_sql = display.sql(dialect=dialect, comments=False)
            report.execution_sql = execution.sql(dialect=dialect, comments=False)
        except SqlglotError as exc:
            report.issues.append(
                ValidationIssue("render_failed", f"The query could not be rewritten: {exc}")
            )
            report.valid = False
            return report

        report.valid = not report.errors
        return report

    # -- table and column resolution ---------------------------------------
    def _resolve(
        self,
        expression: exp.Expr,
        catalog: SchemaCatalog,
        report: ValidationReport,
        bindings: dict[int, tuple[str, str, str]],
    ) -> set[str]:
        """Resolve every table and column, recording problems on the report."""
        tables: set[str] = set()
        try:
            scopes = traverse_scope(expression)
        except (SqlglotError, KeyError, ValueError, RecursionError) as exc:
            report.issues.append(
                ValidationIssue(
                    "unresolvable_query",
                    "The structure of the query could not be analysed, so it cannot be "
                    "checked against the allowed tables.",
                )
            )
            logger.warning("scope_traversal_failed", error_type=type(exc).__name__)
            return tables

        # A column belonging to a subquery is listed both in that subquery's
        # scope and in the scope enclosing it, so a per scope verdict would
        # report a perfectly good column as unknown. Outcomes are collected per
        # column node first and a problem is only reported when the column
        # resolved in no scope at all.
        failures: dict[int, ValidationIssue] = {}

        for scope in scopes:
            resolved_sources: dict[str, TableInfo] = {}
            derived_sources: dict[str, Scope] = {}

            for alias, source in scope.sources.items():
                if isinstance(source, exp.Table):
                    table = self._resolve_table(source, catalog, report)
                    if table is not None:
                        resolved_sources[alias.casefold()] = table
                        tables.add(table.qualified_name)
                elif isinstance(source, Scope):
                    derived_sources[alias.casefold()] = source

            self._resolve_columns(scope, resolved_sources, derived_sources, bindings, failures)

        for node_id, issue in failures.items():
            if node_id not in bindings:
                report.issues.append(issue)

        return tables

    def _resolve_table(
        self, source: exp.Table, catalog: SchemaCatalog, report: ValidationReport
    ) -> TableInfo | None:
        """Resolve one table reference, or record why it is refused."""
        name = source.name
        if not name:
            return None
        reference = f"{source.db}.{name}" if source.db else name
        if not source.db and catalog.is_ambiguous(name):
            report.issues.append(
                ValidationIssue(
                    "ambiguous_table",
                    f"The table {name} exists in more than one schema. Qualify it with its schema.",
                )
            )
            return None
        table = catalog.resolve(reference, default_schema=catalog.default_schema)
        if table is None:
            # The message deliberately does not say whether the table exists
            # and is blocked, or does not exist at all.
            report.issues.append(
                ValidationIssue(
                    "table_not_allowed",
                    f"The table {reference} is not available to this service.",
                )
            )
        return table

    def _resolve_columns(
        self,
        scope: Scope,
        tables: dict[str, TableInfo],
        derived: dict[str, Scope],
        bindings: dict[int, tuple[str, str, str]],
        failures: dict[int, ValidationIssue],
    ) -> None:
        """Bind every column in one scope to a source, or note why it did not bind."""
        output_names = {name.casefold() for name in _named_selects(scope.expression)}

        for column in scope.columns:
            name = column.name
            if not name or isinstance(column.this, exp.Star):
                continue
            node_id = id(column)
            qualifier = (column.table or "").casefold()

            if qualifier:
                table = tables.get(qualifier)
                if table is not None:
                    if table.has_column(name):
                        bindings[node_id] = (table.schema_name, table.name, name)
                    else:
                        failures.setdefault(
                            node_id,
                            ValidationIssue(
                                "column_not_allowed",
                                f"The column {name} is not available on {table.qualified_name}.",
                            ),
                        )
                elif qualifier in derived:
                    if self._derived_has(derived[qualifier], name):
                        bindings.setdefault(node_id, ("", qualifier, name))
                    else:
                        failures.setdefault(
                            node_id,
                            ValidationIssue(
                                "column_not_allowed",
                                f"The column {name} is not produced by {qualifier}.",
                            ),
                        )
                # A qualifier matching neither a table nor a derived source
                # belongs to a table already refused above, so there is nothing
                # useful to add here.
                continue

            matches = [table for table in tables.values() if table.has_column(name)]
            if len(matches) == 1:
                table = matches[0]
                bindings[node_id] = (table.schema_name, table.name, name)
                continue
            if len(matches) > 1:
                failures.setdefault(
                    node_id,
                    ValidationIssue(
                        "ambiguous_column",
                        f"The column {name} exists on more than one table in this query. "
                        "Qualify it with a table alias.",
                    ),
                )
                continue
            # A bare name may legitimately refer to an output alias, but only
            # from ORDER BY, GROUP BY or HAVING. Accepting it anywhere would
            # let a projection vouch for itself: SELECT password_hash would
            # match its own output name and bypass the column allowlist.
            references_alias = (
                name.casefold() in output_names
                and column.find_ancestor(exp.Order, exp.Group, exp.Having) is not None
            )
            if references_alias or any(
                self._derived_has(source, name) for source in derived.values()
            ):
                bindings.setdefault(node_id, ("", "", name))
                continue
            failures.setdefault(
                node_id,
                ValidationIssue(
                    "column_not_allowed",
                    f"The column {name} is not available to this service.",
                ),
            )

    @staticmethod
    def _derived_has(source: Scope, name: str) -> bool:
        """Return whether a derived table or CTE produces a column of this name."""
        names = _named_selects(source.expression)
        if not names or any(output == "*" for output in names):
            return True
        return name.casefold() in {output.casefold() for output in names}

    # -- structural checks -------------------------------------------------
    def _check_stars(self, expression: exp.Expr, report: ValidationReport) -> None:
        """Refuse a star in the projection unless configuration allows it."""
        if self._security.allow_select_star:
            return
        for select in expression.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                    isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
                ):
                    report.issues.append(
                        ValidationIssue(
                            "select_star_not_allowed",
                            "Selecting all columns is not permitted. Name the columns "
                            "the answer needs.",
                        )
                    )
                    return

    def _check_cost(self, expression: exp.Expr, report: ValidationReport, table_count: int) -> None:
        """Apply the query cost ceilings."""
        joins = list(expression.find_all(exp.Join))
        if len(joins) > self._limits.max_joins:
            report.issues.append(
                ValidationIssue(
                    "too_many_joins",
                    f"The query joins {len(joins)} times, above the limit of "
                    f"{self._limits.max_joins}.",
                )
            )
        if not self._security.allow_cross_join:
            for join in joins:
                has_condition = bool(join.args.get("on") or join.args.get("using"))
                kind = (join.args.get("kind") or "").upper()
                if not has_condition and kind != "LATERAL":
                    report.issues.append(
                        ValidationIssue(
                            "cartesian_join",
                            "A join without a condition would combine every row with "
                            "every other row and is not permitted.",
                        )
                    )
                    break
        if table_count > self._limits.max_tables:
            report.issues.append(
                ValidationIssue(
                    "too_many_tables",
                    f"The query reads {table_count} tables, above the limit of "
                    f"{self._limits.max_tables}.",
                )
            )
        depth = self._subquery_depth(expression)
        if depth > self._limits.max_subquery_depth:
            report.issues.append(
                ValidationIssue(
                    "subquery_too_deep",
                    f"The query nests subqueries {depth} deep, above the limit of "
                    f"{self._limits.max_subquery_depth}.",
                )
            )

    @staticmethod
    def _subquery_depth(expression: exp.Expr) -> int:
        """Return how deeply SELECT statements are nested inside one another."""
        deepest = 0
        for select in expression.find_all(exp.Select):
            depth = 0
            node = select.parent
            while node is not None:
                if isinstance(node, exp.Select):
                    depth += 1
                node = node.parent
            deepest = max(deepest, depth)
        return deepest

    @staticmethod
    def _check_declared(
        tables: set[str], declared: Sequence[str], report: ValidationReport
    ) -> None:
        """Warn when the model's own list of tables does not match the query.

        Names are compared without their schema, because a model reporting
        ``facilities`` for a query that reads ``dbo.facilities`` has not
        contradicted itself, and a warning on every query would train a reader
        to ignore the ones that matter.
        """
        if not declared:
            return

        def bare(name: str) -> str:
            cleaned = name.replace("[", "").replace("]", "").replace('"', "").strip()
            return cleaned.rpartition(".")[2].casefold()

        actual = {bare(name) for name in tables}
        claimed = {bare(name) for name in declared if name.strip()}
        if claimed and not claimed.issubset(actual):
            report.issues.append(
                ValidationIssue(
                    "declared_tables_mismatch",
                    "The query reads different tables from the ones the model reported.",
                    severity="warning",
                )
            )

    # -- rewriting ---------------------------------------------------------
    @staticmethod
    def _output_columns(expression: exp.Expr) -> tuple[str, ...]:
        """Return the names of the columns the query produces."""
        return tuple(name for name in _named_selects(expression) if name)

    def _masked_outputs(
        self,
        expression: exp.Expr,
        bindings: dict[int, tuple[str, str, str]],
        catalog: SchemaCatalog,
    ) -> frozenset[str]:
        """Return the output columns whose values must be masked.

        A column is masked when it is a masked column, and so is any expression
        computed from one, because the maximum of a masked column discloses a
        masked value just as surely as the column itself.
        """
        select = expression.find(exp.Select)
        if select is None:
            return frozenset()
        masked: set[str] = set()
        for projection in select.expressions:
            output_name = projection.alias_or_name
            if not output_name:
                continue
            for column in projection.find_all(exp.Column):
                binding = bindings.get(id(column))
                if binding and self._policy.is_column_masked(*binding):
                    masked.add(output_name)
                    break
        return frozenset(masked)

    def _apply_filters(
        self,
        expression: exp.Expr,
        filters: Sequence[QueryFilter],
        output_columns: tuple[str, ...],
        parameters: dict[str, Any],
        report: ValidationReport,
        dialect: str,
    ) -> exp.Expr:
        """Wrap the query so caller filters apply to its result columns."""
        available = {name.casefold(): name for name in output_columns}
        conditions: list[exp.Expr] = []

        for index, item in enumerate(filters):
            actual = available.get(item.column.casefold())
            if actual is None:
                report.issues.append(
                    ValidationIssue(
                        "filter_column_unknown",
                        f"The filter column {item.column} is not one of the columns this "
                        f"query returns ({', '.join(output_columns) or 'none'}).",
                    )
                )
                continue
            condition = self._filter_condition(item, actual, index, parameters, report)
            if condition is not None:
                conditions.append(condition)

        if report.errors or not conditions:
            return expression

        inner = expression.copy()
        if not isinstance(inner, exp.Query):
            report.issues.append(
                ValidationIssue(
                    "filters_not_applicable",
                    "Filters can only be applied to a query that returns columns.",
                )
            )
            return expression
        outer_order: exp.Expr | None = None
        order = inner.args.get("order") if isinstance(inner, exp.Select) else None
        if order is not None:
            if self._order_is_liftable(order, available):
                outer_order = self._lift_order(order)
                inner.set("order", None)
            else:
                # A derived table may not carry an ORDER BY without a row limit
                # in T-SQL, and the order cannot be expressed against the
                # output columns, so it is bounded here and the caller is told.
                inner = inner.limit(self._limits.max_rows)
                report.issues.append(
                    ValidationIssue(
                        "ordering_not_preserved",
                        "The requested filters were applied after ordering, so the order "
                        "of the returned rows may differ from the order of the query.",
                        severity="warning",
                    )
                )

        wrapper = exp.select(*[exp.column(name, RESULT_ALIAS) for name in output_columns]).from_(
            inner.subquery(alias=RESULT_ALIAS)
        )
        for condition in conditions:
            wrapper = wrapper.where(condition)
        if outer_order is not None:
            wrapper.set("order", outer_order)
        return wrapper

    def _filter_condition(
        self,
        item: QueryFilter,
        column_name: str,
        index: int,
        parameters: dict[str, Any],
        report: ValidationReport,
    ) -> exp.Expr | None:
        """Build one filter predicate with its value bound to a marker."""
        column = exp.column(column_name, RESULT_ALIAS)

        if item.operator == "in":
            values = item.value if isinstance(item.value, list | tuple) else [item.value]
            if not values:
                report.issues.append(
                    ValidationIssue("filter_invalid", "An 'in' filter needs at least one value.")
                )
                return None
            markers: list[exp.Expr] = []
            for offset, value in enumerate(values):
                marker = FILTER_MARKER.format(index=f"{index}_{offset}")
                parameters[marker] = value
                markers.append(exp.var(marker))
            return exp.In(this=column, expressions=markers)

        marker = FILTER_MARKER.format(index=index)
        if item.operator == "contains":
            parameters[marker] = f"%{item.value}%"
            return exp.Like(this=column, expression=exp.var(marker))

        operator = _OPERATORS.get(item.operator)
        if operator is None:
            report.issues.append(
                ValidationIssue("filter_invalid", f"Unsupported filter operator {item.operator}.")
            )
            return None
        parameters[marker] = item.value
        return operator(this=column, expression=exp.var(marker))

    @staticmethod
    def _order_is_liftable(order: exp.Expr, available: dict[str, str]) -> bool:
        """Return whether every ordering term refers to an output column."""
        for ordered in order.find_all(exp.Ordered):
            target = ordered.this
            if isinstance(target, exp.Column) and target.name.casefold() in available:
                continue
            if isinstance(target, exp.Literal):
                continue
            return False
        return True

    @staticmethod
    def _lift_order(order: exp.Expr) -> exp.Expr:
        """Rewrite an ORDER BY so it refers to the wrapping query's columns."""
        lifted = order.copy()
        for column in lifted.find_all(exp.Column):
            column.replace(exp.column(column.name, RESULT_ALIAS))
        return lifted

    def _apply_row_limit(self, expression: exp.Expr) -> tuple[exp.Expr, exp.Expr, int | None]:
        """Clamp the row limit and build the rendering that detects truncation."""
        maximum = self._limits.max_rows
        if not isinstance(expression, exp.Query):
            # Nothing that returns rows, so there is nothing to bound.
            return expression, expression, None
        existing = expression.args.get("limit") if isinstance(expression, exp.Select) else None

        if isinstance(existing, exp.Limit):
            value = existing.expression
            if isinstance(value, exp.Literal) and value.is_int:
                requested = int(value.name)
                if 0 < requested <= maximum:
                    # The model asked for fewer rows than the ceiling, which is
                    # usually the question asking for a top N. Honour it, and
                    # there is nothing to truncate.
                    return expression, expression, requested

        display = expression.limit(maximum)
        execution = expression.limit(maximum + 1)
        return display, execution, maximum
