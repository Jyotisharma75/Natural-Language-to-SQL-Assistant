"""Tenant isolation.

The tenant identifier comes from the authenticated principal and from nowhere
else. It is never read from the question, the filters or a request header that
the caller controls, because any of those would let a caller ask for another
tenant's rows simply by asking nicely.

Isolation is applied by rewriting the parsed query: every table that carries a
tenant column gains a predicate binding that column to a parameter. The
predicate is added to the WHERE clause for a table in FROM, and to the ON
clause for a joined table, so an outer join keeps its semantics instead of
being silently turned into an inner join.

The value is always a bound parameter. It is never formatted into SQL text.

Where the database also has row level security, the executor sets the session
context key as a second, independent layer, so a mistake in this rewriting
cannot by itself expose another tenant's data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from sqlglot import exp

from nl2sql.config.settings import TenancySettings
from nl2sql.core.exceptions import TenantContextError
from nl2sql.metadata.models import SchemaCatalog, TableInfo

#: Name of the parameter marker holding the tenant identifier. The reserved
#: prefix is refused in model output by the guard, so a generated statement can
#: never contain something that looks like this marker.
TENANT_PARAMETER = "__NL2SQL_PTENANT__"


@dataclass(slots=True)
class TenantScopeResult:
    """The rewritten query plus the parameters it now needs."""

    expression: exp.Expr
    parameters: dict[str, Any] = field(default_factory=dict)
    scoped_tables: tuple[str, ...] = ()
    unscoped_tables: tuple[str, ...] = ()


class TenantScoper:
    """Adds tenant predicates to a parsed query."""

    def __init__(self, settings: TenancySettings) -> None:
        self._settings = settings
        self._patterns = tuple(settings.tenant_column_patterns)

    @property
    def enabled(self) -> bool:
        """Return whether tenant scoping is switched on."""
        return self._settings.enabled

    def tenant_column(self, table: TableInfo) -> str | None:
        """Return the column of ``table`` that carries the tenant, if any."""
        for column in table.columns:
            folded = column.name.casefold()
            if any(fnmatchcase(folded, pattern.casefold()) for pattern in self._patterns):
                return column.name
        return None

    def apply(
        self,
        expression: exp.Expr,
        catalog: SchemaCatalog,
        *,
        tenant_id: str | None,
    ) -> TenantScopeResult:
        """Return ``expression`` with a tenant predicate on every scoped table."""
        if not self._settings.enabled:
            return TenantScopeResult(expression=expression)

        if tenant_id is None:
            if self._settings.require_tenant:
                raise TenantContextError(
                    "The request has no tenant context but tenant isolation is enabled."
                )
            return TenantScopeResult(expression=expression)

        tree = expression.copy()
        cte_names = {cte.alias.casefold() for cte in tree.find_all(exp.CTE) if cte.alias}
        scoped: list[str] = []
        unscoped: list[str] = []

        for select in tree.find_all(exp.Select):
            # The parser names this argument from_ in current versions and from
            # in older ones. Reading only one of them would silently skip the
            # table in the FROM clause, which is the one that matters most.
            from_clause = select.args.get("from_") or select.args.get("from")
            if isinstance(from_clause, exp.From):
                source = from_clause.this
                if isinstance(source, exp.Table):
                    predicate = self._predicate(source, catalog, cte_names, scoped, unscoped)
                    if predicate is not None:
                        select.where(predicate, copy=False)
            for join in select.args.get("joins") or []:
                source = join.this
                if isinstance(source, exp.Table):
                    predicate = self._predicate(source, catalog, cte_names, scoped, unscoped)
                    if predicate is not None:
                        join.on(predicate, copy=False)

        parameters: dict[str, Any] = {TENANT_PARAMETER: tenant_id} if scoped else {}
        return TenantScopeResult(
            expression=tree,
            parameters=parameters,
            scoped_tables=tuple(dict.fromkeys(scoped)),
            unscoped_tables=tuple(dict.fromkeys(unscoped)),
        )

    def _predicate(
        self,
        source: exp.Table,
        catalog: SchemaCatalog,
        cte_names: set[str],
        scoped: list[str],
        unscoped: list[str],
    ) -> exp.Expr | None:
        """Build the tenant predicate for one table reference."""
        reference = source.sql(dialect=None, comments=False)
        name = source.name or ""
        if not name or name.casefold() in cte_names:
            return None
        qualified = f"{source.db}.{name}" if source.db else name
        table = catalog.resolve(qualified, default_schema=catalog.default_schema)
        if table is None:
            unscoped.append(reference)
            return None
        column_name = self.tenant_column(table)
        if column_name is None:
            unscoped.append(table.qualified_name)
            return None
        scoped.append(table.qualified_name)
        qualifier = source.alias_or_name
        return exp.EQ(
            this=exp.column(column_name, qualifier),
            expression=exp.var(TENANT_PARAMETER),
        )
