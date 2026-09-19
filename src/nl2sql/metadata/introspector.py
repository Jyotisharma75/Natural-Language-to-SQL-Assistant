"""Dynamic schema discovery.

Uses SQLAlchemy reflection, which issues the dialect's own catalogue queries.
On Azure SQL that means the ``sys`` and ``INFORMATION_SCHEMA`` views, including
the ``MS_Description`` extended properties that carry table and column
descriptions. The batched ``get_multi_*`` calls reflect a whole schema in a
handful of round trips instead of one per table, which matters on a database
with hundreds of tables.

No table, column or relationship is ever assumed. Whatever the database
reports is what the assistant knows.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, inspect
from sqlalchemy.engine import ObjectKind
from sqlalchemy.exc import SQLAlchemyError

from nl2sql.core.exceptions import MetadataError
from nl2sql.metadata.models import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    SchemaCatalog,
    TableInfo,
)
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)


class SchemaIntrospector:
    """Reflects the live database into a :class:`SchemaCatalog`."""

    def __init__(
        self,
        engine: Engine,
        *,
        include_views: bool = True,
        include_indexes: bool = True,
        schema_filter: Callable[[str], bool] | None = None,
    ) -> None:
        self._engine = engine
        self._include_views = include_views
        self._include_indexes = include_indexes
        self._schema_filter = schema_filter or (lambda _: True)

    @property
    def dialect(self) -> str:
        """Return the SQLAlchemy dialect name of the inspected database."""
        return self._engine.dialect.name

    def introspect(self) -> SchemaCatalog:
        """Return the current catalogue of every discoverable table and view."""
        try:
            return self._introspect()
        except SQLAlchemyError as exc:
            raise MetadataError(
                f"Schema introspection failed: {type(exc).__name__}",
                details={"error": str(exc)},
            ) from exc

    def _introspect(self) -> SchemaCatalog:
        inspector = inspect(self._engine)
        default_schema = inspector.default_schema_name
        schema_names = [name for name in inspector.get_schema_names() if self._schema_filter(name)]

        tables: list[TableInfo] = []
        for schema in schema_names:
            table_names = set(inspector.get_table_names(schema=schema))
            view_names = (
                set(inspector.get_view_names(schema=schema)) if self._include_views else set()
            )
            if not table_names and not view_names:
                continue

            columns = self._safe_multi(inspector.get_multi_columns, schema)
            pk_constraints = self._safe_multi(inspector.get_multi_pk_constraint, schema)
            foreign_keys = self._safe_multi(inspector.get_multi_foreign_keys, schema)
            indexes = (
                self._safe_multi(inspector.get_multi_indexes, schema)
                if self._include_indexes
                else {}
            )
            comments = self._safe_multi(inspector.get_multi_table_comment, schema)

            for key, column_dicts in columns.items():
                name = key[1]
                if name not in table_names and name not in view_names:
                    continue
                kind = "view" if name in view_names else "table"
                pk_columns = tuple((pk_constraints.get(key) or {}).get("constrained_columns") or ())
                folded_pk = {column.casefold() for column in pk_columns}
                tables.append(
                    TableInfo(
                        schema_name=schema,
                        name=name,
                        kind=kind,
                        columns=tuple(
                            ColumnInfo(
                                name=str(column["name"]),
                                data_type=self._render_type(column.get("type")),
                                nullable=bool(column.get("nullable", True)),
                                is_primary_key=str(column["name"]).casefold() in folded_pk,
                                comment=self._clean(column.get("comment")),
                            )
                            for column in column_dicts
                        ),
                        primary_key=pk_columns,
                        foreign_keys=self._foreign_keys(foreign_keys.get(key) or (), schema),
                        indexes=self._indexes(indexes.get(key) or ()),
                        comment=self._clean((comments.get(key) or {}).get("text")),
                    )
                )

        catalog = SchemaCatalog(
            tables=tuple(tables),
            default_schema=default_schema,
            dialect=self.dialect,
            discovered_at=datetime.now(UTC),
        )
        logger.info(
            "schema_introspected",
            dialect=catalog.dialect,
            schema_count=len(catalog.schemas),
            table_count=len(catalog),
        )
        return catalog

    def _safe_multi(
        self, method: Callable[..., Any], schema: str
    ) -> dict[tuple[str | None, str], Any]:
        """Call a batched reflection method, tolerating dialects that lack it."""
        try:
            raw = dict(method(schema=schema, kind=ObjectKind.ANY))
        except NotImplementedError:
            return {}
        except TypeError:
            try:
                raw = dict(method(schema=schema))
            except NotImplementedError:
                return {}
        return {(key[0] or schema, key[1]): value for key, value in raw.items()}

    @staticmethod
    def _render_type(type_object: Any) -> str:
        """Render a reflected type as the string shown to the model."""
        if type_object is None:
            return "UNKNOWN"
        try:
            rendered: str = str(type_object)
            return rendered
        except Exception:  # a dialect specific type may fail to render
            return str(type_object.__class__.__name__).upper()

    @staticmethod
    def _clean(value: Any) -> str | None:
        """Return a trimmed, single line description, or None."""
        if not value:
            return None
        text = " ".join(str(value).split())
        return text or None

    def _foreign_keys(self, raw: Any, schema: str) -> tuple[ForeignKeyInfo, ...]:
        found: list[ForeignKeyInfo] = []
        for fk in raw:
            constrained = tuple(fk.get("constrained_columns") or ())
            referred_table = fk.get("referred_table")
            if not constrained or not referred_table:
                continue
            found.append(
                ForeignKeyInfo(
                    columns=constrained,
                    referred_schema=fk.get("referred_schema") or schema,
                    referred_table=str(referred_table),
                    referred_columns=tuple(fk.get("referred_columns") or ()),
                    name=fk.get("name"),
                )
            )
        return tuple(found)

    @staticmethod
    def _indexes(raw: Any) -> tuple[IndexInfo, ...]:
        found: list[IndexInfo] = []
        for index in raw:
            name = index.get("name")
            columns = tuple(str(c) for c in (index.get("column_names") or ()) if c)
            if not name or not columns:
                continue
            found.append(
                IndexInfo(name=str(name), columns=columns, unique=bool(index.get("unique")))
            )
        return tuple(found)
