"""Metadata service.

The single place the rest of the application asks what the database contains.
It owns the cache and, more importantly, it applies the access policy on the
way out. Everything downstream, the retriever that builds prompt context, the
validator that checks generated SQL, and the schema endpoints, sees only the
filtered catalogue. A blocked table therefore cannot be described to a model,
cannot pass validation, and cannot appear in an API response.
"""

from __future__ import annotations

import asyncio

from nl2sql.config.settings import SchemaCacheSettings
from nl2sql.core.exceptions import SchemaNotFoundError
from nl2sql.metadata.cache import TTLCache
from nl2sql.metadata.introspector import SchemaIntrospector
from nl2sql.metadata.models import ForeignKeyInfo, IndexInfo, SchemaCatalog, TableInfo
from nl2sql.observability.logging import get_logger
from nl2sql.security.policy import AccessPolicy

logger = get_logger(__name__)


class MetadataService:
    """Caches the discovered schema and filters it through the access policy."""

    def __init__(
        self,
        introspector: SchemaIntrospector,
        policy: AccessPolicy,
        settings: SchemaCacheSettings,
    ) -> None:
        self._introspector = introspector
        self._policy = policy
        self._settings = settings
        self._cache: TTLCache[SchemaCatalog] = TTLCache(
            self._load,
            ttl_seconds=settings.ttl_seconds,
            serve_stale_on_error=settings.serve_stale_on_error,
            name="schema_catalog",
        )

    # -- loading -----------------------------------------------------------
    def _load(self) -> SchemaCatalog:
        return self.apply_policy(self._introspector.introspect())

    def apply_policy(self, catalog: SchemaCatalog) -> SchemaCatalog:
        """Return ``catalog`` with everything the policy forbids removed."""
        kept: list[TableInfo] = []
        dropped_tables = 0
        dropped_columns = 0

        allowed_keys = {
            table.key
            for table in catalog.tables
            if self._policy.is_table_allowed(table.schema_name, table.name)
        }

        for table in catalog.tables:
            if table.key not in allowed_keys:
                dropped_tables += 1
                continue
            columns = tuple(
                column
                for column in table.columns
                if self._policy.is_column_allowed(table.schema_name, table.name, column.name)
            )
            dropped_columns += len(table.columns) - len(columns)
            if not columns:
                dropped_tables += 1
                continue
            visible = {column.key for column in columns}
            foreign_keys = tuple(
                fk for fk in table.foreign_keys if self._keep_foreign_key(fk, visible, allowed_keys)
            )
            indexes = tuple(
                index
                for index in table.indexes
                if all(column.casefold() in visible for column in index.columns)
            )
            kept.append(
                TableInfo(
                    schema_name=table.schema_name,
                    name=table.name,
                    kind=table.kind,
                    columns=columns,
                    primary_key=tuple(
                        column for column in table.primary_key if column.casefold() in visible
                    ),
                    foreign_keys=foreign_keys,
                    indexes=indexes,
                    comment=table.comment,
                )
            )

        if dropped_tables or dropped_columns:
            logger.info(
                "schema_policy_applied",
                tables_visible=len(kept),
                tables_hidden=dropped_tables,
                columns_hidden=dropped_columns,
            )

        return SchemaCatalog(
            tables=tuple(kept),
            default_schema=catalog.default_schema,
            dialect=catalog.dialect,
            discovered_at=catalog.discovered_at,
        )

    @staticmethod
    def _keep_foreign_key(
        fk: ForeignKeyInfo, visible_columns: set[str], allowed_tables: set[str]
    ) -> bool:
        if fk.referred_qualified_name.casefold() not in allowed_tables:
            return False
        return all(column.casefold() in visible_columns for column in fk.columns)

    # -- access ------------------------------------------------------------
    def catalog(self) -> SchemaCatalog:
        """Return the filtered catalogue, refreshing it when the cache is stale."""
        return self._cache.get()

    async def catalog_async(self) -> SchemaCatalog:
        """Return the filtered catalogue without blocking the event loop."""
        return await asyncio.to_thread(self.catalog)

    def refresh(self) -> SchemaCatalog:
        """Force a reload of the catalogue."""
        self._cache.invalidate()
        return self._cache.get()

    @property
    def is_warm(self) -> bool:
        """Return whether the catalogue has been loaded at least once."""
        return self._cache.is_warm

    @property
    def age_seconds(self) -> float | None:
        """Return the age of the cached catalogue."""
        return self._cache.age_seconds

    def list_schemas(self) -> dict[str, tuple[str, ...]]:
        """Return each visible schema with the names of its tables."""
        catalog = self.catalog()
        grouped: dict[str, list[str]] = {}
        for table in catalog.tables:
            grouped.setdefault(table.schema_name, []).append(table.name)
        return {schema: tuple(sorted(names)) for schema, names in sorted(grouped.items())}

    def list_tables(self, schema: str | None = None) -> tuple[TableInfo, ...]:
        """Return the visible tables, optionally restricted to one schema."""
        catalog = self.catalog()
        if schema is None:
            return catalog.tables
        tables = catalog.tables_in(schema)
        if not tables:
            raise SchemaNotFoundError(
                f"Schema {schema} is not available.",
                public_message="The requested schema is not available.",
            )
        return tables

    def unused_index_hint(self, table: TableInfo) -> tuple[IndexInfo, ...]:
        """Return the indexes on a table, used as filtering hints in prompts."""
        return table.indexes
