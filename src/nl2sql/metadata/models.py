"""Immutable description of a discovered database schema.

Nothing here is declared ahead of time. Every instance is produced by
introspecting the live database, so the assistant has no built in knowledge of
any table, column or metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """One column as reported by the database."""

    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    comment: str | None = None

    @property
    def key(self) -> str:
        """Return the case folded name used for lookups."""
        return self.name.casefold()


@dataclass(frozen=True, slots=True)
class ForeignKeyInfo:
    """A foreign key relationship discovered on a table."""

    columns: tuple[str, ...]
    referred_schema: str
    referred_table: str
    referred_columns: tuple[str, ...]
    name: str | None = None

    @property
    def referred_qualified_name(self) -> str:
        """Return ``schema.table`` of the referenced table."""
        return f"{self.referred_schema}.{self.referred_table}"


@dataclass(frozen=True, slots=True)
class IndexInfo:
    """An index, used as a hint about which columns are cheap to filter on."""

    name: str
    columns: tuple[str, ...]
    unique: bool = False


@dataclass(frozen=True, slots=True)
class TableInfo:
    """A table or view with its columns and relationships."""

    schema_name: str
    name: str
    kind: str = "table"
    columns: tuple[ColumnInfo, ...] = ()
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKeyInfo, ...] = ()
    indexes: tuple[IndexInfo, ...] = ()
    comment: str | None = None

    @property
    def qualified_name(self) -> str:
        """Return ``schema.table``."""
        return f"{self.schema_name}.{self.name}"

    @property
    def key(self) -> str:
        """Return the case folded qualified name used for lookups."""
        return self.qualified_name.casefold()

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return every column name in ordinal order."""
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnInfo | None:
        """Return the column called ``name``, case insensitively."""
        folded = name.casefold()
        for column in self.columns:
            if column.key == folded:
                return column
        return None

    def has_column(self, name: str) -> bool:
        """Return whether the table has a column called ``name``."""
        return self.column(name) is not None


@dataclass(slots=True)
class SchemaCatalog:
    """Every table the caller is allowed to see, indexed for lookup."""

    tables: tuple[TableInfo, ...] = ()
    default_schema: str | None = None
    dialect: str = "unknown"
    discovered_at: datetime | None = None
    _by_key: dict[str, TableInfo] = field(default_factory=dict, init=False, repr=False)
    _by_name: dict[str, list[TableInfo]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for table in self.tables:
            self._by_key[table.key] = table
            self._by_name.setdefault(table.name.casefold(), []).append(table)

    def __len__(self) -> int:
        return len(self.tables)

    @property
    def schemas(self) -> tuple[str, ...]:
        """Return the distinct schema names, sorted."""
        return tuple(sorted({table.schema_name for table in self.tables}))

    def tables_in(self, schema: str) -> tuple[TableInfo, ...]:
        """Return the tables in one schema."""
        folded = schema.casefold()
        return tuple(t for t in self.tables if t.schema_name.casefold() == folded)

    def get(self, schema: str, name: str) -> TableInfo | None:
        """Return the table with this exact schema and name."""
        return self._by_key.get(f"{schema}.{name}".casefold())

    def resolve(self, reference: str, *, default_schema: str | None = None) -> TableInfo | None:
        """Resolve a table reference that may or may not carry a schema.

        An unqualified name resolves against the default schema first, then
        against a unique match anywhere in the catalog. An unqualified name
        that exists in more than one schema is ambiguous and returns ``None``,
        because guessing which one the model meant is exactly the kind of
        silent mistake this service must not make.
        """
        cleaned = reference.replace("[", "").replace("]", "").replace('"', "").strip()
        if "." in cleaned:
            schema, _, name = cleaned.rpartition(".")
            return self.get(schema, name)
        candidates = self._by_name.get(cleaned.casefold(), [])
        if not candidates:
            return None
        if default_schema:
            folded_default = default_schema.casefold()
            for table in candidates:
                if table.schema_name.casefold() == folded_default:
                    return table
        if len(candidates) == 1:
            return candidates[0]
        return None

    def is_ambiguous(self, reference: str) -> bool:
        """Return whether an unqualified reference matches several schemas."""
        if "." in reference:
            return False
        return len(self._by_name.get(reference.casefold(), [])) > 1

    def neighbours(self, table: TableInfo) -> tuple[TableInfo, ...]:
        """Return tables joined to ``table`` by a foreign key in either direction."""
        found: dict[str, TableInfo] = {}
        for fk in table.foreign_keys:
            referenced = self.get(fk.referred_schema, fk.referred_table)
            if referenced is not None:
                found[referenced.key] = referenced
        for other in self.tables:
            if other.key == table.key:
                continue
            for fk in other.foreign_keys:
                if f"{fk.referred_schema}.{fk.referred_table}".casefold() == table.key:
                    found[other.key] = other
        return tuple(found.values())
