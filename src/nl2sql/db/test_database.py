"""Test database abstraction.

Integration tests need a real database with a real schema so that reflection,
validation and execution are exercised end to end, but they must not need
Azure SQL to be reachable. This builds one from a declarative specification on
any SQLAlchemy URL, defaulting to SQLite in memory.

The specification is data, not a fixed schema: a test states the tables it
wants and gets them. Nothing in the application knows these tables exist, so a
test schema and a production schema are handled by exactly the same code path.

The same abstraction points at a real Azure SQL database by passing its URL,
which is how the live integration tests run against the genuine dialect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    create_engine,
    insert,
)
from sqlalchemy.pool import StaticPool

#: Type names a specification may use, mapped onto SQLAlchemy types. Keeping
#: this small and explicit means a specification cannot smuggle arbitrary
#: constructor calls into a test fixture.
_TYPES = {
    "int": Integer,
    "float": Float,
    "bool": Boolean,
    "date": Date,
    "datetime": DateTime,
}


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column of a fixture table."""

    name: str
    type: str = "int"
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None
    comment: str | None = None


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One fixture table and the rows to seed it with."""

    name: str
    columns: tuple[ColumnSpec, ...]
    schema: str | None = None
    comment: str | None = None
    rows: tuple[Mapping[str, Any], ...] = ()


def _column_type(spec: str) -> Any:
    """Build a SQLAlchemy type from a short specification such as ``varchar(50)``."""
    text = spec.strip().lower()
    if text.startswith("varchar"):
        length = text[text.find("(") + 1 : text.find(")")] if "(" in text else "255"
        return String(int(length))
    if text.startswith(("decimal", "numeric")):
        inner = text[text.find("(") + 1 : text.find(")")] if "(" in text else "18,4"
        precision, _, scale = inner.partition(",")
        return Numeric(int(precision), int(scale or 0))
    if text not in _TYPES:
        raise ValueError(f"Unsupported fixture column type: {spec}")
    return _TYPES[text]()


@dataclass(slots=True)
class TestDatabase:
    """A live database built from a table specification."""

    __test__ = False  # this is a fixture helper, not a test case

    specs: Sequence[TableSpec]
    url: str = "sqlite+pysqlite:///:memory:"
    engine: Engine = field(init=False)
    metadata: MetaData = field(init=False)

    def __post_init__(self) -> None:
        if ":memory:" in self.url:
            # One shared connection, so every session sees the same in memory
            # database rather than an empty private one.
            self.engine = create_engine(
                self.url, poolclass=StaticPool, connect_args={"check_same_thread": False}
            )
        else:
            self.engine = create_engine(self.url)
        self.metadata = MetaData()

    @property
    def dialect(self) -> str:
        """Return the SQLAlchemy dialect name in use."""
        return self.engine.dialect.name

    def create(self) -> TestDatabase:
        """Create the tables and seed their rows."""
        tables: dict[str, Table] = {}
        for spec in self.specs:
            columns = []
            for column in spec.columns:
                constraints = [ForeignKey(column.foreign_key)] if column.foreign_key else []
                columns.append(
                    Column(
                        column.name,
                        _column_type(column.type),
                        *constraints,
                        primary_key=column.primary_key,
                        nullable=column.nullable and not column.primary_key,
                        comment=column.comment,
                    )
                )
            tables[spec.name] = Table(
                spec.name, self.metadata, *columns, schema=spec.schema, comment=spec.comment
            )
        self.metadata.create_all(self.engine)

        with self.engine.begin() as connection:
            for spec in self.specs:
                if spec.rows:
                    connection.execute(insert(tables[spec.name]), [dict(r) for r in spec.rows])
        return self

    def drop(self) -> None:
        """Drop every fixture table."""
        self.metadata.drop_all(self.engine)

    def dispose(self) -> None:
        """Close the pool."""
        self.engine.dispose()
