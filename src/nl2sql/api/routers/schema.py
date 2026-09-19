"""The schema endpoints.

What these return is the policy filtered catalogue, the same view the model
sees. A table or column the policy hides is absent here too, so this endpoint
cannot be used to discover what exists behind the allowlist.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from nl2sql.api.deps import AdminDep, ContainerDep, PrincipalDep
from nl2sql.api.schemas import (
    ErrorResponse,
    ForeignKeyModel,
    SchemaResponse,
    SchemaSummary,
    TableColumnModel,
    TableModel,
    TablesResponse,
)
from nl2sql.metadata.models import TableInfo

router = APIRouter(prefix="/api/v1", tags=["schema"])

_ERRORS: dict[int | str, dict[str, object]] = {
    401: {"model": ErrorResponse, "description": "Authentication failed."},
    404: {"model": ErrorResponse, "description": "The schema is not available."},
    503: {"model": ErrorResponse, "description": "Schema discovery is unavailable."},
}


def _to_model(table: TableInfo) -> TableModel:
    """Convert a discovered table into its API shape."""
    return TableModel(
        schema_name=table.schema_name,
        name=table.name,
        kind=table.kind,
        description=table.comment,
        columns=[
            TableColumnModel(
                name=column.name,
                type=column.data_type,
                nullable=column.nullable,
                primary_key=column.is_primary_key,
                description=column.comment,
            )
            for column in table.columns
        ],
        primary_key=list(table.primary_key),
        foreign_keys=[
            ForeignKeyModel(
                columns=list(fk.columns),
                references=fk.referred_qualified_name,
                referenced_columns=list(fk.referred_columns),
            )
            for fk in table.foreign_keys
        ],
    )


@router.get(
    "/schema",
    response_model=SchemaResponse,
    responses=_ERRORS,
    summary="List the schemas and tables available to this caller",
)
async def get_schema(container: ContainerDep, principal: PrincipalDep) -> SchemaResponse:
    """Return every visible schema with the names of its tables."""
    catalog = await container.metadata.catalog_async()
    grouped: dict[str, list[str]] = {}
    for table in catalog.tables:
        grouped.setdefault(table.schema_name, []).append(table.name)
    return SchemaResponse(
        schemas=[
            SchemaSummary(name=name, table_count=len(tables), tables=sorted(tables))
            for name, tables in sorted(grouped.items())
        ],
        table_count=len(catalog),
        discovered_at=catalog.discovered_at.isoformat() if catalog.discovered_at else None,
    )


@router.get(
    "/schema/tables",
    response_model=TablesResponse,
    responses=_ERRORS,
    summary="Describe the tables available to this caller",
)
async def get_tables(
    container: ContainerDep,
    principal: PrincipalDep,
    schema_name: str | None = Query(
        default=None, alias="schema", max_length=128, description="Restrict to one schema."
    ),
    table: str | None = Query(default=None, max_length=128, description="Restrict to one table."),
) -> TablesResponse:
    """Return the columns and relationships of the visible tables."""
    tables = await asyncio.to_thread(container.metadata.list_tables, schema_name)
    if table:
        wanted = table.casefold()
        tables = tuple(item for item in tables if item.name.casefold() == wanted)
    models = [_to_model(item) for item in tables]
    return TablesResponse(tables=models, count=len(models))


@router.post(
    "/schema/refresh",
    response_model=SchemaResponse,
    responses=_ERRORS,
    summary="Discard the cached schema and read it again",
)
async def refresh_schema(container: ContainerDep, principal: AdminDep) -> SchemaResponse:
    """Force rediscovery, for use after a schema change."""
    catalog = await asyncio.to_thread(container.metadata.refresh)
    grouped: dict[str, list[str]] = {}
    for item in catalog.tables:
        grouped.setdefault(item.schema_name, []).append(item.name)
    return SchemaResponse(
        schemas=[
            SchemaSummary(name=name, table_count=len(names), tables=sorted(names))
            for name, names in sorted(grouped.items())
        ],
        table_count=len(catalog),
        discovered_at=catalog.discovered_at.isoformat() if catalog.discovered_at else None,
    )
