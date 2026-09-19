"""Health and readiness.

Liveness checks nothing external, so a database blip cannot cause an
orchestrator to restart otherwise healthy replicas. Readiness checks the
dependencies a request actually needs: the database answers, the schema cache
is usable, and at least one model provider is available.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response, status

from nl2sql.api.deps import ContainerDep
from nl2sql.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness")
async def health(container: ContainerDep) -> HealthResponse:
    """Report that the process is running."""
    return HealthResponse(
        status="ok",
        service=container.settings.app.name,
        version=container.settings.app.version,
    )


@router.get("/ready", response_model=ReadyResponse, summary="Readiness")
async def ready(container: ContainerDep, response: Response) -> ReadyResponse:
    """Report whether the service can answer a question right now."""
    database, providers = await asyncio.gather(
        container.executor.ping(),
        container.providers.health(),
    )
    schema_ready = await _schema_ready(container)
    model_ready = any(item.available for item in providers)

    checks = {
        "database": bool(database),
        "schema": schema_ready,
        "language_model": model_ready,
    }
    ok = all(checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="ready" if ok else "not_ready", checks=checks)


async def _schema_ready(container: ContainerDep) -> bool:
    """Return whether the catalogue can be read."""
    try:
        catalog = await container.metadata.catalog_async()
    except Exception:
        return False
    return len(catalog) > 0


@router.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus metrics, when enabled by configuration",
)
async def metrics(container: ContainerDep) -> Response:
    """Expose the in process metrics in the Prometheus text format."""
    if not container.settings.observability.metrics_endpoint_enabled:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return Response(
        content=container.metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4",
    )
