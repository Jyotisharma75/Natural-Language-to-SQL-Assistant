"""Application factory.

Builds the container at startup, warms what is slow, and disposes the pools on
shutdown. The factory takes an optional container so tests can supply their own
engines and model providers and still exercise the real application.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nl2sql.api.errors import register_exception_handlers
from nl2sql.api.middleware import RequestContextMiddleware
from nl2sql.api.routers import health, query, schema
from nl2sql.config.loader import load_settings
from nl2sql.config.settings import Settings
from nl2sql.container import Container
from nl2sql.observability.logging import get_logger
from nl2sql.security.rate_limit import RateLimiter

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, container: Container | None = None) -> FastAPI:
    """Build the application."""
    active_settings = settings or (container.settings if container else load_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = container or Container.build(active_settings)
        app.state.container = built
        app.state.rate_limiter = RateLimiter(active_settings.api.rate_limit_per_minute)
        logger.info(
            "application_started",
            env=active_settings.env,
            auth_mode=active_settings.api.auth_mode,
        )
        try:
            await built.warmup()
            yield
        finally:
            if container is None:
                built.dispose()
            logger.info("application_stopped")

    app = FastAPI(
        title=active_settings.api.title,
        version=active_settings.app.version,
        docs_url="/docs" if active_settings.api.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if active_settings.api.docs_enabled else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        RequestContextMiddleware, max_request_bytes=active_settings.api.max_request_bytes
    )
    if active_settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(active_settings.api.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(query.router)
    app.include_router(schema.router)
    app.include_router(health.router)
    return app
