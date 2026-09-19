"""Shared fixtures.

Tests build the real container over a real SQLite database and scripted model
providers. Nothing in the pipeline is stubbed out, so what the tests exercise
is the code that runs in production, with only the two genuinely external
things replaced: the database server and the model endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from nl2sql.api.app import create_app
from nl2sql.config.loader import load_settings
from nl2sql.config.settings import Settings
from nl2sql.container import Container
from nl2sql.db.repositories import create_all
from nl2sql.db.test_database import TestDatabase
from nl2sql.llm.factory import ProviderRegistry
from nl2sql.pipeline.models import GeneratedSQL, VerificationVerdict
from nl2sql.security.auth import Principal
from tests.support.fakes import ScriptedProvider
from tests.support.schema_fixture import REFERENCE_DATE, build_sample_database

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root."""
    return REPO_ROOT


@pytest.fixture
def settings() -> Settings:
    """Load the test configuration."""
    return load_settings(env="test", config_dir=REPO_ROOT / "configs", load_dotenv_file=False)


@pytest.fixture
def sample_db() -> Iterator[TestDatabase]:
    """Create the sample sustainability database."""
    database = build_sample_database()
    yield database
    database.dispose()


@pytest.fixture
def app_engine() -> Iterator[Engine]:
    """Create an in memory application database with the audit tables."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def azure_provider() -> ScriptedProvider:
    """A scripted stand in for the hosted model."""
    return ScriptedProvider(name_="azure_openai", model="gpt-scripted")


@pytest.fixture
def local_provider() -> ScriptedProvider:
    """A scripted stand in for the local model."""
    return ScriptedProvider(name_="local", model="qwen-scripted")


@pytest.fixture
def providers(
    azure_provider: ScriptedProvider, local_provider: ScriptedProvider
) -> ProviderRegistry:
    """A registry holding both scripted providers."""
    return ProviderRegistry({"azure_openai": azure_provider, "local": local_provider})


@pytest.fixture
def build_container(
    settings: Settings, sample_db: TestDatabase, app_engine: Engine, providers: ProviderRegistry
) -> Any:
    """Return a factory that builds a container with optional setting overrides."""

    def merge(current: Any, value: Any) -> Any:
        """Merge an override into a settings object rather than replacing it.

        Applied recursively, so an override such as
        ``execution={"retry": {"max_attempts": 3}}`` keeps every other field of
        both sections typed instead of turning one into a plain dict.
        """
        if not (isinstance(value, dict) and isinstance(current, BaseModel)):
            return value
        nested = {
            key: merge(getattr(current, key), item) if hasattr(current, key) else item
            for key, item in value.items()
        }
        return current.model_copy(update=nested, deep=True)

    def factory(**overrides: Any) -> Container:
        active = settings
        if overrides:
            merged = {key: merge(getattr(settings, key), value) for key, value in overrides.items()}
            active = settings.model_copy(update=merged, deep=True)
        return Container.build(
            active,
            query_engine=sample_db.engine,
            app_engine=app_engine,
            providers=providers,
            configure_logs=False,
        )

    return factory


@pytest.fixture
def container(build_container: Any) -> Container:
    """The default container for tests."""
    return build_container()


@pytest.fixture
def catalog(container: Container):
    """The policy filtered catalogue discovered from the sample database."""
    return container.metadata.catalog()


@pytest.fixture
def validator(container: Container):
    """The SQL validator wired to the sample database's policy."""
    return container.validator


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    """An HTTP client over the real application."""
    app = create_app(container.settings, container=container)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def principal() -> Principal:
    """A caller with the reader role."""
    return Principal(id="test-principal", tenant_id=None, roles=("reader",))


@pytest.fixture
def reference_date() -> Any:
    """The fixed date the sample data is built around."""
    return REFERENCE_DATE


def sql_response(sql: str, **overrides: Any) -> GeneratedSQL:
    """Build a scripted generation result."""
    payload: dict[str, Any] = {
        "sql": sql,
        "reasoning_summary": "Summarises the requested measure.",
        "tables_used": [],
        "columns_used": [],
        "confidence": 0.9,
        "warnings": [],
    }
    payload.update(overrides)
    return GeneratedSQL(**payload)


def verdict(agrees: bool = True, confidence: float = 0.9, issues: list[str] | None = None):
    """Build a scripted verification verdict."""
    return VerificationVerdict(agrees=agrees, confidence=confidence, issues=issues or [])
