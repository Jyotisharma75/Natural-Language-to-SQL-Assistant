"""Tests against the real services.

Each test is skipped unless its resource is configured. What they prove is
exactly what a double cannot: that the ODBC connection, the Entra token, the
reflection queries, the deployment name, the structured output contract and
the local model all work as configured.
"""

from __future__ import annotations

import pytest

from nl2sql.container import Container
from nl2sql.db.engine import create_query_engine
from nl2sql.llm.azure_openai import AzureOpenAIProvider
from nl2sql.llm.huggingface_local import HuggingFaceLocalProvider
from nl2sql.llm.usage import UsageTracker
from nl2sql.metadata.introspector import SchemaIntrospector
from nl2sql.pipeline.models import GeneratedSQL
from nl2sql.pipeline.orchestrator import QueryCommand
from nl2sql.security.auth import ROLE_READER, Principal
from nl2sql.security.policy import AccessPolicy


# -- Azure SQL --------------------------------------------------------------
@pytest.mark.live_azure_sql
def test_azure_sql_is_reachable(azure_sql_settings):
    """The configured connection opens and answers."""
    engine = create_query_engine(azure_sql_settings.database)
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT 1").scalar() == 1
    finally:
        engine.dispose()


@pytest.mark.live_azure_sql
def test_schema_is_discovered_from_azure_sql(azure_sql_settings):
    """Reflection returns tables, columns and relationships from the real database."""
    engine = create_query_engine(azure_sql_settings.database)
    policy = AccessPolicy(azure_sql_settings.security)
    try:
        introspector = SchemaIntrospector(engine, schema_filter=policy.is_schema_allowed)
        catalog = introspector.introspect()
    finally:
        engine.dispose()

    assert len(catalog) > 0, "no tables were discovered; check the schema allowlist and grants"
    table = catalog.tables[0]
    assert table.columns
    assert catalog.dialect == "mssql"


@pytest.mark.live_azure_sql
def test_the_service_account_cannot_write(azure_sql_settings):
    """The login the service uses must not be able to change anything.

    A failure here means the database grant is wrong. The application would
    still refuse the statement, but defence in depth means the credential
    should not carry the permission in the first place.
    """
    import sqlalchemy

    engine = create_query_engine(azure_sql_settings.database)
    table_name = "nl2sql_write_probe"
    try:
        with engine.connect() as connection, pytest.raises(sqlalchemy.exc.SQLAlchemyError):
            connection.exec_driver_sql(f"CREATE TABLE {table_name} (id INT)")
            connection.commit()
    finally:
        engine.dispose()


@pytest.mark.live_azure_sql
async def test_a_question_runs_end_to_end_against_azure_sql(azure_sql_settings):
    """The whole pipeline answers a question using the real database."""
    container = Container.build(azure_sql_settings, configure_logs=False)
    try:
        if not container.providers.available_names():
            pytest.skip("No language model is configured for an end to end run.")
        catalog = await container.metadata.catalog_async()
        assert len(catalog) > 0
        outcome = await container.pipeline.run(
            QueryCommand(
                question="How many rows are in the largest table?",
                principal=Principal(id="live-test", tenant_id=None, roles=(ROLE_READER,)),
                validate_only=True,
            )
        )
        assert outcome.sql
        assert outcome.valid is True
    finally:
        container.dispose()


# -- Azure OpenAI -----------------------------------------------------------
@pytest.mark.live_azure_openai
async def test_azure_openai_returns_structured_sql(azure_openai_settings, tmp_path):
    """The deployment honours the structured output contract."""
    from nl2sql.llm.prompts import PromptRegistry

    prompts = PromptRegistry(
        azure_openai_settings.prompts.directory,
        active_versions=azure_openai_settings.prompts.versions,
    )
    provider = AzureOpenAIProvider(
        azure_openai_settings.llm.azure_openai,
        usage=UsageTracker(),
        format_prompt=prompts.get("structured_output"),
    )
    assert provider.is_available() is True

    prompt = prompts.render(
        "sql_generation",
        dialect="Microsoft SQL Server (Azure SQL) T-SQL",
        schema=(
            "TABLE dbo.facilities\n  facility_id INTEGER PRIMARY KEY\n  facility_name VARCHAR(100)"
        ),
        question="How many facilities are there?",
        intent="aggregation",
        conversation="No earlier turns.",
        max_rows=100,
        today="2026-02-15",
    )
    result = await provider.generate_structured(prompt, GeneratedSQL)

    assert result.output.sql.upper().startswith("SELECT")
    assert "facilities" in result.output.sql.casefold()
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0


@pytest.mark.live_azure_openai
async def test_azure_openai_health_reports_reachable(azure_openai_settings):
    """Health reports the deployment as usable."""
    provider = AzureOpenAIProvider(
        azure_openai_settings.llm.azure_openai.model_copy(update={"health_probe": True}),
        usage=UsageTracker(),
    )
    health = await provider.health()
    assert health.available is True


# -- the local model --------------------------------------------------------
@pytest.mark.local_model
async def test_local_model_loads_and_generates(local_model_settings):
    """The configured local model downloads, loads and returns valid JSON."""
    from nl2sql.llm.prompts import PromptRegistry

    prompts = PromptRegistry(
        local_model_settings.prompts.directory,
        active_versions=local_model_settings.prompts.versions,
    )
    provider = HuggingFaceLocalProvider(
        local_model_settings.llm.local.model_copy(update={"enabled": True}),
        usage=UsageTracker(),
        format_prompt=prompts.get("structured_output"),
    )
    assert provider.is_available() is True

    await provider.preload()
    assert provider.is_loaded is True

    prompt = prompts.render(
        "sql_generation",
        dialect="SQLite",
        schema=(
            "TABLE main.facilities\n  facility_id INTEGER PRIMARY KEY\n  facility_name VARCHAR(100)"
        ),
        question="How many facilities are there?",
        intent="aggregation",
        conversation="No earlier turns.",
        max_rows=50,
        today="2026-02-15",
    )
    result = await provider.generate_structured(prompt, GeneratedSQL)

    assert isinstance(result.output, GeneratedSQL)
    assert result.usage.prompt_tokens > 0


@pytest.mark.local_model
async def test_local_model_output_is_validated_like_any_other(local_model_settings, tmp_path):
    """Whatever the small model produces still passes through the full validator."""
    import sys

    sys.path.insert(0, str(tmp_path))
    from tests.support.schema_fixture import build_sample_database

    database = build_sample_database()
    llm = local_model_settings.llm.model_copy(update={"default_provider": "local"})
    container = Container.build(
        local_model_settings.model_copy(update={"llm": llm}),
        query_engine=database.engine,
        configure_logs=False,
    )
    try:
        outcome = await container.pipeline.run(
            QueryCommand(
                question="How many facilities are there?",
                principal=Principal(id="live-test", tenant_id=None, roles=(ROLE_READER,)),
                validate_only=True,
            )
        )
        # The point is not that a 0.5B model is always right, it is that
        # whatever it produced was validated before it could run.
        assert outcome.sql
        assert outcome.valid is True
    finally:
        container.dispose()
        database.dispose()
