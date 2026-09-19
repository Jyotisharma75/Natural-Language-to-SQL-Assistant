"""Fixtures for tests that need something real.

These tests are skipped unless the corresponding resource is configured, so
the default suite stays offline and fast. They are the only tests that reach
Azure SQL, Azure OpenAI or a downloaded model, and they are what proves the
adapters work against the genuine services rather than against a double.

    pytest tests/live -m live_azure_sql
    pytest tests/live -m live_azure_openai
    pytest tests/live -m local_model
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from nl2sql.config.loader import load_settings
from nl2sql.config.secrets import resolve_secret

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def live_settings() -> Any:
    """Load the development configuration with the real .env applied."""
    return load_settings(
        env=os.getenv("NL2SQL_ENV", "development"),
        config_dir=REPO_ROOT / "configs",
        load_dotenv_file=True,
    )


@pytest.fixture
def azure_sql_settings(live_settings: Any) -> Any:
    """Skip unless an Azure SQL connection is configured."""
    database = live_settings.database
    configured = bool(
        database.url or database.url_secret or (database.server and database.database)
    )
    if not configured:
        pytest.skip(
            "Azure SQL is not configured. Set NL2SQL_DATABASE__SERVER and "
            "NL2SQL_DATABASE__DATABASE in .env to run this test."
        )
    return live_settings


@pytest.fixture
def azure_openai_settings(live_settings: Any) -> Any:
    """Skip unless an Azure OpenAI deployment is configured."""
    azure = live_settings.llm.azure_openai
    has_credential = azure.use_managed_identity or bool(resolve_secret(azure.api_key_secret))
    if not (azure.endpoint and azure.deployment and has_credential):
        pytest.skip(
            "Azure OpenAI is not configured. Set NL2SQL_LLM__AZURE_OPENAI__ENDPOINT, "
            "NL2SQL_LLM__AZURE_OPENAI__DEPLOYMENT and AZURE_OPENAI_API_KEY in .env."
        )
    return live_settings


@pytest.fixture
def local_model_settings(live_settings: Any) -> Any:
    """Skip unless the local model extra is installed and enabled."""
    import importlib.util

    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
        pytest.skip("The local model extra is not installed. Install with pip install '.[local]'.")
    if os.getenv("NL2SQL_RUN_LOCAL_MODEL_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set NL2SQL_RUN_LOCAL_MODEL_TESTS=1 to download and run the local model.")
    return live_settings
