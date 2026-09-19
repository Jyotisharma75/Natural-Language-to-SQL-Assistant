"""Engine construction.

Two engines exist and they are deliberately different things:

* the **query engine** reads the business database. It should be configured
  with a login that holds nothing but read permission, and it opens
  connections with ``ApplicationIntent=ReadOnly`` so a geo replicated database
  routes the work to a secondary replica.
* the **application engine** owns the audit and conversation tables, is
  managed by Alembic, and is the only engine the service writes through.

Connection details are assembled from settings. The ODBC connection string is
built from typed fields with proper escaping; no part of it comes from a
request, and the password is read from the environment at connect time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL

from nl2sql.config.secrets import require_secret, resolve_secret
from nl2sql.config.settings import AppDatabaseSettings, DatabaseSettings
from nl2sql.core.exceptions import ConfigurationError
from nl2sql.db.azure_auth import SQL_COPT_SS_ACCESS_TOKEN, AzureSQLTokenProvider
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)

#: SQLAlchemy dialect name mapped onto the sqlglot dialect that parses it.
_SQLGLOT_DIALECTS = {
    "mssql": "tsql",
    "sqlite": "sqlite",
    "postgresql": "postgres",
    "mysql": "mysql",
    "oracle": "oracle",
    "snowflake": "snowflake",
}

#: Human readable name of the SQL flavour, used in prompts.
_DIALECT_LABELS = {
    "mssql": "Microsoft SQL Server (Azure SQL) T-SQL",
    "sqlite": "SQLite",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
}


def sqlglot_dialect(dialect_name: str) -> str:
    """Return the sqlglot dialect for a SQLAlchemy dialect name."""
    return _SQLGLOT_DIALECTS.get(dialect_name, dialect_name)


def dialect_label(dialect_name: str) -> str:
    """Return the human readable SQL flavour name shown to a model."""
    return _DIALECT_LABELS.get(dialect_name, dialect_name)


def _odbc_value(value: str) -> str:
    """Escape a value for an ODBC connection string."""
    if any(character in value for character in "{};,="):
        return "{" + value.replace("}", "}}") + "}"
    return value


def build_odbc_connect_string(settings: DatabaseSettings) -> str:
    """Build the ODBC connection string for Azure SQL from typed settings."""
    if not settings.server or not settings.database:
        raise ConfigurationError(
            "Azure SQL requires database.server and database.database, or a full "
            "database.url. Set NL2SQL_DATABASE__SERVER and NL2SQL_DATABASE__DATABASE."
        )
    parts: list[tuple[str, str]] = [
        ("Driver", settings.driver),
        ("Server", f"tcp:{settings.server},{settings.port}"),
        ("Database", settings.database),
        ("Encrypt", "yes" if settings.encrypt else "no"),
        ("TrustServerCertificate", "yes" if settings.trust_server_certificate else "no"),
        ("Connection Timeout", str(settings.login_timeout_seconds)),
    ]
    if settings.application_intent_read_only:
        parts.append(("ApplicationIntent", "ReadOnly"))
    if settings.auth_mode == "sql_password":
        if not settings.username:
            raise ConfigurationError(
                "database.auth_mode is sql_password but database.username is not set."
            )
        password = require_secret(settings.password_secret, purpose="the Azure SQL login")
        parts.append(("Uid", settings.username))
        parts.append(("Pwd", password))
    for key, value in settings.odbc_extra.items():
        parts.append((key, value))
    return ";".join(f"{key}={_odbc_value(value)}" for key, value in parts)


def _resolve_url(url: str | None, url_secret: str | None) -> str | None:
    """Return an explicit URL from settings or from the named environment variable."""
    if url:
        return url
    return resolve_secret(url_secret)


def prepare_sqlite_path(url: str) -> None:
    """Create the parent directory of a SQLite file so the first connect succeeds."""
    if not url.startswith("sqlite"):
        return
    path = urlsplit(url).path.lstrip("/")
    if not path or path == ":memory:":
        return
    parent = Path(path).expanduser().parent
    if str(parent) not in {"", "."}:
        parent.mkdir(parents=True, exist_ok=True)


def create_query_engine(
    settings: DatabaseSettings,
    *,
    token_provider: AzureSQLTokenProvider | None = None,
) -> Engine:
    """Create the read only engine for the database being questioned."""
    explicit_url = _resolve_url(settings.url, settings.url_secret)
    if explicit_url:
        prepare_sqlite_path(explicit_url)
        engine = create_engine(explicit_url, pool_pre_ping=settings.pool_pre_ping, future=True)
        logger.info("query_engine_created", dialect=engine.dialect.name, source="url")
        return engine

    connect_string = build_odbc_connect_string(settings)
    url = URL.create("mssql+pyodbc", query={"odbc_connect": connect_string})
    engine = create_engine(
        url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_recycle=settings.pool_recycle_seconds,
        pool_pre_ping=settings.pool_pre_ping,
        future=True,
    )

    if settings.auth_mode != "sql_password":
        provider = token_provider or AzureSQLTokenProvider(settings)

        @event.listens_for(engine, "do_connect")
        def _provide_token(
            dialect: Any, conn_rec: Any, cargs: Any, cparams: dict[str, Any]
        ) -> None:
            cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: provider.token_struct()}

    logger.info(
        "query_engine_created",
        dialect="mssql",
        auth_mode=settings.auth_mode,
        read_only_intent=settings.application_intent_read_only,
    )
    return engine


def create_app_engine(settings: AppDatabaseSettings) -> Engine:
    """Create the engine for the application owned audit and history tables."""
    url = _resolve_url(settings.url, settings.url_secret)
    if not url:
        raise ConfigurationError(
            "No application database is configured. Set app_database.url or the "
            "environment variable named by app_database.url_secret."
        )
    prepare_sqlite_path(url)
    engine = create_engine(url, echo=settings.echo, pool_pre_ping=True, future=True)
    logger.info("app_engine_created", dialect=engine.dialect.name)
    return engine
