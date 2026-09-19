"""Alembic environment.

Migrations apply only to the application owned tables. The database that
questions are answered from is never migrated by this service: it belongs to
whoever owns the data, and this service holds read permission on it.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from nl2sql.config.loader import load_settings
from nl2sql.config.secrets import resolve_secret
from nl2sql.db.app_models import AppBase
from nl2sql.db.engine import prepare_sqlite_path

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = AppBase.metadata


def _database_url() -> str:
    """Return the application database URL from configuration."""
    settings = load_settings()
    url = resolve_secret(settings.app_database.url_secret) or settings.app_database.url
    # A file backed SQLite database needs its directory to exist before the
    # first connection, which is the common case for a local upgrade.
    prepare_sqlite_path(url)
    return url


def run_migrations_offline() -> None:
    """Emit SQL without connecting."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot alter a column in place, so changes are applied by
            # rebuilding the table. This is a no op on SQL Server.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
