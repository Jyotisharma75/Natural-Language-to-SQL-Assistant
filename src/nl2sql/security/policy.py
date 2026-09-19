"""The access policy: which schemas, tables and columns exist as far as the
assistant is concerned.

The policy is applied at the metadata layer, before anything reaches a prompt,
the validator or an API response. That ordering is the point: a blocked column
is not something the model is asked politely to avoid, it is something the
model is never told about and the validator will reject if it invents it.

Patterns are shell style globs matched case insensitively against
``schema.table`` or ``schema.table.column``. A pattern with fewer dotted parts
than the subject matches on the trailing parts, so ``*password*`` blocks a
column of that name in every table, and ``hr.*`` blocks a whole schema.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from nl2sql.config.settings import SecuritySettings


def _matches(pattern: str, parts: tuple[str, ...]) -> bool:
    """Return whether a dotted glob pattern matches the trailing parts of a name."""
    pattern_parts = tuple(p.casefold() for p in pattern.split("."))
    subject = tuple(p.casefold() for p in parts)
    if len(pattern_parts) > len(subject):
        return False
    tail = subject[-len(pattern_parts) :]
    return all(fnmatchcase(value, rule) for value, rule in zip(tail, pattern_parts, strict=True))


class AccessPolicy:
    """Decides what the caller may see and what statements may run."""

    def __init__(self, settings: SecuritySettings) -> None:
        self._settings = settings
        self._allowed_schemas = tuple(settings.allowed_schemas)
        self._system_schemas = frozenset(s.casefold() for s in settings.system_schemas)
        self._blocked_tables = tuple(settings.blocked_tables)
        self._blocked_columns = tuple(settings.blocked_columns)
        self._masked_columns = tuple(settings.masked_columns)
        self._allowed_statements = frozenset(settings.allowed_statements)

    @property
    def settings(self) -> SecuritySettings:
        """Return the settings this policy was built from."""
        return self._settings

    @property
    def allowed_statements(self) -> frozenset[str]:
        """Return the statement kinds that may be executed."""
        return self._allowed_statements

    def is_system_schema(self, schema: str) -> bool:
        """Return whether ``schema`` is a database catalogue schema."""
        return schema.casefold() in self._system_schemas

    def is_schema_allowed(self, schema: str) -> bool:
        """Return whether a schema is visible to the assistant."""
        if self.is_system_schema(schema) and not self._settings.allow_system_catalogs:
            return False
        if not self._allowed_schemas:
            return True
        return any(fnmatchcase(schema.casefold(), p.casefold()) for p in self._allowed_schemas)

    def is_table_allowed(self, schema: str, table: str) -> bool:
        """Return whether a table is visible to the assistant."""
        if not self.is_schema_allowed(schema):
            return False
        return not any(_matches(p, (schema, table)) for p in self._blocked_tables)

    def is_column_allowed(self, schema: str, table: str, column: str) -> bool:
        """Return whether a column is visible to the assistant."""
        if not self.is_table_allowed(schema, table):
            return False
        return not any(_matches(p, (schema, table, column)) for p in self._blocked_columns)

    def is_column_masked(self, schema: str, table: str, column: str) -> bool:
        """Return whether a column's values must be masked in results."""
        return any(_matches(p, (schema, table, column)) for p in self._masked_columns)

    def is_statement_allowed(self, kind: str) -> bool:
        """Return whether a statement kind may be executed."""
        return kind.upper() in self._allowed_statements
