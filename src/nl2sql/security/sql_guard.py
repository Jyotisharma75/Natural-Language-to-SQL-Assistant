"""SQL safety guard.

This is the layer that decides whether a statement is allowed to exist at all,
before any question of which tables it touches. It works on the parsed syntax
tree rather than on the text, because text matching is defeated by comments,
casing, whitespace and unicode, while a parser sees what the database will see.

What it enforces:

* the statement parses, and there is exactly one of them, so stacked
  statements such as ``SELECT 1; DROP TABLE t`` are rejected
* the statement kind is on the configured allowlist, which by default holds
  SELECT alone, so INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE,
  MERGE and EXEC are refused unless an operator deliberately enables them
* no nested statement of a forbidden kind hides inside a CTE or subquery
* ``SELECT ... INTO`` is refused, because it writes a table
* no blocked function is called, which covers OPENROWSET, OPENQUERY,
  OPENDATASOURCE and the extended stored procedures
* no variables, session parameters such as ``@@VERSION``, placeholders, temp
  tables, table valued functions, cross database or linked server names
* a raw token scan as defence in depth, so a construct this version of the
  parser does not model cannot slip through as an opaque command

Everything it consults, including the blocked function, keyword and prefix
lists, comes from configuration.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import TokenType, exp
from sqlglot.errors import SqlglotError

from nl2sql.config.settings import SecuritySettings
from nl2sql.security.policy import AccessPolicy

#: Reserved prefix for the parameter markers this application injects. Model
#: output containing it is rejected, so generated text can never be mistaken
#: for a marker added during rewriting.
RESERVED_PARAMETER_PREFIX = "__NL2SQL_P"

#: Token kinds whose text is data or a quoted name, never a keyword.
_LITERAL_TOKENS = frozenset(
    {
        TokenType.STRING,
        TokenType.NATIONAL_STRING,
        TokenType.RAW_STRING,
        TokenType.HEREDOC_STRING,
        TokenType.UNICODE_STRING,
        TokenType.BYTE_STRING,
        TokenType.HEX_STRING,
        TokenType.NUMBER,
        TokenType.IDENTIFIER,
    }
)


def _statement_nodes() -> dict[str, tuple[type[exp.Expr], ...]]:
    """Map a statement kind onto the node types that represent it."""
    mapping: dict[str, tuple[type[exp.Expr], ...]] = {
        "SELECT": (exp.Select, exp.SetOperation, exp.Subquery),
        "INSERT": (exp.Insert,),
        "UPDATE": (exp.Update,),
        "DELETE": (exp.Delete,),
        "MERGE": (exp.Merge,),
        "CREATE": (exp.Create,),
        "DROP": (exp.Drop,),
        "ALTER": (exp.Alter,),
        "TRUNCATE": (exp.TruncateTable,),
        "EXEC": (exp.Execute,),
        "DECLARE": (exp.Declare,),
    }
    return mapping


STATEMENT_NODES = _statement_nodes()

#: Node types that carry a statement of their own and must be checked wherever
#: they appear, including nested inside a query.
_WRITE_NODES: tuple[tuple[str, type[exp.Expr]], ...] = tuple(
    (kind, node) for kind, nodes in STATEMENT_NODES.items() if kind != "SELECT" for node in nodes
)


@dataclass(frozen=True, slots=True)
class GuardIssue:
    """One reason a statement was refused."""

    code: str
    message: str
    severity: str = "error"


@dataclass(slots=True)
class GuardResult:
    """The outcome of guarding one statement."""

    sql: str
    statement_kind: str | None = None
    expression: exp.Expr | None = None
    issues: list[GuardIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether the statement may proceed to the next validation stage."""
        return not any(issue.severity == "error" for issue in self.issues)


class SQLGuard:
    """Statement level safety checks on parsed SQL."""

    def __init__(self, settings: SecuritySettings, policy: AccessPolicy) -> None:
        self._settings = settings
        self._policy = policy
        self._blocked_functions = frozenset(f.upper() for f in settings.blocked_functions)
        self._blocked_prefixes = tuple(p.upper() for p in settings.blocked_identifier_prefixes)
        # A keyword that names a statement an operator has deliberately enabled
        # must not then be blocked by the raw token scan, otherwise enabling a
        # statement in configuration would have no effect. EXEC and EXECUTE are
        # the same statement spelled two ways.
        permitted = set(policy.allowed_statements)
        if "EXEC" in permitted:
            permitted.add("EXECUTE")
        if "TRUNCATE" in permitted:
            permitted.add("TRUNCATE TABLE")
        self._blocked_keywords = frozenset(
            k.upper() for k in settings.blocked_keywords if k.upper() not in permitted
        )

    def check(self, sql: str, *, dialect: str) -> GuardResult:
        """Parse ``sql`` and return every safety problem found."""
        text = (sql or "").strip()
        while text.endswith(";"):
            text = text[:-1].rstrip()

        result = GuardResult(sql=text)
        if not text:
            result.issues.append(GuardIssue("empty_statement", "No SQL statement was produced."))
            return result
        if len(text) > self._settings.max_sql_chars:
            result.issues.append(
                GuardIssue(
                    "statement_too_long",
                    f"The statement is longer than the configured limit of "
                    f"{self._settings.max_sql_chars} characters.",
                )
            )
            return result
        if RESERVED_PARAMETER_PREFIX in text.upper():
            result.issues.append(
                GuardIssue(
                    "reserved_marker_present",
                    "The statement contains a reserved internal parameter marker.",
                )
            )
            return result

        try:
            parsed = [statement for statement in sqlglot.parse(text, read=dialect) if statement]
        except SqlglotError as exc:
            result.issues.append(
                GuardIssue("syntax_error", f"The statement could not be parsed: {exc}")
            )
            return result

        if len(parsed) != 1 or isinstance(parsed[0], exp.Block):
            result.issues.append(
                GuardIssue(
                    "multiple_statements",
                    "Only a single statement may be executed. Batches and stacked "
                    "statements are refused.",
                )
            )
            return result

        root = parsed[0]
        result.expression = root
        kind = self.statement_kind(root)
        result.statement_kind = kind

        if not self._policy.is_statement_allowed(kind):
            result.issues.append(
                GuardIssue(
                    "statement_not_allowed",
                    f"{kind} statements are not permitted. Allowed statements: "
                    f"{', '.join(sorted(self._policy.allowed_statements))}.",
                )
            )
            return result

        self._check_nested_statements(root, result)
        self._check_select_into(root, result)
        self._check_functions(root, result)
        self._check_tables(root, result)
        self._check_variables_and_placeholders(root, result)
        self._check_tokens(text, dialect, result)
        return result

    # -- individual checks -------------------------------------------------
    @staticmethod
    def statement_kind(node: exp.Expr) -> str:
        """Return the statement kind of a parsed node."""
        for kind, node_types in STATEMENT_NODES.items():
            if isinstance(node, node_types):
                return kind
        if isinstance(node, exp.Command):
            command = str(node.this or "COMMAND").upper()
            return command.split()[0] if command.strip() else "COMMAND"
        return type(node).__name__.upper()

    def _check_nested_statements(self, root: exp.Expr, result: GuardResult) -> None:
        for kind, node_type in _WRITE_NODES:
            if self._policy.is_statement_allowed(kind):
                continue
            for node in root.find_all(node_type):
                if node is root:
                    continue
                result.issues.append(
                    GuardIssue(
                        "nested_statement_not_allowed",
                        f"A nested {kind} statement is not permitted.",
                    )
                )
                return
        for node in root.find_all(exp.Command):
            command = str(node.this or "").upper().split()[:1]
            name = command[0] if command else "COMMAND"
            if not self._policy.is_statement_allowed(name):
                result.issues.append(
                    GuardIssue(
                        "command_not_allowed",
                        f"The command {name} is not permitted.",
                    )
                )
                return

    def _check_select_into(self, root: exp.Expr, result: GuardResult) -> None:
        for select in root.find_all(exp.Select):
            if select.args.get("into") is not None:
                result.issues.append(
                    GuardIssue(
                        "select_into_not_allowed",
                        "SELECT INTO creates a table and is not permitted.",
                    )
                )
                return

    def _check_functions(self, root: exp.Expr, result: GuardResult) -> None:
        for node in root.find_all(exp.Func):
            names = self._function_names(node)
            blocked = names & self._blocked_functions
            if blocked:
                result.issues.append(
                    GuardIssue(
                        "function_not_allowed",
                        f"The function {sorted(blocked)[0]} is not permitted.",
                    )
                )
                return
            if any(name.startswith(self._blocked_prefixes) for name in names):
                result.issues.append(
                    GuardIssue(
                        "function_not_allowed",
                        "A routine referenced by this query is not permitted.",
                    )
                )
                return

    @staticmethod
    def _function_names(node: exp.Func) -> frozenset[str]:
        """Return every name a function node could be known by, upper cased.

        A parser normalises some vendor functions onto a canonical node, for
        example T-SQL ``SUSER_SNAME()`` becomes ``CURRENT_USER``. Collecting
        the aliases means a blocklist entry works whichever spelling was used.
        """
        names: set[str] = set()
        if isinstance(node, exp.Anonymous) and node.this:
            names.add(str(node.this).upper())
        with contextlib.suppress(Exception):
            # Not every node type exposes a SQL name; the class name below
            # still identifies it.
            names.add(str(node.sql_name()).upper())
        names.add(type(node).__name__.upper())
        return frozenset(name for name in names if name)

    def _check_tables(self, root: exp.Expr, result: GuardResult) -> None:
        for table in root.find_all(exp.Table):
            if isinstance(table.this, exp.Func | exp.Anonymous):
                result.issues.append(
                    GuardIssue(
                        "table_function_not_allowed",
                        "Table valued functions are not permitted.",
                    )
                )
                return
            if isinstance(table.this, exp.Dot) or (
                table.args.get("catalog") and not self._settings.allow_cross_database
            ):
                result.issues.append(
                    GuardIssue(
                        "cross_database_reference",
                        "Cross database and linked server references are not permitted.",
                    )
                )
                return
            name = table.name or ""
            # A parser marks a T-SQL temp table on the identifier rather than
            # keeping the hash in the name, so both are checked.
            is_temporary = bool(
                isinstance(table.this, exp.Identifier) and table.this.args.get("temporary")
            )
            if not self._settings.allow_temp_tables and (
                is_temporary or name.startswith(("#", "@"))
            ):
                result.issues.append(
                    GuardIssue(
                        "temp_table_not_allowed",
                        "Temporary tables and table variables are not permitted.",
                    )
                )
                return
            schema = table.db or ""
            if schema and self._policy.is_system_schema(schema):
                result.issues.append(
                    GuardIssue(
                        "system_catalog_not_allowed",
                        "Database catalogue objects are not readable through this service.",
                    )
                )
                return

    def _check_variables_and_placeholders(self, root: exp.Expr, result: GuardResult) -> None:
        if not self._settings.allow_variables:
            for node_type in (exp.Parameter, exp.SessionParameter):
                if any(True for _ in root.find_all(node_type)):
                    result.issues.append(
                        GuardIssue(
                            "variables_not_allowed",
                            "Variables and session parameters are not permitted.",
                        )
                    )
                    return
        if any(True for _ in root.find_all(exp.Placeholder)):
            result.issues.append(
                GuardIssue(
                    "placeholder_not_allowed",
                    "Parameter placeholders may not be produced by the model.",
                )
            )

    def _check_tokens(self, text: str, dialect: str, result: GuardResult) -> None:
        """Scan raw tokens for blocked keywords the parser may have absorbed."""
        try:
            tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(text)
        except SqlglotError:
            return
        for token in tokens:
            if token.token_type in _LITERAL_TOKENS:
                continue
            word = token.text.upper()
            if word in self._blocked_keywords:
                result.issues.append(
                    GuardIssue(
                        "blocked_keyword",
                        f"The keyword {word} is not permitted in a query.",
                    )
                )
                return
            if any(word.startswith(prefix) for prefix in self._blocked_prefixes):
                result.issues.append(
                    GuardIssue(
                        "blocked_identifier",
                        "The statement references a routine that is not permitted.",
                    )
                )
                return


def render(expression: exp.Expr, dialect: str, **kwargs: Any) -> str:
    """Render an expression back to SQL text without comments."""
    return expression.sql(dialect=dialect, comments=False, **kwargs)
