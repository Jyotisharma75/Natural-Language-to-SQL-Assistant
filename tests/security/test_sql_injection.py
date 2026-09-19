"""The attack corpus.

Every statement here is something a model could be induced to emit, whether by
a hostile question, a poisoned table comment or simply by being wrong. None of
them may reach the database.

The corpus is checked through the validator rather than the guard alone, so
what is asserted is the behaviour of the whole safety path, which is what
actually protects the database.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.security

DESTRUCTIVE = [
    ("insert", "INSERT INTO facilities (facility_name) VALUES ('x')"),
    ("update", "UPDATE facilities SET facility_name = 'x'"),
    ("delete", "DELETE FROM facilities"),
    ("drop", "DROP TABLE facilities"),
    ("alter", "ALTER TABLE facilities ADD extra INTEGER"),
    ("truncate", "TRUNCATE TABLE facilities"),
    ("create", "CREATE TABLE evil (a INTEGER)"),
    ("create_view", "CREATE VIEW evil AS SELECT facility_name FROM facilities"),
    ("merge", "MERGE INTO facilities USING regions ON 1 = 1 WHEN MATCHED THEN DELETE"),
    ("exec", "EXEC xp_cmdshell 'dir'"),
    ("execute", "EXECUTE sp_executesql N'SELECT 1'"),
    ("grant", "GRANT SELECT ON facilities TO public"),
    ("select_into", "SELECT facility_name INTO copied FROM facilities"),
]

STACKED = [
    ("stacked_drop", "SELECT facility_name FROM facilities; DROP TABLE facilities"),
    ("stacked_delete", "SELECT 1; DELETE FROM facilities"),
    (
        "stacked_comment",
        "SELECT facility_name FROM facilities; -- harmless\nDROP TABLE regions",
    ),
]

EXFILTRATION = [
    ("openrowset", "SELECT a FROM OPENROWSET('SQLNCLI', 'Server=evil;', 'SELECT 1') AS x"),
    ("openquery", "SELECT a FROM OPENQUERY(linked, 'SELECT 1') AS x"),
    ("opendatasource", "SELECT a FROM OPENDATASOURCE('SQLNCLI', 'Server=evil;').db.dbo.t"),
    ("linked_server", "SELECT facility_name FROM evil.remote.dbo.facilities"),
    ("system_catalog", "SELECT name FROM sys.tables"),
    ("information_schema", "SELECT table_name FROM INFORMATION_SCHEMA.TABLES"),
    ("session_parameter", "SELECT @@VERSION AS version"),
    ("identity_function", "SELECT SUSER_SNAME() AS who"),
    ("temp_table", "SELECT facility_name FROM #staging"),
    ("variable", "SELECT facility_name FROM facilities WHERE facility_id = @injected"),
]

BLOCKED_DATA = [
    ("blocked_column", "SELECT password_hash FROM user_credentials"),
    ("blocked_column_aliased", "SELECT u.api_key AS k FROM user_credentials u"),
    ("blocked_column_in_where", "SELECT username FROM user_credentials WHERE password_hash = 'x'"),
    ("unknown_table", "SELECT secret FROM payroll"),
    ("select_star", "SELECT * FROM facilities"),
    ("select_star_qualified", "SELECT f.* FROM facilities f"),
    ("cartesian", "SELECT f.facility_name, r.region_name FROM facilities f, regions r"),
    ("cross_join", "SELECT f.facility_name, r.region_name FROM facilities f CROSS JOIN regions r"),
]


@pytest.mark.parametrize(("label", "sql"), DESTRUCTIVE, ids=[item[0] for item in DESTRUCTIVE])
def test_destructive_statements_are_refused(validator, catalog, label, sql):
    """No statement that writes or changes structure may pass."""
    report = validator.validate(sql, catalog, dialect="tsql")
    assert report.valid is False
    assert report.errors


@pytest.mark.parametrize(("label", "sql"), STACKED, ids=[item[0] for item in STACKED])
def test_stacked_statements_are_refused(validator, catalog, label, sql):
    """A second statement after a semicolon is refused, comment or not."""
    report = validator.validate(sql, catalog, dialect="tsql")
    assert report.valid is False
    assert any(issue.code == "multiple_statements" for issue in report.errors)


@pytest.mark.parametrize(("label", "sql"), EXFILTRATION, ids=[item[0] for item in EXFILTRATION])
def test_exfiltration_attempts_are_refused(validator, catalog, label, sql):
    """Remote sources, catalogue views, variables and temp tables are refused."""
    report = validator.validate(sql, catalog, dialect="tsql")
    assert report.valid is False
    assert report.errors


@pytest.mark.parametrize(("label", "sql"), BLOCKED_DATA, ids=[item[0] for item in BLOCKED_DATA])
def test_restricted_data_access_is_refused(validator, catalog, label, sql):
    """Hidden columns, unknown tables, stars and cartesian joins are refused."""
    report = validator.validate(sql, catalog, dialect="sqlite")
    assert report.valid is False
    assert report.errors


def test_refusal_messages_do_not_disclose_hidden_objects(validator, catalog):
    """A refusal does not reveal whether the object exists or is merely hidden."""
    hidden = validator.validate(
        "SELECT password_hash FROM user_credentials", catalog, dialect="sqlite"
    )
    absent = validator.validate(
        "SELECT nonexistent FROM user_credentials", catalog, dialect="sqlite"
    )
    hidden_messages = [issue.message for issue in hidden.errors]
    absent_messages = [issue.message for issue in absent.errors]
    assert hidden_messages and absent_messages
    assert "blocked" not in " ".join(hidden_messages).lower()
    assert "restricted" not in " ".join(hidden_messages).lower()


def test_union_to_a_hidden_column_is_refused(validator, catalog):
    """A union cannot be used to reach a column the policy hides."""
    report = validator.validate(
        "SELECT facility_name FROM facilities UNION ALL SELECT password_hash FROM user_credentials",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False


def test_subquery_to_a_hidden_column_is_refused(validator, catalog):
    """Nesting does not hide a reference to a blocked column."""
    report = validator.validate(
        "SELECT facility_name FROM facilities WHERE facility_name IN "
        "(SELECT password_hash FROM user_credentials)",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False


def test_comment_hidden_dml_is_refused(validator, catalog):
    """A comment cannot smuggle a second statement past the parser."""
    report = validator.validate(
        "SELECT facility_name FROM facilities /* */; /* */ DROP TABLE facilities",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False


def test_reserved_parameter_marker_in_model_output_is_refused(validator, catalog):
    """Model output may not contain the marker the rewriter uses for parameters."""
    report = validator.validate(
        "SELECT facility_name FROM facilities WHERE tenant_id = __NL2SQL_PTENANT__",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False
    assert any(issue.code == "reserved_marker_present" for issue in report.errors)


def test_a_legitimate_query_still_passes(validator, catalog):
    """The corpus must not be passing by refusing everything."""
    report = validator.validate(
        "SELECT f.facility_name AS facility_name, SUM(e.energy_kwh) AS total_kwh "
        "FROM energy_readings e JOIN facilities f ON f.facility_id = e.facility_id "
        "GROUP BY f.facility_name",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is True
    assert report.errors == ()
