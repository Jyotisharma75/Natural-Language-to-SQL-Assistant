"""SQL validation: what passes, what is rewritten and what is refused."""

from __future__ import annotations

import pytest

from nl2sql.pipeline.models import QueryFilter

pytestmark = pytest.mark.sql_validation

ALLOWED = [
    ("plain select", "SELECT facility_name FROM facilities"),
    ("qualified", "SELECT f.facility_name FROM main.facilities f"),
    ("aggregate", "SELECT COUNT(*) AS n FROM facilities"),
    (
        "join",
        "SELECT f.facility_name, r.region_name FROM facilities f "
        "JOIN regions r ON r.region_id = f.region_id",
    ),
    (
        "group by with alias order",
        "SELECT f.facility_name AS name, SUM(e.energy_kwh) AS total FROM energy_readings e "
        "JOIN facilities f ON f.facility_id = e.facility_id GROUP BY f.facility_name "
        "ORDER BY total DESC",
    ),
    (
        "cte",
        "WITH totals AS (SELECT facility_id, SUM(energy_kwh) AS total FROM energy_readings "
        "GROUP BY facility_id) SELECT f.facility_name, t.total FROM totals t "
        "JOIN facilities f ON f.facility_id = t.facility_id",
    ),
    (
        "scalar subquery",
        "SELECT facility_name FROM facilities WHERE facility_id = "
        "(SELECT MAX(facility_id) FROM energy_readings)",
    ),
    (
        "union",
        "SELECT facility_name FROM facilities UNION SELECT region_name FROM regions",
    ),
    (
        "case expression",
        "SELECT facility_name, CASE WHEN status = 'active' THEN 1 ELSE 0 END AS is_active "
        "FROM facilities",
    ),
    (
        "having",
        "SELECT facility_id, SUM(energy_kwh) AS total FROM energy_readings "
        "GROUP BY facility_id HAVING SUM(energy_kwh) > 100",
    ),
]

REFUSED = [
    ("unknown column", "SELECT no_such_column FROM facilities", "column_not_allowed"),
    ("unknown table", "SELECT a FROM no_such_table", "table_not_allowed"),
    ("hidden column", "SELECT password_hash FROM user_credentials", "column_not_allowed"),
    ("star", "SELECT * FROM facilities", "select_star_not_allowed"),
    ("write", "DELETE FROM facilities", "statement_not_allowed"),
    ("two statements", "SELECT 1; SELECT 2", "multiple_statements"),
    ("not sql", "this is not sql at all", "syntax_error"),
]


@pytest.mark.parametrize(("label", "sql"), ALLOWED, ids=[item[0] for item in ALLOWED])
def test_valid_queries_pass(validator, catalog, label, sql):
    """A safe, resolvable read only query is accepted."""
    report = validator.validate(sql, catalog, dialect="sqlite")
    assert report.valid is True, [issue.message for issue in report.errors]


@pytest.mark.parametrize(("label", "sql", "code"), REFUSED, ids=[item[0] for item in REFUSED])
def test_invalid_queries_are_refused_with_a_reason(validator, catalog, label, sql, code):
    """A refusal names the specific reason, which is what repair depends on."""
    report = validator.validate(sql, catalog, dialect="sqlite")
    assert report.valid is False
    assert code in {issue.code for issue in report.errors}


def test_row_limit_is_added_when_missing(validator, catalog, settings):
    """An unbounded query is bounded, and one extra row is fetched to spot truncation."""
    report = validator.validate("SELECT facility_name FROM facilities", catalog, dialect="sqlite")
    assert report.row_limit_applied == settings.limits.max_rows
    assert f"LIMIT {settings.limits.max_rows}" in report.display_sql
    assert f"LIMIT {settings.limits.max_rows + 1}" in report.execution_sql


def test_row_limit_is_clamped_when_too_large(validator, catalog, settings):
    """A model asking for more rows than the ceiling gets the ceiling."""
    report = validator.validate(
        "SELECT facility_name FROM facilities LIMIT 100000", catalog, dialect="sqlite"
    )
    assert report.row_limit_applied == settings.limits.max_rows


def test_smaller_limit_is_respected(validator, catalog):
    """A top N question keeps its own smaller limit."""
    report = validator.validate(
        "SELECT facility_name FROM facilities ORDER BY facility_name LIMIT 3",
        catalog,
        dialect="sqlite",
    )
    assert report.row_limit_applied == 3
    assert report.display_sql == report.execution_sql


def test_join_ceiling_is_enforced(build_container, catalog):
    """A query with more joins than the ceiling is refused."""
    container = build_container(limits={"max_joins": 1})
    report = container.validator.validate(
        "SELECT f.facility_name, r.region_name, e.energy_kwh FROM facilities f "
        "JOIN regions r ON r.region_id = f.region_id "
        "JOIN energy_readings e ON e.facility_id = f.facility_id",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False
    assert "too_many_joins" in {issue.code for issue in report.errors}


def test_table_ceiling_is_enforced(build_container, catalog):
    """A query touching more tables than the ceiling is refused."""
    container = build_container(limits={"max_tables": 1})
    report = container.validator.validate(
        "SELECT f.facility_name, r.region_name FROM facilities f "
        "JOIN regions r ON r.region_id = f.region_id",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False
    assert "too_many_tables" in {issue.code for issue in report.errors}


def test_subquery_depth_ceiling_is_enforced(build_container, catalog):
    """Deeply nested queries are refused."""
    container = build_container(limits={"max_subquery_depth": 1})
    report = container.validator.validate(
        "SELECT facility_name FROM facilities WHERE facility_id IN "
        "(SELECT facility_id FROM energy_readings WHERE facility_id IN "
        "(SELECT facility_id FROM emissions))",
        catalog,
        dialect="sqlite",
    )
    assert report.valid is False
    assert "subquery_too_deep" in {issue.code for issue in report.errors}


def test_filters_wrap_the_query_and_bind_values(validator, catalog):
    """Filters are applied to output columns as parameters."""
    report = validator.validate(
        "SELECT facility_name AS facility_name, status AS status FROM facilities",
        catalog,
        dialect="sqlite",
        filters=[
            QueryFilter(column="status", operator="eq", value="active"),
            QueryFilter(column="facility_name", operator="contains", value="Plant"),
        ],
    )
    assert report.valid is True
    assert "nl2sql_result" in report.execution_sql
    assert len(report.parameters) == 2
    assert "active" not in report.execution_sql


def test_filter_on_an_unknown_column_is_refused(validator, catalog):
    """A filter naming a column the query does not return is refused."""
    report = validator.validate(
        "SELECT facility_name FROM facilities",
        catalog,
        dialect="sqlite",
        filters=[QueryFilter(column="salary", operator="gt", value=1)],
    )
    assert report.valid is False
    assert "filter_column_unknown" in {issue.code for issue in report.errors}


def test_in_filter_binds_every_value(validator, catalog):
    """An in filter produces one marker per value."""
    report = validator.validate(
        "SELECT facility_name AS facility_name FROM facilities",
        catalog,
        dialect="sqlite",
        filters=[QueryFilter(column="facility_name", operator="in", value=["a", "b", "c"])],
    )
    assert report.valid is True
    assert len(report.parameters) == 3


def test_declared_tables_that_do_not_match_raise_a_warning(validator, catalog):
    """A model misreporting which tables it read is a warning, not a refusal."""
    report = validator.validate(
        "SELECT facility_name FROM facilities",
        catalog,
        dialect="sqlite",
        declared_tables=["main.emissions"],
    )
    assert report.valid is True
    assert "declared_tables_mismatch" in {issue.code for issue in report.warnings}


def test_report_lists_the_tables_and_columns_read(validator, catalog):
    """The report says what the query actually touches, for the audit trail."""
    report = validator.validate(
        "SELECT f.facility_name, r.region_name FROM facilities f "
        "JOIN regions r ON r.region_id = f.region_id",
        catalog,
        dialect="sqlite",
    )
    assert set(report.tables) == {"main.facilities", "main.regions"}
    assert "main.facilities.facility_name" in report.columns
    assert "main.regions.region_name" in report.columns
