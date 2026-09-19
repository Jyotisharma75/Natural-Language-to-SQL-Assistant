"""The sample schema used by tests.

This is test data, not an assumption baked into the application. Nothing in
``src`` knows these tables exist: they are discovered by the same introspection
that runs against Azure SQL. The shape is deliberately realistic for the
sustainability questions in the README, so the tests exercise joins, dates,
aggregation and a masked column.
"""

from __future__ import annotations

from datetime import date, timedelta

from nl2sql.db.test_database import ColumnSpec, TableSpec, TestDatabase

#: A fixed reference date, so tests do not change behaviour as time passes.
REFERENCE_DATE = date(2026, 2, 15)


def _reading_rows() -> list[dict[str, object]]:
    """Build deterministic energy readings, including one clear outlier."""
    rows: list[dict[str, object]] = []
    reading_id = 1
    for facility_id in (1, 2, 3):
        for month in range(6):
            day = REFERENCE_DATE - timedelta(days=30 * month)
            base = 1000 + facility_id * 250 + month * 10
            # Facility 3 spikes in the most recent month, which is what an
            # abnormal usage question should find.
            value = base * 8 if (facility_id == 3 and month == 0) else base
            rows.append(
                {
                    "reading_id": reading_id,
                    "facility_id": facility_id,
                    "reading_date": day,
                    "energy_kwh": float(value),
                }
            )
            reading_id += 1
    return rows


def _emission_rows() -> list[dict[str, object]]:
    """Build deterministic emissions across two scopes."""
    rows: list[dict[str, object]] = []
    emission_id = 1
    for facility_id in (1, 2, 3):
        for month in range(6):
            for scope in ("Scope 1", "Scope 2"):
                rows.append(
                    {
                        "emission_id": emission_id,
                        "facility_id": facility_id,
                        "reporting_date": REFERENCE_DATE - timedelta(days=30 * month),
                        "scope": scope,
                        "co2e_tonnes": float(
                            10 * facility_id + month + (5 if scope == "Scope 1" else 0)
                        ),
                    }
                )
                emission_id += 1
    return rows


def sample_tables() -> list[TableSpec]:
    """Return the table specification for the sample sustainability schema."""
    return [
        TableSpec(
            name="regions",
            comment="Geographic reporting regions.",
            columns=(
                ColumnSpec("region_id", "int", primary_key=True),
                ColumnSpec("region_name", "varchar(80)", nullable=False, comment="Region name."),
            ),
            rows=(
                {"region_id": 1, "region_name": "Europe"},
                {"region_id": 2, "region_name": "North America"},
            ),
        ),
        TableSpec(
            name="facilities",
            comment="Operating sites that report energy and emissions.",
            columns=(
                ColumnSpec("facility_id", "int", primary_key=True),
                ColumnSpec(
                    "facility_name", "varchar(100)", nullable=False, comment="Facility name."
                ),
                ColumnSpec("region_id", "int", foreign_key="regions.region_id"),
                ColumnSpec("status", "varchar(20)", comment="active or closed."),
                ColumnSpec("contact_email", "varchar(200)", comment="Site contact, restricted."),
                ColumnSpec("tenant_id", "varchar(40)", comment="Owning tenant."),
            ),
            rows=(
                {
                    "facility_id": 1,
                    "facility_name": "Rotterdam Plant",
                    "region_id": 1,
                    "status": "active",
                    "contact_email": "ops.rotterdam@example.invalid",
                    "tenant_id": "acme",
                },
                {
                    "facility_id": 2,
                    "facility_name": "Lyon Plant",
                    "region_id": 1,
                    "status": "active",
                    "contact_email": "ops.lyon@example.invalid",
                    "tenant_id": "acme",
                },
                {
                    "facility_id": 3,
                    "facility_name": "Ohio Plant",
                    "region_id": 2,
                    "status": "active",
                    "contact_email": "ops.ohio@example.invalid",
                    "tenant_id": "globex",
                },
            ),
        ),
        TableSpec(
            name="energy_readings",
            comment="Monthly metered energy consumption per facility.",
            columns=(
                ColumnSpec("reading_id", "int", primary_key=True),
                ColumnSpec("facility_id", "int", foreign_key="facilities.facility_id"),
                ColumnSpec("reading_date", "date", comment="Date of the meter reading."),
                ColumnSpec(
                    "energy_kwh", "decimal(18,4)", comment="Energy consumed in kilowatt hours."
                ),
                ColumnSpec("tenant_id", "varchar(40)"),
            ),
            rows=tuple(
                {**row, "tenant_id": "acme" if row["facility_id"] in (1, 2) else "globex"}
                for row in _reading_rows()
            ),
        ),
        TableSpec(
            name="emissions",
            comment="Greenhouse gas emissions by scope.",
            columns=(
                ColumnSpec("emission_id", "int", primary_key=True),
                ColumnSpec("facility_id", "int", foreign_key="facilities.facility_id"),
                ColumnSpec("reporting_date", "date"),
                ColumnSpec("scope", "varchar(20)", comment="Reporting scope, for example Scope 1."),
                ColumnSpec(
                    "co2e_tonnes", "decimal(18,4)", comment="Emissions in tonnes of CO2 equivalent."
                ),
                ColumnSpec("tenant_id", "varchar(40)"),
            ),
            rows=tuple(
                {**row, "tenant_id": "acme" if row["facility_id"] in (1, 2) else "globex"}
                for row in _emission_rows()
            ),
        ),
        TableSpec(
            name="user_credentials",
            comment="Deliberately present so tests can prove it is never exposed.",
            columns=(
                ColumnSpec("user_id", "int", primary_key=True),
                ColumnSpec("username", "varchar(80)"),
                ColumnSpec("password_hash", "varchar(200)"),
                ColumnSpec("api_key", "varchar(200)"),
            ),
            rows=({"user_id": 1, "username": "svc", "password_hash": "x", "api_key": "y"},),
        ),
    ]


def build_sample_database(url: str = "sqlite+pysqlite:///:memory:") -> TestDatabase:
    """Create and seed the sample database."""
    return TestDatabase(specs=sample_tables(), url=url).create()
