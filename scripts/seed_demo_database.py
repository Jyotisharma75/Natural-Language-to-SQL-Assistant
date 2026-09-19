"""Create and seed a demonstration schema.

This is sample data for development and for the evaluation dataset that ships
with the repository. The application itself never refers to these tables: it
discovers whatever the database holds. Point it at a scratch database, never
at production, and use a login that may create tables, which is deliberately
not the read only login the service uses.

    python scripts/seed_demo_database.py --url "sqlite:///./data/demo.db"
    python scripts/seed_demo_database.py --url "mssql+pyodbc:///?odbc_connect=..." --schema dbo
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    create_engine,
    insert,
)

#: A fixed seed, so the same command always produces the same data and an
#: evaluation run is comparable with the one before it.
SEED = 20260201
MONTHS = 24


def build_metadata(schema: str | None) -> tuple[MetaData, dict[str, Table]]:
    """Define the demonstration tables."""
    metadata = MetaData()
    regions = Table(
        "regions",
        metadata,
        Column("region_id", Integer, primary_key=True, autoincrement=False),
        Column("region_name", String(80), nullable=False, comment="Reporting region name."),
        schema=schema,
        comment="Geographic reporting regions.",
    )
    facilities = Table(
        "facilities",
        metadata,
        Column("facility_id", Integer, primary_key=True, autoincrement=False),
        Column("facility_name", String(120), nullable=False, comment="Site name."),
        Column(
            "region_id",
            Integer,
            ForeignKey(regions.c.region_id),
            comment="Region the site reports into.",
        ),
        Column("site_type", String(40), comment="Plant, warehouse or office."),
        Column("status", String(20), comment="active or closed."),
        Column("floor_area_m2", Numeric(18, 2), comment="Gross floor area in square metres."),
        schema=schema,
        comment="Operating sites that report energy and emissions.",
    )
    energy = Table(
        "energy_consumption",
        metadata,
        Column("reading_id", Integer, primary_key=True, autoincrement=False),
        Column("facility_id", Integer, ForeignKey(facilities.c.facility_id)),
        Column("reading_date", Date, comment="First day of the reporting month."),
        Column("energy_kwh", Numeric(18, 4), comment="Energy consumed in kilowatt hours."),
        Column("renewable_kwh", Numeric(18, 4), comment="Portion supplied from renewables."),
        schema=schema,
        comment="Monthly metered energy consumption per facility.",
    )
    emissions = Table(
        "emissions",
        metadata,
        Column("emission_id", Integer, primary_key=True, autoincrement=False),
        Column("facility_id", Integer, ForeignKey(facilities.c.facility_id)),
        Column("reporting_date", Date, comment="First day of the reporting month."),
        Column("scope", String(20), comment="Scope 1, Scope 2 or Scope 3."),
        Column("co2e_tonnes", Numeric(18, 4), comment="Emissions in tonnes of CO2 equivalent."),
        schema=schema,
        comment="Greenhouse gas emissions by scope.",
    )
    return metadata, {
        "regions": regions,
        "facilities": facilities,
        "energy_consumption": energy,
        "emissions": emissions,
    }


def build_rows(start: date) -> dict[str, list[dict[str, object]]]:
    """Generate deterministic demonstration data, including one anomaly."""
    rng = random.Random(SEED)
    regions = [
        {"region_id": 1, "region_name": "Europe"},
        {"region_id": 2, "region_name": "North America"},
        {"region_id": 3, "region_name": "Asia Pacific"},
    ]
    names = [
        ("Rotterdam Plant", 1, "plant"),
        ("Lyon Plant", 1, "plant"),
        ("Hamburg Warehouse", 1, "warehouse"),
        ("Ohio Plant", 2, "plant"),
        ("Dallas Warehouse", 2, "warehouse"),
        ("Toronto Office", 2, "office"),
        ("Osaka Plant", 3, "plant"),
        ("Singapore Office", 3, "office"),
    ]
    facilities = [
        {
            "facility_id": index,
            "facility_name": name,
            "region_id": region,
            "site_type": site_type,
            "status": "active" if index != 6 else "closed",
            "floor_area_m2": round(rng.uniform(1500, 42000), 2),
        }
        for index, (name, region, site_type) in enumerate(names, start=1)
    ]

    energy: list[dict[str, object]] = []
    emissions: list[dict[str, object]] = []
    reading_id = 1
    emission_id = 1

    for facility in facilities:
        base = {"plant": 480000, "warehouse": 120000, "office": 45000}[str(facility["site_type"])]
        for month in range(MONTHS):
            period = _add_months(start, -month)
            seasonal = 1 + 0.12 * ((month % 12) - 5) / 6
            value = base * seasonal * rng.uniform(0.94, 1.06)
            # One site develops a clear anomaly in the most recent month, so
            # an abnormal usage question has something real to find.
            if facility["facility_id"] == 4 and month == 0:
                value *= 3.4
            renewable = value * rng.uniform(0.05, 0.6)
            energy.append(
                {
                    "reading_id": reading_id,
                    "facility_id": facility["facility_id"],
                    "reading_date": period,
                    "energy_kwh": round(value, 4),
                    "renewable_kwh": round(renewable, 4),
                }
            )
            reading_id += 1

            for scope, factor in (("Scope 1", 0.00021), ("Scope 2", 0.00037), ("Scope 3", 0.00052)):
                emissions.append(
                    {
                        "emission_id": emission_id,
                        "facility_id": facility["facility_id"],
                        "reporting_date": period,
                        "scope": scope,
                        "co2e_tonnes": round(value * factor * rng.uniform(0.9, 1.1), 4),
                    }
                )
                emission_id += 1

    return {
        "regions": regions,
        "facilities": facilities,
        "energy_consumption": energy,
        "emissions": emissions,
    }


def _add_months(start: date, delta: int) -> date:
    """Return the first day of the month ``delta`` months from ``start``."""
    month_index = start.year * 12 + (start.month - 1) + delta
    return date(month_index // 12, month_index % 12 + 1, 1)


def main(argv: list[str] | None = None) -> int:
    """Create the demonstration tables and insert the sample rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="SQLAlchemy URL of a scratch database.")
    parser.add_argument("--schema", default=None, help="Schema to create the tables in.")
    parser.add_argument("--drop", action="store_true", help="Drop the tables first.")
    parser.add_argument(
        "--start",
        default=date.today().replace(day=1).isoformat(),
        help="Most recent reporting month, as an ISO date.",
    )
    args = parser.parse_args(argv)

    engine = create_engine(args.url)
    metadata, tables = build_metadata(args.schema)
    if args.drop:
        metadata.drop_all(engine)
    metadata.create_all(engine)

    rows = build_rows(date.fromisoformat(args.start).replace(day=1))
    with engine.begin() as connection:
        for name in ("regions", "facilities", "energy_consumption", "emissions"):
            connection.execute(insert(tables[name]), rows[name])

    total = sum(len(values) for values in rows.values())
    target = engine.url.render_as_string()
    print(f"Created {len(tables)} tables and inserted {total} rows into {target}")
    print(f"Energy readings cover {MONTHS} months up to {args.start}.")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
