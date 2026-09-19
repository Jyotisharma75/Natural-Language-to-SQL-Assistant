"""The evaluation framework.

Two things are checked: that the metric functions compute what they claim,
and that the runner scores a real run honestly, including reporting a case it
could not judge as not scored rather than counting it as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nl2sql.evaluation.dataset import ExpectedResult, load_dataset
from nl2sql.evaluation.metrics import (
    ast_equivalent,
    check_characteristics,
    normalise_name,
    percentile,
    rate,
    results_match,
    set_score,
)
from nl2sql.evaluation.report import to_dict, to_markdown, write_report
from nl2sql.evaluation.runner import EvaluationRunner
from nl2sql.pipeline.models import AnswerOutput
from tests.conftest import sql_response

pytestmark = pytest.mark.evaluation

DATASET = Path(__file__).parent / "dataset.jsonl"

ENERGY_SQL = (
    "SELECT f.facility_name AS facility_name, SUM(e.energy_kwh) AS total_kwh "
    "FROM energy_readings e JOIN facilities f ON f.facility_id = e.facility_id "
    "GROUP BY f.facility_name ORDER BY total_kwh DESC"
)
SCOPE_SQL = (
    "SELECT r.region_name AS region_name, SUM(m.co2e_tonnes) AS scope_one_tonnes "
    "FROM emissions m JOIN facilities f ON f.facility_id = m.facility_id "
    "JOIN regions r ON r.region_id = f.region_id WHERE m.scope = 'Scope 1' "
    "GROUP BY r.region_name"
)
COUNT_SQL = "SELECT COUNT(*) AS facility_count FROM facilities"


# -- metrics ----------------------------------------------------------------
def test_qualified_names_compare_across_schemas():
    """A dataset written for one schema still scores against another."""
    assert normalise_name("[dbo].[Facilities]") == "dbo.facilities"
    assert set_score(["dbo.facilities"], ["dbo.facilities"]).f1 == 1.0


def test_set_score_reports_precision_and_recall():
    """Reading more tables than needed is not scored the same as reading the right ones."""
    score = set_score(["dbo.a", "dbo.b"], ["dbo.a", "dbo.c"])
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.missing == ("dbo.b",)
    assert score.extra == ("dbo.c",)


def test_empty_expectations_do_not_penalise():
    """A case that states no expectation on an axis scores full marks on it."""
    assert set_score([], ["anything"]).f1 == 1.0


def test_results_match_ignores_row_order_and_tiny_float_differences():
    """Two correct queries that order rows differently still agree."""
    assert results_match([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]]) is True
    assert results_match([[1.0000001]], [[1.0]], tolerance=1e-3) is True
    assert results_match([[1]], [[2]]) is False
    assert results_match([[1]], [[1], [2]]) is False


def test_semantic_equivalence_ignores_aliases_and_case():
    """The same query written differently is recognised as the same query."""
    assert ast_equivalent(
        "SELECT a FROM t WHERE b = 1", "select A from T where B = 1", dialect="sqlite"
    )
    assert not ast_equivalent("SELECT a FROM t", "SELECT a FROM other", dialect="sqlite")


def test_characteristics_catch_a_result_of_the_wrong_shape():
    """Row counts, columns and ordering are checked against the expectation."""
    expected = ExpectedResult(row_count=2, columns=["total"], ordered_by="total", descending=True)
    assert check_characteristics(expected, ["name", "total"], [["a", 5], ["b", 3]]) == []

    failures = check_characteristics(expected, ["name", "total"], [["a", 3], ["b", 5]])
    assert any("ordered" in failure for failure in failures)

    failures = check_characteristics(expected, ["name"], [["a"]])
    assert any("exactly 2 rows" in failure for failure in failures)
    assert any("missing expected columns" in failure for failure in failures)


def test_percentiles_and_rates():
    """The summary arithmetic is what it claims to be."""
    assert percentile([10, 20, 30, 40], 0.5) == 25.0
    assert percentile([], 0.95) == 0.0
    assert rate(1, 4) == 0.25
    assert rate(1, 0) == 0.0


# -- dataset ----------------------------------------------------------------
def test_dataset_loads_with_expectations():
    """The shipped test dataset parses into cases."""
    cases = load_dataset(DATASET)
    assert len(cases) == 3
    assert cases[0].expected_tables
    assert cases[0].reference_sql
    assert cases[0].expected_result.ordered_by == "total_kwh"


def test_the_demo_dataset_is_well_formed(repo_root):
    """The dataset used against the demo schema parses and is unique."""
    cases = load_dataset(repo_root / "evaluation" / "datasets" / "sustainability_demo.jsonl")
    assert len(cases) >= 8
    assert all(case.reference_sql for case in cases)
    assert len({case.id for case in cases}) == len(cases)


def test_duplicate_case_ids_are_refused(tmp_path):
    """A dataset with repeated identifiers is a mistake worth failing on."""
    from nl2sql.core.exceptions import ConfigurationError

    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": "same", "question": "q"}) for _ in range(2)),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_dataset(path)


# -- runner -----------------------------------------------------------------
def _runner(container) -> EvaluationRunner:
    return EvaluationRunner(
        pipeline=container.pipeline,
        validator=container.validator,
        executor=container.executor,
        metadata=container.metadata,
        settings=container.settings.evaluation,
    )


async def test_runner_scores_a_correct_run(container, azure_provider, local_provider):
    """A run that matches the reference on every case is reported as correct."""
    local_provider.available = False
    for sql in (ENERGY_SQL, SCOPE_SQL, COUNT_SQL):
        azure_provider.queue("sql_generation", sql_response(sql))
        azure_provider.queue("answer_generation", AnswerOutput(answer="Done."))

    report = await _runner(container).run(load_dataset(DATASET), dataset_name="test")

    assert report.aggregates["cases"] == 3
    assert report.aggregates["sql_validity_rate"] == 1.0
    assert report.aggregates["execution_success_rate"] == 1.0
    assert report.aggregates["result_match_rate_of_compared"] == 1.0
    assert report.aggregates["correctness_rate_of_scored"] == 1.0
    assert report.aggregates["table_f1_mean"] == 1.0
    assert report.aggregates["latency_ms_p95"] > 0


async def test_runner_reports_a_wrong_answer_as_wrong(container, azure_provider, local_provider):
    """A query that runs but returns the wrong rows is not counted as correct."""
    local_provider.available = False
    azure_provider.queue("sql_generation", sql_response(COUNT_SQL))
    azure_provider.queue("answer_generation", AnswerOutput(answer="Done."))
    azure_provider.queue("sql_generation", sql_response(SCOPE_SQL))
    azure_provider.queue("answer_generation", AnswerOutput(answer="Done."))
    azure_provider.queue("sql_generation", sql_response(COUNT_SQL))
    azure_provider.queue("answer_generation", AnswerOutput(answer="Done."))

    report = await _runner(container).run(load_dataset(DATASET), dataset_name="test")

    first = report.cases[0]
    assert first.sql_valid is True
    assert first.executed is True
    assert first.results_match is False
    assert first.correct is False
    assert first.table_score.f1 < 1.0
    assert report.aggregates["correctness_rate_of_scored"] < 1.0


async def test_runner_records_a_refusal_without_crashing(container, azure_provider, local_provider):
    """A case the pipeline refuses is reported with its error code."""
    local_provider.available = False
    for _ in range(3):
        azure_provider.queue("sql_generation", sql_response(""))
        azure_provider.queue("sql_repair", sql_response(""))

    report = await _runner(container).run(load_dataset(DATASET))

    assert report.aggregates["errors"] == 3
    assert report.aggregates["sql_validity_rate"] == 0.0
    assert all(case.error_code == "sql_validation_failed" for case in report.cases)


async def test_reports_are_written(container, azure_provider, local_provider, tmp_path):
    """The report is written as JSON and Markdown with only measured numbers."""
    local_provider.available = False
    for sql in (ENERGY_SQL, SCOPE_SQL, COUNT_SQL):
        azure_provider.queue("sql_generation", sql_response(sql))
        azure_provider.queue("answer_generation", AnswerOutput(answer="Done."))

    report = await _runner(container).run(load_dataset(DATASET), dataset_name="test")
    json_path, markdown_path = write_report(report, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["aggregates"]["cases"] == 3
    assert len(payload["cases"]) == 3

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Evaluation report" in markdown
    assert "energy-by-facility" in markdown
    assert to_dict(report)["dialect"] == "sqlite"
    assert "not scored" in to_markdown(report) or "yes" in to_markdown(report)
