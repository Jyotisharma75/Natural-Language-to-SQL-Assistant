"""End to end pipeline tests over the sample database.

Only the model is scripted. Schema discovery, retrieval, routing, validation,
rewriting, execution, masking and answering all run for real against SQLite.
"""

from __future__ import annotations

import pytest

from nl2sql.core.exceptions import SQLValidationError
from nl2sql.pipeline.models import QueryFilter
from nl2sql.pipeline.orchestrator import QueryCommand
from tests.conftest import sql_response, verdict

pytestmark = pytest.mark.integration

TOTAL_BY_FACILITY = """
SELECT f.facility_name AS facility_name, SUM(e.energy_kwh) AS total_kwh
FROM energy_readings e
JOIN facilities f ON f.facility_id = e.facility_id
GROUP BY f.facility_name
ORDER BY total_kwh DESC
"""

SCOPE_ONE_BY_REGION = """
SELECT r.region_name AS region_name, SUM(m.co2e_tonnes) AS scope_one_tonnes
FROM emissions m
JOIN facilities f ON f.facility_id = m.facility_id
JOIN regions r ON r.region_id = f.region_id
WHERE m.scope = 'Scope 1'
GROUP BY r.region_name
"""


async def test_question_produces_answer_rows_and_sql(container, azure_provider, local_provider):
    """A routed question runs end to end and returns rows with an answer."""
    local_provider.available = False
    azure_provider.queue("sql_generation", sql_response(TOTAL_BY_FACILITY))
    azure_provider.queue("answer_generation", _answer("Rotterdam Plant used the most energy."))

    outcome = await container.pipeline.run(
        QueryCommand(question="Which facilities used the most energy?", principal=_reader())
    )

    assert outcome.executed is True
    assert outcome.row_count == 3
    assert [column["name"] for column in outcome.columns] == ["facility_name", "total_kwh"]
    assert outcome.answer == "Rotterdam Plant used the most energy."
    assert "LIMIT" in outcome.sql.upper()
    assert outcome.confidence > 0.5
    assert outcome.tables_used == ("main.energy_readings", "main.facilities")
    assert set(outcome.stage_timings) >= {"retrieval", "generation", "execution", "answer"}


async def test_validate_only_does_not_execute(container, azure_provider, local_provider):
    """The validate path produces SQL and stops before the database."""
    local_provider.available = False
    azure_provider.queue("sql_generation", sql_response(SCOPE_ONE_BY_REGION))

    outcome = await container.pipeline.run(
        QueryCommand(
            question="Compare Scope 1 emissions across regions.",
            principal=_reader(),
            validate_only=True,
        )
    )

    assert outcome.valid is True
    assert outcome.executed is False
    assert outcome.rows == []
    assert azure_provider.calls_for("answer_generation") == []


async def test_invalid_sql_is_repaired_with_validator_feedback(
    container, azure_provider, local_provider
):
    """A rejected query is corrected using the validator's own reasons."""
    local_provider.available = False
    azure_provider.queue(
        "sql_generation",
        sql_response("SELECT f.made_up_column FROM facilities f"),
    )
    azure_provider.queue("sql_repair", sql_response(TOTAL_BY_FACILITY, confidence=0.7))
    azure_provider.queue("answer_generation", _answer("Three facilities reported energy use."))

    outcome = await container.pipeline.run(
        QueryCommand(question="Total energy consumption by facility", principal=_reader())
    )

    assert outcome.executed is True
    repair_calls = azure_provider.calls_for("sql_repair")
    assert len(repair_calls) == 1
    assert "made_up_column" in repair_calls[0].user


async def test_failure_escalates_to_the_other_model(
    build_container, azure_provider, local_provider
):
    """When repair does not help, the other model generates instead."""
    # The threshold is raised so this question routes to the local model,
    # which is the case escalation exists for.
    container = build_container(routing={"local_max_complexity": 0.9})
    local_provider.queue(
        "sql_generation", sql_response("SELECT nonsense FROM nowhere", confidence=0.4)
    )
    local_provider.queue("sql_repair", sql_response("SELECT still_wrong FROM nowhere"))
    azure_provider.queue("sql_generation", sql_response(TOTAL_BY_FACILITY))
    azure_provider.queue("answer_generation", _answer("Energy totals by facility."))

    outcome = await container.pipeline.run(
        QueryCommand(question="energy by facility", principal=_reader())
    )

    assert outcome.executed is True
    assert any("second model" in warning for warning in outcome.warnings)
    # The answer comes from the model that produced the winning query.
    assert azure_provider.calls_for("answer_generation")


async def test_verifier_disagreement_triggers_a_second_opinion(
    container, azure_provider, local_provider
):
    """A low confidence query is checked, and disagreement produces an alternative."""
    azure_provider.queue("sql_generation", sql_response(TOTAL_BY_FACILITY, confidence=0.4))
    azure_provider.queue("answer_generation", _answer("Energy totals by facility."))
    local_provider.queue(
        "sql_verification", verdict(agrees=False, confidence=0.8, issues=["Wrong period."])
    )
    local_provider.queue("sql_generation", sql_response(SCOPE_ONE_BY_REGION, confidence=0.55))
    local_provider.queue("answer_generation", _answer("Scope 1 emissions by region."))

    outcome = await container.pipeline.run(
        QueryCommand(question="Which facilities had the highest energy use?", principal=_reader())
    )

    assert local_provider.calls_for("sql_verification")
    assert outcome.executed is True


async def test_blocked_columns_never_reach_the_model_or_the_result(
    container, azure_provider, local_provider
):
    """Credential columns are absent from the prompt and cannot be selected."""
    local_provider.available = False
    azure_provider.queue(
        "sql_generation", sql_response("SELECT password_hash FROM user_credentials")
    )
    azure_provider.queue("sql_repair", sql_response("SELECT username FROM user_credentials"))
    azure_provider.queue("answer_generation", _answer("One service account exists."))

    outcome = await container.pipeline.run(
        QueryCommand(question="how many service accounts are configured", principal=_reader())
    )

    generation = azure_provider.calls_for("sql_generation")[0]
    assert "password_hash" not in generation.user
    assert "api_key" not in generation.user
    assert outcome.rows == [["svc"]]


async def test_filters_are_applied_as_bound_parameters(container, azure_provider, local_provider):
    """Caller filters wrap the query and travel as parameters."""
    local_provider.available = False
    azure_provider.queue("sql_generation", sql_response(TOTAL_BY_FACILITY))
    azure_provider.queue("answer_generation", _answer("One facility matched."))

    outcome = await container.pipeline.run(
        QueryCommand(
            question="energy by facility",
            principal=_reader(),
            filters=(QueryFilter(column="facility_name", operator="contains", value="Ohio"),),
        )
    )

    assert outcome.row_count == 1
    assert outcome.rows[0][0] == "Ohio Plant"


async def test_masked_column_values_are_replaced(build_container, azure_provider, local_provider):
    """A masked column may be queried but its values never leave."""
    container = build_container(security={"masked_columns": ["*.facilities.contact_email"]})
    local_provider.available = False
    azure_provider.queue(
        "sql_generation",
        sql_response("SELECT facility_name, contact_email FROM facilities"),
    )
    azure_provider.queue("answer_generation", _answer("Three facilities."))

    outcome = await container.pipeline.run(
        QueryCommand(question="facility contacts", principal=_reader())
    )

    assert {row[1] for row in outcome.rows} == {"***"}
    assert any("masked" in warning for warning in outcome.warnings)


async def test_unanswerable_question_is_refused(container, azure_provider, local_provider):
    """A model that declines produces a refusal, not an empty result."""
    local_provider.available = False
    azure_provider.queue("sql_generation", sql_response("", warnings=["No salary data exists."]))
    # The pipeline offers one repair attempt before giving up, and the model
    # declines again.
    azure_provider.queue("sql_repair", sql_response("", warnings=["No salary data exists."]))

    with pytest.raises(SQLValidationError):
        await container.pipeline.run(
            QueryCommand(question="What is the average salary?", principal=_reader())
        )


async def test_audit_row_is_written(build_container, azure_provider, local_provider, app_engine):
    """Every question leaves one audit row with the masked question and the SQL."""
    from nl2sql.db.repositories import AuditRepository

    container = build_container(observability={"audit_enabled": True})
    local_provider.available = False
    azure_provider.queue("sql_generation", sql_response(TOTAL_BY_FACILITY))
    azure_provider.queue("answer_generation", _answer("Energy totals."))

    outcome = await container.pipeline.run(
        QueryCommand(question="energy by facility", principal=_reader())
    )

    record = AuditRepository(app_engine).get(outcome.query_id)
    assert record is not None
    assert record.execution_status == "succeeded"
    assert record.validation_passed is True
    assert record.row_count == 3
    assert record.primary_provider == "azure_openai"
    assert record.prompt_tokens > 0


def _reader():
    from nl2sql.security.auth import Principal

    return Principal(id="test-principal", tenant_id=None, roles=("reader",))


def _answer(text: str):
    from nl2sql.pipeline.models import AnswerOutput

    return AnswerOutput(answer=text)
