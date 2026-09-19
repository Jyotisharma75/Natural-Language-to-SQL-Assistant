"""Schema discovery and caching, conversation memory, question handling."""

from __future__ import annotations

import pytest

from nl2sql.config.settings import ConversationSettings
from nl2sql.conversation.memory_store import InMemoryConversationStore
from nl2sql.conversation.store import ConversationService
from nl2sql.core.exceptions import InputValidationError, SchemaNotFoundError
from nl2sql.core.masking import Masker
from nl2sql.metadata.cache import TTLCache
from nl2sql.pipeline.answer_generator import AnswerGenerator
from nl2sql.pipeline.intent import IntentAnalyzer
from nl2sql.pipeline.models import FormattedResult
from nl2sql.pipeline.normalizer import QuestionNormalizer
from nl2sql.security.prompt_injection import PromptInjectionDetector

pytestmark = pytest.mark.unit


# -- schema discovery -------------------------------------------------------
def test_tables_columns_and_keys_are_discovered(catalog):
    """Introspection reports what the database actually holds."""
    facilities = catalog.get("main", "facilities")
    assert facilities is not None
    assert facilities.primary_key == ("facility_id",)
    assert set(facilities.column_names) >= {"facility_name", "region_id", "status"}
    assert facilities.column("facility_name").data_type.startswith("VARCHAR")
    assert facilities.kind == "table"


def test_descriptions_are_read_where_the_dialect_provides_them():
    """Comments are mapped when reflection supplies them.

    SQLite reports no comments at all, and the introspector tolerates that
    rather than failing. Azure SQL supplies them from the MS_Description
    extended properties, which is the path this checks.
    """
    from nl2sql.metadata.introspector import SchemaIntrospector

    assert SchemaIntrospector._clean("  Operating   sites.  ") == "Operating sites."
    assert SchemaIntrospector._clean(None) is None
    assert SchemaIntrospector._clean("") is None


def test_relationships_are_discovered(catalog):
    """Foreign keys are read from the database, not declared in code."""
    readings = catalog.get("main", "energy_readings")
    targets = {fk.referred_qualified_name for fk in readings.foreign_keys}
    assert "main.facilities" in targets
    assert {t.name for t in catalog.neighbours(readings)} >= {"facilities"}


def test_blocked_columns_are_removed_from_the_catalogue(catalog):
    """The policy is applied before anything can read the catalogue."""
    credentials = catalog.get("main", "user_credentials")
    names = set(credentials.column_names)
    assert "username" in names
    assert "password_hash" not in names
    assert "api_key" not in names


def test_blocked_tables_disappear_entirely(build_container):
    """A blocked table is not visible at all."""
    container = build_container(security={"blocked_tables": ["*.user_credentials"]})
    catalog = container.metadata.catalog()
    assert catalog.get("main", "user_credentials") is None
    assert "main.user_credentials" not in [table.qualified_name for table in catalog.tables]


def test_unknown_schema_is_reported(container):
    """Asking for a schema that is not visible is a clear failure."""
    with pytest.raises(SchemaNotFoundError):
        container.metadata.list_tables("no_such_schema")


def test_schema_is_cached_and_can_be_refreshed(container):
    """Repeated reads hit the cache; refresh discards it."""
    first = container.metadata.catalog()
    assert container.metadata.catalog() is first
    assert container.metadata.refresh() is not first


def test_cache_reloads_after_the_ttl_expires():
    """A stale entry is reloaded when it ages out."""
    clock = {"now": 0.0}
    loads = {"count": 0}

    def loader() -> int:
        loads["count"] += 1
        return loads["count"]

    cache: TTLCache[int] = TTLCache(loader, ttl_seconds=10, clock=lambda: clock["now"], name="test")
    assert cache.get() == 1
    assert cache.get() == 1
    clock["now"] = 11
    assert cache.get() == 2


def test_cache_serves_stale_data_when_a_refresh_fails():
    """A schema that was right five minutes ago beats an outage."""
    clock = {"now": 0.0}
    state = {"fail": False}

    def loader() -> str:
        if state["fail"]:
            raise RuntimeError("database unavailable")
        return "value"

    cache: TTLCache[str] = TTLCache(
        loader, ttl_seconds=1, clock=lambda: clock["now"], serve_stale_on_error=True
    )
    assert cache.get() == "value"
    state["fail"] = True
    clock["now"] = 5
    assert cache.get() == "value"


def test_cache_raises_when_it_has_nothing_to_serve():
    """With no previous value there is nothing to fall back to."""

    def loader() -> str:
        raise RuntimeError("database unavailable")

    cache: TTLCache[str] = TTLCache(loader, ttl_seconds=1)
    with pytest.raises(RuntimeError):
        cache.get()


# -- conversation memory ----------------------------------------------------
def _service(**overrides) -> ConversationService:
    settings = ConversationSettings(**overrides)
    store = InMemoryConversationStore(
        max_turns=settings.max_turns, ttl_seconds=settings.ttl_seconds
    )
    return ConversationService(settings, store, Masker())


async def test_only_the_configured_fields_are_stored():
    """A deployment chooses what history holds, and nothing else is kept."""
    service = _service(stored_fields=["sql"])
    await service.record(
        "conversation-1",
        "principal-1",
        {"question": "secret question", "sql": "SELECT 1", "answer": "forty two"},
    )
    history = await service.history("conversation-1", "principal-1")
    assert history == ["sql: SELECT 1"]


async def test_stored_values_are_masked():
    """A credential pasted into a question is not persisted."""
    service = _service(stored_fields=["question"])
    await service.record(
        "conversation-1", "principal-1", {"question": "connect with Pwd=SuperSecret123"}
    )
    history = await service.history("conversation-1", "principal-1")
    assert "SuperSecret123" not in history[0]


async def test_history_is_private_to_the_principal():
    """Guessing a conversation id does not reveal another caller's turns."""
    service = _service(stored_fields=["sql"])
    await service.record("shared-id", "principal-1", {"sql": "SELECT 1"})
    assert await service.history("shared-id", "principal-2") == []


async def test_history_is_bounded():
    """Only the configured number of turns is kept and returned."""
    service = _service(stored_fields=["sql"], max_turns=3, context_turns=2)
    for index in range(5):
        await service.record("conversation-1", "principal-1", {"sql": f"SELECT {index}"})
    history = await service.history("conversation-1", "principal-1")
    assert history == ["sql: SELECT 3", "sql: SELECT 4"]


async def test_history_can_be_disabled():
    """With the feature off nothing is stored and nothing is returned."""
    service = _service(enabled=False)
    await service.record("conversation-1", "principal-1", {"sql": "SELECT 1"})
    assert await service.history("conversation-1", "principal-1") == []
    assert service.enabled is False


# -- question handling ------------------------------------------------------
def _normalizer(settings, container) -> QuestionNormalizer:
    return QuestionNormalizer(
        limits=settings.limits,
        detector=PromptInjectionDetector(settings.security.prompt_injection),
        prompts=container.prompts,
        conversation=settings.conversation,
    )


async def test_questions_are_normalised(settings, container):
    """Unicode, invisible characters and whitespace are normalised away."""
    normalizer = _normalizer(settings, container)
    result = await normalizer.normalize("  total​   energy usage  ")  # noqa: RUF001
    assert result.text == "total energy usage"


async def test_empty_and_oversized_questions_are_refused(settings, container):
    """An empty or enormous question fails before any model is called."""
    normalizer = _normalizer(settings, container)
    with pytest.raises(InputValidationError):
        await normalizer.normalize("   ")
    with pytest.raises(InputValidationError):
        await normalizer.normalize("a" * (settings.limits.max_question_chars + 1))


async def test_followup_is_rewritten_to_stand_alone(settings, container, azure_provider):
    """A follow up question is resolved against the earlier turns."""
    from nl2sql.pipeline.models import FollowupRewrite

    azure_provider.queue(
        "followup_rewrite",
        FollowupRewrite(standalone_question="What was the energy use of Ohio Plant in 2026?"),
    )
    normalizer = _normalizer(settings, container)
    result = await normalizer.normalize(
        "what about Ohio?",
        history=["standalone_question: What was the energy use in 2026?"],
        provider=azure_provider,
    )
    assert result.is_followup is True
    assert "Ohio Plant" in result.text


async def test_a_failed_rewrite_keeps_the_original_question(settings, container, azure_provider):
    """If the rewrite fails the question is still answered, with a warning."""
    from nl2sql.core.exceptions import LLMServiceError

    azure_provider.queue("followup_rewrite", LLMServiceError("upstream is down"))
    normalizer = _normalizer(settings, container)
    result = await normalizer.normalize(
        "what about Ohio?", history=["standalone_question: energy use"], provider=azure_provider
    )
    assert result.text == "what about Ohio?"
    assert result.warnings


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What was the total energy consumption last quarter?", "aggregation"),
        ("Which facilities had the highest emissions?", "ranking"),
        ("Compare Scope 1 emissions across regions.", "comparison"),
        ("Show the facilities with abnormal energy usage.", "anomaly"),
        ("How has energy use changed over time?", "trend"),
        ("Show the facilities in Europe", "lookup"),
    ],
)
def test_intent_is_classified_from_configured_keywords(settings, container, question, expected):
    """The heuristic classifier recognises the shapes the router cares about."""
    analyzer = IntentAnalyzer(settings.intent, container.prompts)
    assert analyzer.heuristic(question).intent_type == expected


def test_time_expressions_are_extracted(settings, container):
    """Period phrases are picked out, because they drive complexity."""
    analyzer = IntentAnalyzer(settings.intent, container.prompts)
    intent = analyzer.heuristic("total energy in Q3 2025 compared with last quarter")
    assert intent.time_expressions


async def test_intent_falls_back_to_the_heuristic_when_the_model_fails(
    settings, container, azure_provider
):
    """A model failure degrades to the keyword classifier rather than failing."""
    from nl2sql.core.exceptions import LLMServiceError

    azure_provider.queue("intent_analysis", LLMServiceError("upstream is down"))
    analyzer = IntentAnalyzer(settings.intent.model_copy(update={"mode": "llm"}), container.prompts)
    intent = await analyzer.analyze(
        "which facility had the highest emissions", provider=azure_provider
    )
    assert intent.intent_type == "ranking"


# -- answers ----------------------------------------------------------------
async def test_answer_falls_back_to_a_deterministic_summary(settings, container):
    """A query that ran is described even when no model is available."""
    generator = AnswerGenerator(container.prompts, settings.answer)
    result = FormattedResult(columns=[{"name": "total_kwh", "type": "number"}], rows=[[1234.5]])
    answer = await generator.generate(
        None, question="total energy", sql_summary="sums energy", result=result
    )
    assert "total_kwh" in answer


async def test_empty_results_are_described_as_an_answer(settings, container):
    """No rows is stated plainly rather than reported as a failure."""
    generator = AnswerGenerator(container.prompts, settings.answer)
    answer = await generator.generate(
        None, question="q", sql_summary="s", result=FormattedResult(columns=[], rows=[])
    )
    assert "no rows matched" in answer


async def test_rows_are_withheld_from_the_model_when_configured(
    settings, container, azure_provider
):
    """With row sending off, only statistics reach the model."""
    from nl2sql.pipeline.models import AnswerOutput

    azure_provider.queue("answer_generation", AnswerOutput(answer="Summarised."))
    generator = AnswerGenerator(
        container.prompts, settings.answer.model_copy(update={"send_rows_to_llm": False})
    )
    result = FormattedResult(
        columns=[{"name": "facility_name", "type": "string"}, {"name": "total", "type": "number"}],
        rows=[["Rotterdam Plant", 10.0], ["Lyon Plant", 20.0]],
    )
    await generator.generate(
        azure_provider, question="totals", sql_summary="sums energy", result=result
    )

    sent = azure_provider.calls_for("answer_generation")[0].user
    assert "Rotterdam Plant" not in sent
    assert "min 10.0" in sent
