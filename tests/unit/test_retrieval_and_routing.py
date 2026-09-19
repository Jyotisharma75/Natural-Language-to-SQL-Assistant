"""Schema retrieval and model routing."""

from __future__ import annotations

import pytest

from nl2sql.core.exceptions import LLMUnavailableError, NoRelevantSchemaError
from nl2sql.llm.factory import ProviderRegistry
from nl2sql.metadata.models import TableInfo
from nl2sql.pipeline.models import IntentAnalysis, NormalizedQuestion, RetrievedSchema
from nl2sql.pipeline.router import ModelRouter
from nl2sql.pipeline.schema_retriever import SchemaRetriever

pytestmark = pytest.mark.unit


@pytest.fixture
def retriever(settings, container):
    """A retriever wired to the test configuration."""
    return SchemaRetriever(settings.retrieval, container.prompts)


def test_identifiers_are_split_into_comparable_words(retriever):
    """Camel case, snake case and plurals all reduce to the same tokens."""
    assert retriever.tokenize("EnergyConsumption") == ["energy", "consumption"]
    assert retriever.tokenize("energy_consumption") == ["energy", "consumption"]
    assert retriever.tokenize("facilities") == ["facility"]
    assert retriever.tokenize("emissions") == ["emission"]


def test_stopwords_are_dropped(retriever):
    """Question filler does not contribute to scoring."""
    assert "the" not in retriever.tokenize("what is the total for the facility")


async def test_relevant_tables_score_above_unrelated_ones(retriever, catalog):
    """A question about energy retrieves the energy table first."""
    scores = retriever.score_tables(
        retriever.tokenize("total energy consumption by facility"), catalog
    )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    assert ranked[0][0] == "main.energy_readings"
    assert scores.get("main.user_credentials", 0) < scores["main.energy_readings"]


async def test_foreign_keys_pull_in_the_naming_table(retriever, catalog):
    """A question about a measure also retrieves the table holding the names."""
    result = await retriever.retrieve(
        "total energy consumption", IntentAnalysis(intent_type="aggregation"), catalog
    )
    names = set(result.table_names)
    assert "main.energy_readings" in names
    assert "main.facilities" in names
    assert result.join_edges >= 1


async def test_context_describes_columns_keys_and_relationships(retriever, catalog):
    """The prompt context carries types, keys and join paths."""
    result = await retriever.retrieve(
        "emissions by region", IntentAnalysis(intent_type="comparison"), catalog
    )
    assert "TABLE main.emissions" in result.context
    assert "co2e_tonnes" in result.context
    assert "PRIMARY KEY" in result.context
    assert "RELATIONSHIPS" in result.context


async def test_hidden_columns_are_absent_from_the_context(retriever, catalog):
    """A blocked column is never described to a model."""
    result = await retriever.retrieve(
        "user credentials", IntentAnalysis(intent_type="lookup"), catalog
    )
    assert "password_hash" not in result.context
    assert "api_key" not in result.context


async def test_column_cap_is_respected(settings, container, catalog):
    """A wide table is truncated, and the context says so."""
    narrow = SchemaRetriever(
        settings.retrieval.model_copy(update={"max_columns_per_table": 2}), container.prompts
    )
    result = await narrow.retrieve("facilities", IntentAnalysis(intent_type="lookup"), catalog)
    assert "further columns are not listed" in result.context


async def test_a_question_with_no_relevant_tables_is_refused(settings, container, catalog):
    """When nothing matches and the schema is large, the request is refused."""
    retriever = SchemaRetriever(
        settings.retrieval.model_copy(update={"top_k_tables": 1}), container.prompts
    )
    with pytest.raises(NoRelevantSchemaError):
        await retriever.retrieve("zzzz qqqq xxxx", IntentAnalysis(intent_type="lookup"), catalog)


# -- routing ---------------------------------------------------------------
def _question(text: str = "how many facilities are there", followup: bool = False):
    return NormalizedQuestion(original=text, text=text, is_followup=followup)


def _tables(count: int) -> tuple[TableInfo, ...]:
    """Build placeholder tables, since routing only counts them."""
    return tuple(TableInfo(schema_name="main", name=f"t{index}") for index in range(count))


def _schema(tables: int = 1, edges: int = 0) -> RetrievedSchema:
    return RetrievedSchema(tables=_tables(tables), join_edges=edges, candidate_count=tables)


def test_a_simple_question_goes_to_the_local_model(settings, providers):
    """A low complexity question is answered locally and verified centrally."""
    router = ModelRouter(settings.routing, providers)
    decision = router.route(
        _question("how many facilities"), IntentAnalysis(intent_type="lookup"), _schema()
    )
    assert decision.primary == "local"
    assert decision.verifier == "azure_openai"
    assert decision.complexity <= settings.routing.local_max_complexity


def test_a_complex_question_goes_to_the_hosted_model(settings, providers):
    """An anomaly question over several joined tables escalates by default."""
    router = ModelRouter(settings.routing, providers)
    schema = RetrievedSchema(tables=_tables(6), join_edges=4, candidate_count=6)
    decision = router.route(
        _question("show facilities with abnormal energy usage compared with last year"),
        IntentAnalysis(intent_type="anomaly", ambiguity=0.8, time_expressions=["last year"]),
        schema,
    )
    assert decision.primary == "azure_openai"
    assert decision.verifier == "local"
    assert decision.complexity > settings.routing.local_max_complexity


def test_routing_falls_back_when_the_local_model_is_missing(settings, providers, local_provider):
    """With no local model everything goes to the hosted one, with no verifier."""
    local_provider.available = False
    router = ModelRouter(settings.routing, providers)
    decision = router.route(_question(), IntentAnalysis(), _schema())
    assert decision.primary == "azure_openai"
    assert decision.verifier is None


def test_routing_falls_back_when_the_hosted_model_is_missing(settings, providers, azure_provider):
    """With no hosted model the local one answers everything."""
    azure_provider.available = False
    router = ModelRouter(settings.routing, providers)
    decision = router.route(
        _question("compare emissions across regions over time"),
        IntentAnalysis(intent_type="comparison"),
        _schema(),
    )
    assert decision.primary == "local"
    assert decision.verifier is None


def test_no_model_at_all_is_an_error(settings, azure_provider, local_provider):
    """With nothing available the request fails rather than guessing."""
    azure_provider.available = False
    local_provider.available = False
    router = ModelRouter(
        settings.routing,
        ProviderRegistry({"azure_openai": azure_provider, "local": local_provider}),
    )
    with pytest.raises(LLMUnavailableError):
        router.route(_question(), IntentAnalysis(), _schema())


def test_verification_is_triggered_by_low_confidence(settings, providers):
    """A confident answer is not double checked; an unsure one is."""
    router = ModelRouter(settings.routing, providers)
    decision = router.route(_question(), IntentAnalysis(), _schema())
    assert router.should_verify(decision, confidence=0.5) is True
    assert router.should_verify(decision, confidence=0.95) is False


def test_verification_is_always_triggered_by_high_complexity(settings, providers):
    """Above the complexity threshold, confidence does not excuse a check."""
    router = ModelRouter(settings.routing, providers)
    decision = router.route(
        _question("compare abnormal emissions across regions year over year"),
        IntentAnalysis(intent_type="anomaly", ambiguity=1.0, time_expressions=["a", "b"]),
        RetrievedSchema(tables=_tables(9), join_edges=8, candidate_count=9),
    )
    assert decision.complexity >= settings.routing.always_verify_above_complexity
    assert router.should_verify(decision, confidence=0.99) is True


def test_routing_can_be_disabled(settings, providers):
    """With routing off, the configured default provider answers everything."""
    router = ModelRouter(
        settings.routing.model_copy(update={"enabled": False}),
        providers,
        default_provider="azure_openai",
    )
    decision = router.route(_question(), IntentAnalysis(), _schema())
    assert decision.primary == "azure_openai"
    assert decision.verifier is None
