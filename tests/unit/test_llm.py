"""The provider base class, Azure OpenAI adapter and local model adapter.

Azure OpenAI is exercised through an injected client double, so the request
shape, the structured output contract, the error mapping and the repair loop
are all tested without a network call.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from nl2sql.config.settings import AzureOpenAISettings, LocalModelSettings
from nl2sql.core.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from nl2sql.llm.azure_openai import AzureOpenAIProvider
from nl2sql.llm.base import extract_json, strict_json_schema
from nl2sql.llm.huggingface_local import HuggingFaceLocalProvider
from nl2sql.llm.prompts import RenderedPrompt
from nl2sql.llm.usage import UsageTracker, end_collection, start_collection
from nl2sql.observability.metrics import MetricsRegistry

pytestmark = pytest.mark.unit


class Answer(BaseModel):
    """The contract used by these tests."""

    sql: str = ""
    confidence: float = 0.5


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 120
        self.completion_tokens = 40


class _Response:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage()


class FakeAzureClient:
    """Records requests and returns queued responses or raises queued errors."""

    def __init__(self, *responses: Any) -> None:
        self.queued = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = self

    @property
    def completions(self) -> FakeAzureClient:
        """Mirror the SDK's chat.completions attribute chain."""
        return self

    async def create(self, **kwargs: Any) -> Any:
        """Return the next queued response."""
        self.requests.append(kwargs)
        item = self.queued.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _settings(**overrides: Any) -> AzureOpenAISettings:
    return AzureOpenAISettings(
        enabled=True,
        endpoint="https://example.openai.azure.com",
        deployment="gpt-test",
        **overrides,
    )


def _provider(client: FakeAzureClient, **overrides: Any) -> AzureOpenAIProvider:
    return AzureOpenAIProvider(
        _settings(**overrides),
        usage=UsageTracker(MetricsRegistry()),
        client_factory=lambda: client,
    )


def _prompt() -> RenderedPrompt:
    return RenderedPrompt(name="sql_generation", version="v1", system="rules", user="question")


# -- structured output ------------------------------------------------------
def test_strict_schema_closes_objects_and_requires_every_field():
    """Strict enforcement needs closed objects with all properties required."""
    schema = strict_json_schema(Answer)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"sql", "confidence"}
    assert "default" not in str(schema)


@pytest.mark.parametrize(
    "text",
    [
        '{"sql": "SELECT 1", "confidence": 0.5}',
        '```json\n{"sql": "SELECT 1", "confidence": 0.5}\n```',
        'Here you go: {"sql": "SELECT 1", "confidence": 0.5} Hope that helps.',
    ],
)
def test_json_is_extracted_from_the_shapes_models_actually_return(text):
    """Fences and surrounding prose do not defeat parsing."""
    assert extract_json(text)["sql"] == "SELECT 1"


def test_empty_or_unparsable_output_is_an_error():
    """Output that is not JSON at all is reported as such."""
    with pytest.raises(LLMResponseError):
        extract_json("")
    with pytest.raises(LLMResponseError):
        extract_json("I cannot help with that.")


async def test_structured_output_is_requested_and_validated():
    """The schema is sent to Azure and the reply is validated against it."""
    client = FakeAzureClient(_Response('{"sql": "SELECT 1", "confidence": 0.8}'))
    provider = _provider(client)
    result = await provider.generate_structured(_prompt(), Answer)

    assert result.output.sql == "SELECT 1"
    assert result.usage.prompt_tokens == 120
    assert result.prompt_ref == "sql_generation@v1"
    request = client.requests[0]
    assert request["model"] == "gpt-test"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["max_tokens"] > 0


async def test_invalid_output_is_repaired_once():
    """A malformed reply produces one corrective attempt before failing."""
    client = FakeAzureClient(
        _Response("not json at all"), _Response('{"sql": "SELECT 2", "confidence": 0.4}')
    )
    provider = _provider(client)
    result = await provider.generate_structured(_prompt(), Answer)

    assert result.output.sql == "SELECT 2"
    assert len(client.requests) == 2
    assert client.requests[1]["messages"][-1]["role"] == "user"


async def test_output_that_never_validates_raises():
    """Repeated invalid output is reported rather than guessed at."""
    client = FakeAzureClient(_Response("nonsense"), _Response("still nonsense"))
    provider = _provider(client)
    with pytest.raises(LLMResponseError):
        await provider.generate_structured(_prompt(), Answer)


async def test_json_object_mode_sends_the_schema_in_the_prompt(container):
    """Without native schema support the contract goes into the system message."""
    client = FakeAzureClient(_Response('{"sql": "SELECT 1", "confidence": 0.5}'))
    provider = AzureOpenAIProvider(
        _settings(structured_output_mode="json_object"),
        usage=UsageTracker(),
        format_prompt=container.prompts.get("structured_output"),
        client_factory=lambda: client,
    )
    await provider.generate_structured(_prompt(), Answer)

    request = client.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert "JSON" in request["messages"][0]["content"]


async def test_token_limit_parameter_is_configurable():
    """Newer deployments take max_completion_tokens instead of max_tokens."""
    client = FakeAzureClient(_Response('{"sql": "SELECT 1", "confidence": 0.5}'))
    provider = _provider(
        client, token_limit_parameter="max_completion_tokens", send_temperature=False
    )
    await provider.generate_structured(_prompt(), Answer)

    request = client.requests[0]
    assert "max_completion_tokens" in request
    assert "temperature" not in request


# -- error handling ---------------------------------------------------------
async def test_content_filter_is_reported_without_leaking_the_question():
    """A filtered request fails with a message safe to return to a caller."""
    client = FakeAzureClient(_Response("", finish_reason="content_filter"))
    provider = _provider(client)
    with pytest.raises(LLMError) as caught:
        await provider.generate_structured(_prompt(), Answer)
    assert "question" in caught.value.public_message.lower()


@pytest.mark.parametrize(
    ("error_name", "expected"),
    [
        ("RateLimitError", LLMRateLimitError),
        ("APITimeoutError", LLMTimeoutError),
        ("APIConnectionError", LLMServiceError),
        ("InternalServerError", LLMServiceError),
    ],
)
def test_vendor_errors_map_onto_the_domain_hierarchy(error_name, expected):
    """Retryability is decided by the mapping, not by the call site."""
    import httpx
    import openai

    provider = _provider(FakeAzureClient())
    request = httpx.Request("POST", "https://example.openai.azure.com")
    response = httpx.Response(429, request=request)
    error_class = getattr(openai, error_name)
    if error_name in {"APITimeoutError", "APIConnectionError"}:
        error = error_class(request=request)
    else:
        error = error_class("failed", response=response, body=None)

    translated = provider._translate_error(error)
    assert isinstance(translated, expected)
    assert translated.retryable is True


async def test_transient_failures_are_retried_then_succeed():
    """A throttled call is retried under the shared policy."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://example.openai.azure.com")
    throttled = openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )
    client = FakeAzureClient(throttled, _Response('{"sql": "SELECT 3", "confidence": 0.6}'))
    settings = _settings()
    settings.retry.initial_backoff_seconds = 0
    settings.retry.jitter_seconds = 0
    provider = AzureOpenAIProvider(settings, usage=UsageTracker(), client_factory=lambda: client)

    result = await provider.generate_structured(_prompt(), Answer)
    assert result.output.sql == "SELECT 3"
    assert len(client.requests) == 2


# -- availability and usage -------------------------------------------------
def test_azure_is_unavailable_without_configuration():
    """A provider with no endpoint or credential reports itself unusable."""
    provider = AzureOpenAIProvider(AzureOpenAISettings(enabled=True), usage=UsageTracker())
    assert provider.is_available() is False


def test_local_model_is_unavailable_when_disabled():
    """The local provider reports unusable when configuration disables it."""
    provider = HuggingFaceLocalProvider(LocalModelSettings(enabled=False), usage=UsageTracker())
    assert provider.is_available() is False
    assert provider.model_name == "Qwen/Qwen2.5-Coder-0.5B-Instruct"


async def test_local_model_health_reports_unavailable_rather_than_raising():
    """Health never raises, so readiness can report it."""
    provider = HuggingFaceLocalProvider(LocalModelSettings(enabled=False), usage=UsageTracker())
    health = await provider.health()
    assert health.available is False


async def test_local_model_refuses_to_generate_when_unavailable():
    """Calling an unusable local provider fails with a clear error."""
    provider = HuggingFaceLocalProvider(LocalModelSettings(enabled=False), usage=UsageTracker())
    with pytest.raises(LLMUnavailableError):
        await provider._chat(
            [{"role": "user", "content": "hi"}],
            json_schema=None,
            schema_name=None,
            temperature=0.0,
            max_tokens=16,
        )


async def test_usage_is_collected_per_request():
    """Tokens are attributed to the question that caused them."""
    client = FakeAzureClient(_Response('{"sql": "SELECT 1", "confidence": 0.5}'))
    provider = _provider(client)
    collection, token = start_collection()
    try:
        await provider.generate_structured(_prompt(), Answer)
    finally:
        end_collection(token)

    assert collection.usage.prompt_tokens == 120
    assert collection.usage.completion_tokens == 40
    assert collection.models == ("gpt-test",)


async def test_usage_is_recorded_in_metrics():
    """The same call also increments the process wide counters."""
    metrics = MetricsRegistry()
    client = FakeAzureClient(_Response('{"sql": "SELECT 1", "confidence": 0.5}'))
    provider = AzureOpenAIProvider(
        _settings(), usage=UsageTracker(metrics), client_factory=lambda: client
    )
    await provider.generate_structured(_prompt(), Answer)

    labels = {"provider": "azure_openai", "model": "gpt-test"}
    assert metrics.counter_value("nl2sql_llm_calls_total", labels) == 1
    assert metrics.counter_value("nl2sql_llm_prompt_tokens_total", labels) == 120
    assert "nl2sql_llm_latency_ms" in metrics.render_prometheus()
