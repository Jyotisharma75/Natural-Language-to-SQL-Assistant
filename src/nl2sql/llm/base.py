"""Shared behaviour for language model providers.

Everything true of every provider lives here: the retry policy, usage
accounting, prompt version propagation, JSON extraction, schema validation and
a single repair attempt when a model returns something that does not match the
contract. An adapter is then responsible only for the shape of its own request
and response.

Structured output is the contract everywhere. A provider that supports native
schema enforcement is given the schema; a provider that does not is given the
same schema in its system message through a prompt file, and the result is
validated identically either way. That is what makes a small local model and a
hosted model interchangeable to the rest of the pipeline.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from nl2sql.core.exceptions import LLMError, LLMResponseError
from nl2sql.core.retry import RetryPolicy, build_async_retrying
from nl2sql.llm.prompts import PromptTemplate, RenderedPrompt
from nl2sql.llm.usage import LLMCall, TokenUsage, UsageTracker
from nl2sql.observability.logging import get_logger

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)

#: Keys that are useful to pydantic but rejected by strict schema enforcement.
_SCHEMA_KEYS_TO_STRIP = ("default", "title", "examples", "$comment", "format")


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Whether a provider is usable right now."""

    name: str
    model: str
    available: bool
    detail: str = ""
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class LLMResult(Generic[T]):
    """One structured completion and what it cost."""

    output: T
    raw_text: str
    usage: TokenUsage
    provider: str
    model: str
    prompt_ref: str
    latency_ms: float


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema for ``model`` that strict enforcement accepts.

    Every object is closed and lists all of its properties as required, and
    annotations that schema enforcement rejects are removed.
    """

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {
            key: clean(value) for key, value in node.items() if key not in _SCHEMA_KEYS_TO_STRIP
        }
        if result.get("type") == "object" or "properties" in result:
            properties = result.get("properties") or {}
            result["additionalProperties"] = False
            result["required"] = list(properties)
        return result

    cleaned: dict[str, Any] = clean(model.model_json_schema())
    return cleaned


def extract_json(text: str) -> Any:
    """Parse the JSON object out of model output.

    Handles the two things models do even when told not to: wrapping the object
    in a fenced code block, and adding a sentence before or after it.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise LLMResponseError("The model returned an empty response.")
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    "The model did not return valid JSON.",
                    details={"preview": candidate[:200]},
                ) from exc
        raise LLMResponseError(
            "The model did not return valid JSON.",
            details={"preview": candidate[:200]},
        ) from None


class LLMProvider(ABC):
    """The interface the pipeline depends on."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the configuration key of this provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model or deployment in use."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider is configured and its dependencies are present."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: RenderedPrompt,
        output_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[T]:
        """Return a completion validated against ``output_model``."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Report whether the provider is usable."""


class BaseLLMProvider(LLMProvider):
    """Retry, accounting, parsing and repair, shared by every adapter."""

    #: Whether the service can enforce a JSON schema on its own.
    supports_native_schema: bool = False

    def __init__(
        self,
        *,
        retry: RetryPolicy,
        usage: UsageTracker,
        format_prompt: PromptTemplate | None = None,
        repair_attempts: int = 1,
    ) -> None:
        self._retry = retry
        self._usage = usage
        self._format_prompt = format_prompt
        self._repair_attempts = max(0, repair_attempts)

    # -- adapter contract --------------------------------------------------
    @abstractmethod
    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None,
        schema_name: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """Perform one call and return the text and the tokens it used."""

    @abstractmethod
    def _translate_error(self, exc: Exception) -> LLMError:
        """Map a vendor exception onto the domain error hierarchy."""

    @abstractmethod
    def _defaults(self) -> tuple[float, int]:
        """Return the configured temperature and output token ceiling."""

    # -- public surface ----------------------------------------------------
    async def generate_structured(
        self,
        prompt: RenderedPrompt,
        output_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[T]:
        """Return a completion validated against ``output_model``."""
        schema = strict_json_schema(output_model)
        schema_name = output_model.__name__
        default_temperature, default_max_tokens = self._defaults()
        system = prompt.system
        if not self.supports_native_schema and self._format_prompt is not None:
            instruction = self._format_prompt.render(schema=json.dumps(schema), error="").system
            system = f"{system}\n\n{instruction}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt.user},
        ]

        last_error: Exception | None = None
        for attempt in range(self._repair_attempts + 1):
            text, usage, latency_ms = await self._invoke(
                messages,
                json_schema=schema if self.supports_native_schema else None,
                schema_name=schema_name,
                temperature=temperature if temperature is not None else default_temperature,
                max_tokens=max_tokens or default_max_tokens,
                prompt_ref=prompt.ref,
            )
            try:
                output = output_model.model_validate(extract_json(text))
            except (LLMResponseError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "llm_structured_output_invalid",
                    provider=self.name,
                    model=self.model_name,
                    prompt=prompt.ref,
                    attempt=attempt + 1,
                    error=str(exc)[:300],
                )
                if attempt >= self._repair_attempts:
                    break
                messages = [
                    *messages,
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content": self._repair_message(exc, schema)},
                ]
                continue
            return LLMResult(
                output=output,
                raw_text=text,
                usage=usage,
                provider=self.name,
                model=self.model_name,
                prompt_ref=prompt.ref,
                latency_ms=latency_ms,
            )

        raise LLMResponseError(
            f"The model did not return output matching {schema_name} after "
            f"{self._repair_attempts + 1} attempts.",
            details={"error": str(last_error)[:300] if last_error else None},
        )

    def _repair_message(self, error: Exception, schema: dict[str, Any]) -> str:
        """Build the follow up that asks the model to correct its output."""
        if self._format_prompt is not None:
            return self._format_prompt.render(
                schema=json.dumps(schema), error=str(error)[:500]
            ).user
        return (
            f"That response was not valid: {str(error)[:500]}. "
            "Reply with only a single JSON object that matches the schema."
        )

    async def health(self) -> ProviderHealth:
        """Report whether the provider is configured and reachable."""
        if not self.is_available():
            return ProviderHealth(
                name=self.name,
                model=self.model_name,
                available=False,
                detail="not configured",
            )
        started = time.monotonic()
        try:
            detail = await self._probe()
        except Exception as exc:
            return ProviderHealth(
                name=self.name,
                model=self.model_name,
                available=False,
                detail=f"{type(exc).__name__}",
                latency_ms=(time.monotonic() - started) * 1000,
            )
        return ProviderHealth(
            name=self.name,
            model=self.model_name,
            available=True,
            detail=detail,
            latency_ms=(time.monotonic() - started) * 1000,
        )

    async def _probe(self) -> str:
        """Make the cheapest check that the provider is usable."""
        return "configured"

    # -- the shared call path ---------------------------------------------
    async def _invoke(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None,
        schema_name: str | None,
        temperature: float,
        max_tokens: int,
        prompt_ref: str,
    ) -> tuple[str, TokenUsage, float]:
        """Run one call under the retry policy, recording usage and latency."""
        started = time.monotonic()
        attempt_number = 0

        async for attempt in build_async_retrying(self._retry):
            with attempt:
                attempt_number += 1
                try:
                    text, usage = await self._chat(
                        messages,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except LLMError:
                    self._usage.record(
                        LLMCall(
                            provider=self.name,
                            model=self.model_name,
                            prompt_ref=prompt_ref,
                            usage=TokenUsage(),
                            latency_ms=(time.monotonic() - started) * 1000,
                            success=False,
                        )
                    )
                    raise
                except Exception as exc:
                    self._usage.record(
                        LLMCall(
                            provider=self.name,
                            model=self.model_name,
                            prompt_ref=prompt_ref,
                            usage=TokenUsage(),
                            latency_ms=(time.monotonic() - started) * 1000,
                            success=False,
                        )
                    )
                    raise self._translate_error(exc) from exc

                latency_ms = (time.monotonic() - started) * 1000
                self._usage.record(
                    LLMCall(
                        provider=self.name,
                        model=self.model_name,
                        prompt_ref=prompt_ref,
                        usage=usage,
                        latency_ms=latency_ms,
                    )
                )
                logger.info(
                    "llm_call_completed",
                    provider=self.name,
                    model=self.model_name,
                    prompt=prompt_ref,
                    attempt=attempt_number,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency_ms=round(latency_ms, 1),
                )
                return text, usage, latency_ms

        raise AssertionError("unreachable: tenacity either returns or reraises")
