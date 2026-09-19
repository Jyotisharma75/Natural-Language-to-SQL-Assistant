"""Test doubles.

``ScriptedProvider`` implements the real provider interface and returns
prepared answers per prompt name. Using the interface rather than patching
means the pipeline under test is the production pipeline: the same validation,
the same routing and the same error handling run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from nl2sql.llm.base import LLMProvider, LLMResult, ProviderHealth
from nl2sql.llm.prompts import RenderedPrompt
from nl2sql.llm.usage import LLMCall, TokenUsage, UsageTracker

T = TypeVar("T", bound=BaseModel)

#: A scripted answer: a model instance, an exception to raise, or a callable
#: that receives the rendered prompt and returns one of those.
Response = BaseModel | Exception | Callable[[RenderedPrompt], BaseModel | Exception]


@dataclass
class RecordedCall:
    """One call made to a scripted provider."""

    prompt_name: str
    prompt_version: str
    system: str
    user: str
    output_model: str


@dataclass
class ScriptedProvider(LLMProvider):
    """A provider that returns prepared answers."""

    name_: str = "azure_openai"
    model: str = "scripted-model"
    responses: dict[str, list[Response]] = field(default_factory=dict)
    available: bool = True
    calls: list[RecordedCall] = field(default_factory=list)
    tracker: UsageTracker = field(default_factory=UsageTracker)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self.name_

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    def is_available(self) -> bool:
        """Return whether this provider should be treated as usable."""
        return self.available

    def queue(self, prompt_name: str, *responses: Response) -> ScriptedProvider:
        """Add answers for one prompt, consumed in order."""
        self.responses.setdefault(prompt_name, []).extend(responses)
        return self

    def calls_for(self, prompt_name: str) -> list[RecordedCall]:
        """Return the calls made for one prompt."""
        return [call for call in self.calls if call.prompt_name == prompt_name]

    async def generate_structured(
        self,
        prompt: RenderedPrompt,
        output_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult[T]:
        """Return the next prepared answer for this prompt."""
        self.calls.append(
            RecordedCall(
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                system=prompt.system,
                user=prompt.user,
                output_model=output_model.__name__,
            )
        )
        queued: Sequence[Response] = self.responses.get(prompt.name, [])
        if not queued:
            raise AssertionError(f"{self.name_} has no scripted response for prompt {prompt.name}.")
        response: Any = self.responses[prompt.name].pop(0)
        if callable(response) and not isinstance(response, BaseModel | Exception):
            response = response(prompt)
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, output_model):
            raise AssertionError(
                f"Scripted response for {prompt.name} is {type(response).__name__}, "
                f"but {output_model.__name__} was requested."
            )
        usage = TokenUsage(prompt_tokens=max(1, len(prompt.user) // 4), completion_tokens=32)
        # Recorded the way a real provider does, so token accounting, metrics
        # and the audit trail are exercised rather than bypassed.
        self.tracker.record(
            LLMCall(
                provider=self.name_,
                model=self.model,
                prompt_ref=prompt.ref,
                usage=usage,
                latency_ms=1.0,
            )
        )
        return LLMResult(
            output=response,
            raw_text=response.model_dump_json(),
            usage=usage,
            provider=self.name_,
            model=self.model,
            prompt_ref=prompt.ref,
            latency_ms=1.0,
        )

    async def health(self) -> ProviderHealth:
        """Report the scripted availability."""
        return ProviderHealth(
            name=self.name_,
            model=self.model,
            available=self.available,
            detail="scripted",
        )
