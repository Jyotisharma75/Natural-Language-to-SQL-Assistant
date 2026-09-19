"""Token accounting.

Usage is recorded twice, for two different questions. The context scoped
collector answers "what did this one question cost", which is what the audit
row and the response latency budget need. The registry answers "what is this
service spending", which is what a dashboard needs.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from nl2sql.observability.metrics import MetricsRegistry


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens consumed by one model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return the sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One completed model call."""

    provider: str
    model: str
    prompt_ref: str
    usage: TokenUsage
    latency_ms: float
    success: bool = True


_collector: ContextVar[list[LLMCall] | None] = ContextVar("nl2sql_llm_calls", default=None)


@dataclass(slots=True)
class UsageCollection:
    """The calls made while answering one question."""

    calls: list[LLMCall] = field(default_factory=list)

    @property
    def usage(self) -> TokenUsage:
        """Return the total tokens across every call."""
        total = TokenUsage()
        for call in self.calls:
            total = total + call.usage
        return total

    @property
    def models(self) -> tuple[str, ...]:
        """Return the distinct models used, in call order."""
        return tuple(dict.fromkeys(call.model for call in self.calls))


class UsageTracker:
    """Records calls into the active collection and into metrics."""

    def __init__(self, metrics: MetricsRegistry | None = None) -> None:
        self._metrics = metrics

    def record(self, call: LLMCall) -> None:
        """Record one model call."""
        active = _collector.get()
        if active is not None:
            active.append(call)
        if self._metrics is None:
            return
        labels = {"provider": call.provider, "model": call.model}
        self._metrics.increment("nl2sql_llm_calls_total", labels=labels)
        self._metrics.increment(
            "nl2sql_llm_prompt_tokens_total", call.usage.prompt_tokens, labels=labels
        )
        self._metrics.increment(
            "nl2sql_llm_completion_tokens_total", call.usage.completion_tokens, labels=labels
        )
        self._metrics.observe("nl2sql_llm_latency_ms", call.latency_ms, labels=labels)
        if not call.success:
            self._metrics.increment("nl2sql_llm_errors_total", labels=labels)


def start_collection() -> tuple[UsageCollection, Token[list[LLMCall] | None]]:
    """Begin collecting calls for the current context."""
    collection = UsageCollection()
    token = _collector.set(collection.calls)
    return collection, token


def end_collection(token: Token[list[LLMCall] | None]) -> None:
    """Stop collecting calls for the current context."""
    _collector.reset(token)
