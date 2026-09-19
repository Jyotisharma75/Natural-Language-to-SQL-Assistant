"""Intent understanding.

Knowing whether a question wants a total, a ranking, a comparison over groups,
a trend or an outlier changes three later decisions: which columns matter to
the retriever, how complex the router judges the question, and what the
generator is told to aim for.

Two modes are supported. The heuristic mode is free and instant and uses
keyword lists from configuration. The model mode is better on questions that
do not use obvious words. Either way the result has the same shape, and a
model failure falls back to the heuristic rather than failing the request.
"""

from __future__ import annotations

import re

from nl2sql.config.settings import IntentSettings
from nl2sql.core.exceptions import LLMError
from nl2sql.llm.base import LLMProvider
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import IntentAnalysis, IntentType

logger = get_logger(__name__)


class IntentAnalyzer:
    """Classifies the shape of the answer a question wants."""

    def __init__(self, settings: IntentSettings, prompts: PromptRegistry) -> None:
        self._settings = settings
        self._prompts = prompts
        self._time_patterns = [re.compile(pattern) for pattern in settings.time_patterns]

    async def analyze(
        self, question: str, *, provider: LLMProvider | None = None
    ) -> IntentAnalysis:
        """Return the intent of ``question``."""
        if self._settings.mode == "llm" and provider is not None and provider.is_available():
            try:
                prompt = self._prompts.render("intent_analysis", question=question)
                result = await provider.generate_structured(prompt, IntentAnalysis)
            except LLMError as exc:
                logger.warning("intent_analysis_failed", error_type=type(exc).__name__)
            else:
                return result.output
        return self.heuristic(question)

    def heuristic(self, question: str) -> IntentAnalysis:
        """Classify a question from the configured keyword lists."""
        lowered = question.casefold()
        words = lowered.split()

        intent: IntentType = self._settings.default_intent  # type: ignore[assignment]
        for candidate in self._settings.intent_priority:
            keywords = self._settings.keywords.get(candidate, [])
            if any(keyword.casefold() in lowered for keyword in keywords):
                intent = candidate  # type: ignore[assignment]
                break

        time_expressions: list[str] = []
        for pattern in self._time_patterns:
            time_expressions.extend(match.group(0) for match in pattern.finditer(question))

        metrics = [word for word in self._settings.aggregation_words if word.casefold() in lowered]

        ambiguity = 0.25
        if len(words) <= 4:
            ambiguity += 0.2
        if not time_expressions and intent in {"trend", "comparison", "anomaly"}:
            ambiguity += 0.2
        if not metrics and intent in {"aggregation", "ranking"}:
            ambiguity += 0.1

        return IntentAnalysis(
            intent_type=intent,
            time_expressions=list(dict.fromkeys(time_expressions)),
            entities=[],
            metrics=metrics,
            ambiguity=min(ambiguity, 1.0),
            is_data_question=True,
        )
