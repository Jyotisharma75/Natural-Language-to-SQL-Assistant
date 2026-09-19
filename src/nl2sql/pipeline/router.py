"""Model routing.

Two models are available and they are good at different things. The local
model is free, private and fast enough for a question that reads one or two
tables. The hosted model is better at questions that need several joins, a
window, a comparison across groups or a judgement about what counts as
abnormal.

The router scores a question before any SQL exists, using signals that are
already known: how many tables retrieval chose, how many join paths connect
them, what kind of answer the intent analysis expects, how many time
expressions are involved, how long the question is, whether it was a follow up
and how ambiguous it looked. Each signal is normalised to the range 0 to 1 and
weighted by configuration, so the routing policy is tuned without a code
change.

The model that does not generate becomes the verifier, which is what makes the
cross checking in the next stage independent.
"""

from __future__ import annotations

from nl2sql.config.settings import RoutingSettings
from nl2sql.core.exceptions import LLMUnavailableError
from nl2sql.llm.factory import AZURE, LOCAL, ProviderRegistry
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import (
    IntentAnalysis,
    NormalizedQuestion,
    RetrievedSchema,
    RoutingDecision,
)

logger = get_logger(__name__)


class ModelRouter:
    """Chooses which model generates and which one verifies."""

    def __init__(
        self,
        settings: RoutingSettings,
        providers: ProviderRegistry,
        *,
        default_provider: str = AZURE,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._default = default_provider

    def complexity(
        self,
        question: NormalizedQuestion,
        intent: IntentAnalysis,
        schema: RetrievedSchema,
    ) -> tuple[float, dict[str, float]]:
        """Return the complexity score and the normalised features behind it."""
        settings = self._settings
        features = {
            "table_count": _saturate(len(schema.tables), settings.table_count_saturation),
            "join_paths": _saturate(schema.join_edges, settings.join_path_saturation),
            "intent": settings.intent_complexity.get(intent.intent_type, 0.5),
            "time_expressions": _saturate(
                len(intent.time_expressions), settings.time_expression_saturation
            ),
            "question_length": _saturate(
                len(question.text.split()), settings.question_length_saturation
            ),
            "followup": 1.0 if question.is_followup else 0.0,
            "ambiguity": intent.ambiguity,
        }
        weights = settings.weights.model_dump()
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0, features
        score = sum(features[name] * weight for name, weight in weights.items()) / total_weight
        return round(min(max(score, 0.0), 1.0), 4), features

    def route(
        self,
        question: NormalizedQuestion,
        intent: IntentAnalysis,
        schema: RetrievedSchema,
    ) -> RoutingDecision:
        """Decide which model generates the SQL and which one checks it."""
        local_available = self._providers.try_get(LOCAL) is not None
        azure_available = self._providers.try_get(AZURE) is not None
        if not local_available and not azure_available:
            raise LLMUnavailableError(
                "No language model is available. Configure Azure OpenAI or enable the local model."
            )

        score, features = self.complexity(question, intent, schema)

        if not self._settings.enabled:
            primary = (
                self._default
                if self._providers.try_get(self._default)
                else (AZURE if azure_available else LOCAL)
            )
            decision = RoutingDecision(
                primary=primary,
                verifier=None,
                complexity=score,
                features=features,
                reason="routing is disabled, the default provider is used",
            )
            self._log(decision)
            return decision

        if score <= self._settings.local_max_complexity and local_available:
            primary = LOCAL
            verifier = AZURE if azure_available else None
            reason = "the question scored below the local threshold"
        else:
            primary = AZURE if azure_available else LOCAL
            if primary == AZURE and local_available and self._settings.local_can_verify:
                verifier = LOCAL
            elif primary == LOCAL and azure_available:
                verifier = AZURE
            else:
                verifier = None
            reason = (
                "the question scored above the local threshold"
                if azure_available
                else "only the local model is available"
            )

        decision = RoutingDecision(
            primary=primary,
            verifier=verifier,
            complexity=score,
            features=features,
            reason=reason,
        )
        self._log(decision)
        return decision

    def should_verify(self, decision: RoutingDecision, confidence: float) -> bool:
        """Return whether the verifier model should review a generated query."""
        if decision.verifier is None:
            return False
        if decision.complexity >= self._settings.always_verify_above_complexity:
            return True
        return confidence < self._settings.verify_below_confidence

    @staticmethod
    def _log(decision: RoutingDecision) -> None:
        logger.info(
            "routing_decision",
            primary=decision.primary,
            verifier=decision.verifier,
            complexity=decision.complexity,
            reason=decision.reason,
        )


def _saturate(value: float, saturation: int) -> float:
    """Map a count onto 0 to 1, reaching 1 at the saturation point."""
    if saturation <= 0:
        return 0.0
    return min(float(value) / float(saturation), 1.0)
