"""Question normalisation.

The first stage every question passes through. It does four things, in order,
and the order matters:

1. normalises unicode and strips invisible characters, so two questions that
   look identical are identical, and so zero width characters cannot be used
   to hide words from the injection screen
2. removes anything shaped like one of the prompt's own block delimiters, so
   the question cannot appear to close its block
3. screens for prompt injection, rejecting or warning as configured
4. resolves a follow up question into one that stands on its own, so every
   later stage sees a complete question and nothing depends on hidden state
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from nl2sql.config.settings import ConversationSettings, LimitsSettings
from nl2sql.core.exceptions import InputValidationError, LLMError
from nl2sql.llm.base import LLMProvider
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import FollowupRewrite, NormalizedQuestion
from nl2sql.security.prompt_injection import PromptInjectionDetector, neutralise_delimiters

logger = get_logger(__name__)


class QuestionNormalizer:
    """Cleans, screens and, where needed, rewrites the incoming question."""

    def __init__(
        self,
        *,
        limits: LimitsSettings,
        detector: PromptInjectionDetector,
        prompts: PromptRegistry,
        conversation: ConversationSettings,
    ) -> None:
        self._limits = limits
        self._detector = detector
        self._prompts = prompts
        self._conversation = conversation

    def clean(self, question: str) -> str:
        """Return the question with unicode normalised and invisible characters removed."""
        if question is None:
            raise InputValidationError("A question is required.")
        text = unicodedata.normalize("NFKC", question)
        # Cc is control characters, Cf is formatting characters such as zero
        # width joiners and right to left marks, both of which can hide text
        # from a reader and from a pattern.
        text = "".join(
            character
            for character in text
            if unicodedata.category(character) not in {"Cc", "Cf"} or character in "\n\t"
        )
        text = " ".join(text.split())
        return text.strip()

    async def normalize(
        self,
        question: str,
        *,
        history: Sequence[str] = (),
        provider: LLMProvider | None = None,
    ) -> NormalizedQuestion:
        """Return the cleaned, screened and standalone form of the question."""
        text = self.clean(question)
        if not text:
            raise InputValidationError("The question is empty.")
        if len(text) > self._limits.max_question_chars:
            raise InputValidationError(
                f"The question is longer than the {self._limits.max_question_chars} "
                "character limit."
            )

        text = neutralise_delimiters(text, self._prompts.untrusted_delimiters)
        findings = self._detector.enforce(text)
        result = NormalizedQuestion(original=question, text=text, injection_findings=findings)
        if findings:
            result.warnings.append(
                "The question matched prompt injection screening and was answered as "
                "a plain question."
            )
            logger.warning("prompt_injection_warning", rules=list(findings))

        if history and self._conversation.enabled and self._conversation.rewrite_followups:
            result.text = await self._rewrite(text, history, provider, result)
            result.is_followup = result.text != text

        return result

    async def _rewrite(
        self,
        text: str,
        history: Sequence[str],
        provider: LLMProvider | None,
        result: NormalizedQuestion,
    ) -> str:
        """Ask a model to make a follow up question self contained."""
        if provider is None or not provider.is_available():
            return text
        prompt = self._prompts.render(
            "followup_rewrite",
            conversation="\n".join(history),
            question=text,
        )
        try:
            rewritten = await provider.generate_structured(prompt, FollowupRewrite)
        except LLMError as exc:
            logger.warning("followup_rewrite_failed", error_type=type(exc).__name__)
            result.warnings.append(
                "The previous turns could not be taken into account for this question."
            )
            return text
        candidate = self.clean(rewritten.output.standalone_question)
        if not candidate or len(candidate) > self._limits.max_question_chars:
            return text
        # The rewrite is model output, so it is screened exactly like caller
        # input before it is allowed to become the question.
        candidate = neutralise_delimiters(candidate, self._prompts.untrusted_delimiters)
        self._detector.enforce(candidate)
        return candidate
