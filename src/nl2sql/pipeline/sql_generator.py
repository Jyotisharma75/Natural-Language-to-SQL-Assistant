"""SQL generation.

A thin stage on purpose. It renders a versioned prompt, calls whichever
provider the router chose, and returns the structured result. It holds no
instructions of its own, so changing how the model is asked is a change to a
prompt file rather than to code.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from nl2sql.config.settings import LimitsSettings
from nl2sql.llm.base import LLMProvider, LLMResult
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.pipeline.models import GeneratedSQL, ValidationIssue

#: Shown to the model when a question has no conversation behind it.
NO_CONVERSATION = "No earlier turns."


class SQLGenerator:
    """Asks a provider for one query, or for a correction to one."""

    def __init__(self, prompts: PromptRegistry, *, limits: LimitsSettings) -> None:
        self._prompts = prompts
        self._limits = limits

    async def generate(
        self,
        provider: LLMProvider,
        *,
        question: str,
        intent: str,
        schema_context: str,
        dialect_label: str,
        conversation: str | None = None,
        today: date | None = None,
    ) -> LLMResult[GeneratedSQL]:
        """Generate a query for one question."""
        prompt = self._prompts.render(
            "sql_generation",
            dialect=dialect_label,
            schema=schema_context,
            question=question,
            intent=intent,
            conversation=conversation or NO_CONVERSATION,
            max_rows=self._limits.max_rows,
            today=(today or date.today()).isoformat(),
        )
        return await provider.generate_structured(prompt, GeneratedSQL)

    async def repair(
        self,
        provider: LLMProvider,
        *,
        question: str,
        schema_context: str,
        previous_sql: str,
        issues: Sequence[ValidationIssue],
        dialect_label: str,
    ) -> LLMResult[GeneratedSQL]:
        """Ask for a correction, given the validator's own reasons."""
        prompt = self._prompts.render(
            "sql_repair",
            dialect=dialect_label,
            schema=schema_context,
            question=question,
            previous_sql=previous_sql or "no query was produced",
            issues=format_issues(issues),
            max_rows=self._limits.max_rows,
        )
        return await provider.generate_structured(prompt, GeneratedSQL)


def format_issues(issues: Sequence[ValidationIssue]) -> str:
    """Render validation issues as the numbered list the repair prompt expects."""
    if not issues:
        return "No specific issues were recorded."
    return "\n".join(
        f"{index}. [{issue.code}] {issue.message}" for index, issue in enumerate(issues, start=1)
    )
