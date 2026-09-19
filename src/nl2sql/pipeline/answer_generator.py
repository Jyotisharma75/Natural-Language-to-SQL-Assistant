"""Answer generation.

Turns the result set into a sentence or two. Three things make this stage
safer than it looks:

* it sees only what survived result validation, so masked values are already
  masked and oversized values already trimmed
* how much data leaves the process is configuration. A deployment that will
  not send row values to a hosted model sets ``send_rows_to_llm`` to false and
  the model receives only counts, ranges and totals per column
* if the model fails or is unavailable, a deterministic summary is produced
  instead. A query that ran successfully should not be reported as a failure
  because a sentence could not be written about it
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from nl2sql.config.settings import AnswerSettings
from nl2sql.core.exceptions import LLMError
from nl2sql.llm.base import LLMProvider
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import AnswerOutput, FormattedResult

logger = get_logger(__name__)


class AnswerGenerator:
    """Writes the natural language answer."""

    def __init__(self, prompts: PromptRegistry, settings: AnswerSettings) -> None:
        self._prompts = prompts
        self._settings = settings

    async def generate(
        self,
        provider: LLMProvider | None,
        *,
        question: str,
        sql_summary: str,
        result: FormattedResult,
        warnings: Sequence[str] = (),
        truncated: bool = False,
    ) -> str:
        """Return the written answer, falling back to a deterministic summary."""
        fallback = self.summarise(result, truncated=truncated)
        if not self._settings.enabled or provider is None or not provider.is_available():
            return fallback

        prompt = self._prompts.render(
            "answer_generation",
            question=question,
            sql_summary=sql_summary or "ran a query against the reporting tables",
            columns=", ".join(column["name"] for column in result.columns) or "none",
            rows=self._render_rows(result),
            row_count=len(result.rows),
            truncated="yes" if truncated else "no",
            warnings="\n".join(warnings) or "none",
        )
        try:
            response = await provider.generate_structured(prompt, AnswerOutput)
        except LLMError as exc:
            logger.warning("answer_generation_failed", error_type=type(exc).__name__)
            return fallback
        answer = response.output.answer.strip()
        return answer or fallback

    # -- payload -----------------------------------------------------------
    def _render_rows(self, result: FormattedResult) -> str:
        """Render the rows, or a statistical summary when values must not leave."""
        if not result.rows:
            return "No rows were returned."
        if not self._settings.send_rows_to_llm:
            return self._render_statistics(result)
        limit = self._settings.max_rows_in_prompt
        shown = result.rows[:limit]
        names = [column["name"] for column in result.columns]
        lines = [json.dumps(dict(zip(names, row, strict=False)), default=str) for row in shown]
        if len(result.rows) > len(shown):
            lines.append(f"... {len(result.rows) - len(shown)} further rows are not shown")
        return "\n".join(lines)

    def _render_statistics(self, result: FormattedResult) -> str:
        """Describe the result without disclosing individual values."""
        lines = [f"Row count: {len(result.rows)}", "Column summaries:"]
        for index, column in enumerate(result.columns):
            values = [row[index] for row in result.rows if index < len(row)]
            present = [value for value in values if value is not None]
            numeric = [value for value in present if isinstance(value, int | float)]
            summary = [f"non null {len(present)} of {len(values)}"]
            if numeric:
                summary.append(f"min {min(numeric)}")
                summary.append(f"max {max(numeric)}")
                summary.append(f"sum {sum(numeric)}")
                summary.append(f"mean {sum(numeric) / len(numeric):.4g}")
            else:
                summary.append(f"distinct {len({str(value) for value in present})}")
            lines.append(f"  {column['name']}: {', '.join(summary)}")
        return "\n".join(lines)

    # -- fallback ----------------------------------------------------------
    def summarise(self, result: FormattedResult, *, truncated: bool = False) -> str:
        """Describe a result set without a model."""
        if not result.rows:
            return "The query ran successfully and no rows matched."

        column_names = [column["name"] for column in result.columns]
        if len(result.rows) == 1 and len(column_names) == 1:
            value = _readable(result.rows[0][0])
            return f"{column_names[0]} is {value}."

        sentence = (
            f"The query returned {len(result.rows)} rows with "
            f"{len(column_names)} columns: {', '.join(column_names)}."
        )
        if truncated:
            sentence += " The results were truncated at the configured row limit."
        return sentence


def _readable(value: Any) -> str:
    """Render a single value for a sentence."""
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
