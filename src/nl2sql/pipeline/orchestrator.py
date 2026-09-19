"""The query pipeline.

Runs the stages in order and owns everything cross cutting: the query
identifier, stage timings, token accounting, the audit record and conversation
memory. Each stage is a component that can be built and tested on its own; this
module is only the sequence and the bookkeeping.

    question
      -> normalise and screen
      -> understand intent
      -> discover schema and retrieve the relevant tables
      -> route between models
      -> generate, validate, repair, verify, choose
      -> execute
      -> validate and mask the result
      -> write the answer

``validate_only`` stops after a query has been chosen, which is what the
validate endpoint needs: the full safety analysis with nothing executed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from nl2sql.config.settings import Settings
from nl2sql.conversation.store import ConversationService
from nl2sql.core.context import new_id, reset_query_id, set_query_id
from nl2sql.core.exceptions import AppError
from nl2sql.db.engine import dialect_label
from nl2sql.llm.base import LLMProvider
from nl2sql.llm.factory import ProviderRegistry
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.llm.usage import end_collection, start_collection
from nl2sql.metadata.service import MetadataService
from nl2sql.observability.audit import AuditEntry, AuditLogger
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.answer_generator import AnswerGenerator
from nl2sql.pipeline.intent import IntentAnalyzer
from nl2sql.pipeline.models import (
    QueryFilter,
    QueryOutcome,
    SelectionOutcome,
)
from nl2sql.pipeline.normalizer import QuestionNormalizer
from nl2sql.pipeline.result_formatter import ResultFormatter
from nl2sql.pipeline.result_validator import ResultValidator
from nl2sql.pipeline.router import ModelRouter
from nl2sql.pipeline.schema_retriever import SchemaRetriever
from nl2sql.pipeline.sql_executor import SQLExecutor
from nl2sql.pipeline.sql_verifier import SQLVerifier
from nl2sql.security.auth import Principal

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryCommand:
    """One request to answer a question."""

    question: str
    principal: Principal
    conversation_id: str | None = None
    filters: tuple[QueryFilter, ...] = ()
    validate_only: bool = False
    today: date | None = None


@dataclass(slots=True)
class _Timings:
    """Stage durations in milliseconds."""

    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = (time.perf_counter() - started) * 1000


class QueryPipeline:
    """Answers one question end to end."""

    def __init__(
        self,
        *,
        settings: Settings,
        normalizer: QuestionNormalizer,
        intent_analyzer: IntentAnalyzer,
        metadata: MetadataService,
        retriever: SchemaRetriever,
        router: ModelRouter,
        verifier: SQLVerifier,
        executor: SQLExecutor,
        result_validator: ResultValidator,
        formatter: ResultFormatter,
        answer_generator: AnswerGenerator,
        conversation: ConversationService,
        audit: AuditLogger,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
    ) -> None:
        self._settings = settings
        self._normalizer = normalizer
        self._intent = intent_analyzer
        self._metadata = metadata
        self._retriever = retriever
        self._router = router
        self._verifier = verifier
        self._executor = executor
        self._result_validator = result_validator
        self._formatter = formatter
        self._answer = answer_generator
        self._conversation = conversation
        self._audit = audit
        self._providers = providers
        self._prompts = prompts

    async def run(self, command: QueryCommand) -> QueryOutcome:
        """Answer one question, recording an audit entry either way."""
        query_id = new_id()
        token = set_query_id(query_id)
        usage, usage_token = start_collection()
        timings = _Timings()
        started = time.perf_counter()

        entry = AuditEntry(
            query_id=query_id,
            principal_id=command.principal.id,
            tenant_id=command.principal.tenant_id,
            question=command.question,
            prompt_versions=self._prompts.active_versions(),
        )

        try:
            outcome = await self._run(command, query_id, timings, entry)
        except AppError as exc:
            entry.error_code = exc.code
            entry.execution_status = "failed"
            entry.latency_ms = (time.perf_counter() - started) * 1000
            entry.stage_timings = timings.values
            entry.prompt_tokens = usage.usage.prompt_tokens
            entry.completion_tokens = usage.usage.completion_tokens
            await self._audit.record(entry)
            raise
        except Exception as exc:
            entry.error_code = "internal_error"
            entry.execution_status = "failed"
            entry.latency_ms = (time.perf_counter() - started) * 1000
            entry.stage_timings = timings.values
            logger.exception("pipeline_failed", error_type=type(exc).__name__)
            await self._audit.record(entry)
            raise
        else:
            outcome.total_time_ms = (time.perf_counter() - started) * 1000
            outcome.stage_timings = timings.values
            entry.latency_ms = outcome.total_time_ms
            entry.stage_timings = timings.values
            entry.prompt_tokens = usage.usage.prompt_tokens
            entry.completion_tokens = usage.usage.completion_tokens
            await self._audit.record(entry)
            return outcome
        finally:
            end_collection(usage_token)
            reset_query_id(token)

    # -- the sequence ------------------------------------------------------
    async def _run(
        self,
        command: QueryCommand,
        query_id: str,
        timings: _Timings,
        entry: AuditEntry,
    ) -> QueryOutcome:
        auxiliary = self._providers.first_available(
            self._settings.llm.auxiliary_provider_preference
        )

        with timings.stage("history"):
            history = await self._conversation.history(
                command.conversation_id, command.principal.id
            )

        with timings.stage("normalize"):
            question = await self._normalizer.normalize(
                command.question, history=history, provider=auxiliary
            )

        with timings.stage("intent"):
            intent = await self._intent.analyze(question.text, provider=auxiliary)
        entry.intent = intent.intent_type

        with timings.stage("schema"):
            catalog = await self._metadata.catalog_async()

        with timings.stage("retrieval"):
            schema = await self._retriever.retrieve(
                question.text, intent, catalog, provider=auxiliary
            )

        with timings.stage("routing"):
            decision = self._router.route(question, intent, schema)
        entry.complexity = decision.complexity
        entry.primary_provider = decision.primary
        entry.primary_model = self._model_name(decision.primary)
        if decision.verifier:
            entry.verifier_provider = decision.verifier
            entry.verifier_model = self._model_name(decision.verifier)

        conversation_context = "\n".join(history) if history else None

        with timings.stage("generation"):
            selection = await self._verifier.produce(
                question=question.text,
                intent=intent.intent_type,
                schema=schema,
                catalog=catalog,
                decision=decision,
                dialect=self._executor.sqlglot_dialect,
                dialect_label=dialect_label(self._executor.dialect),
                tenant_id=command.principal.tenant_id,
                filters=command.filters,
                conversation=conversation_context,
                today=command.today,
            )

        winner = selection.winner
        entry.generated_sql = winner.report.display_sql
        entry.tables_used = winner.report.tables
        entry.validation_passed = True
        entry.confidence = selection.confidence
        entry.validation_issues = [
            {"code": issue.code, "message": issue.message} for issue in winner.report.warnings
        ]

        warnings = list(question.warnings) + list(selection.warnings)

        outcome = QueryOutcome(
            query_id=query_id,
            sql=winner.report.display_sql,
            confidence=selection.confidence,
            valid=True,
            warnings=warnings,
            intent=intent.intent_type,
            routing=decision,
            tables_used=winner.report.tables,
            columns_used=winner.report.columns,
            reasoning_summary=winner.generated.reasoning_summary,
            issues=list(winner.report.issues),
        )

        if command.validate_only:
            entry.execution_status = "not_executed"
            return outcome

        with timings.stage("execution"):
            raw = await self._executor.execute(
                winner.report.execution_sql,
                winner.report.parameters,
                max_rows=winner.report.row_limit_applied,
                tenant_id=command.principal.tenant_id,
            )
        entry.execution_status = "succeeded"
        entry.row_count = raw.row_count
        entry.truncated = raw.truncated

        with timings.stage("result_validation"):
            checked, result_warnings = self._result_validator.check(raw, winner.report)
            formatted = self._formatter.format(checked)
        warnings.extend(result_warnings)

        with timings.stage("answer"):
            answer = await self._answer.generate(
                self._answer_provider(winner.provider, auxiliary),
                question=question.text,
                sql_summary=winner.generated.reasoning_summary,
                result=formatted,
                warnings=warnings,
                truncated=checked.truncated,
            )

        outcome.answer = answer
        outcome.columns = formatted.columns
        outcome.rows = formatted.rows
        outcome.row_count = checked.row_count
        outcome.truncated = checked.truncated
        outcome.execution_time_ms = raw.execution_ms
        outcome.warnings = warnings
        outcome.executed = True

        await self._conversation.record(
            command.conversation_id,
            command.principal.id,
            {
                "question": command.question,
                "standalone_question": question.text,
                "sql": winner.report.display_sql,
                "answer": answer,
                "tables_used": list(winner.report.tables),
            },
        )
        return outcome

    # -- helpers -----------------------------------------------------------
    def _model_name(self, provider_name: str) -> str | None:
        provider = self._providers.try_get(provider_name)
        return provider.model_name if provider else None

    def _answer_provider(
        self, winning_provider: str, auxiliary: LLMProvider | None
    ) -> LLMProvider | None:
        """Choose which model writes the answer.

        The default follows the query: the model that wrote the winning query
        already has the context to describe its result, and after an
        escalation that is not the model the router first chose.
        """
        configured = self._settings.answer.provider
        if configured == "primary":
            return self._providers.try_get(winning_provider) or auxiliary
        return self._providers.try_get(configured) or auxiliary

    @staticmethod
    def selection_warnings(selection: SelectionOutcome) -> Sequence[str]:
        """Return the warnings recorded while choosing a query."""
        return selection.warnings
