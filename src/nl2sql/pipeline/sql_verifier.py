"""Candidate production and cross model verification.

This is where routing turns into an actual guarantee about quality. The
deterministic validator can prove a query is safe and that every name in it
exists. It cannot tell whether the query answers the question. Four mechanisms
cover that gap, and each one is used only when it is worth its cost:

1. **Repair.** When validation fails, the same model is shown the exact
   reasons and asked to correct them. Precise feedback fixes most failures in
   one attempt, and it costs one call.
2. **Escalation.** When repair does not produce a valid query, the other model
   generates from scratch. A local model that cannot express a window function
   is not helped by being asked again.
3. **Verification.** When confidence is low or the question scored as complex,
   the other model judges whether the query answers the question. Independence
   is the point: the model that wrote the query is a poor judge of it.
4. **Dry run.** The candidate is bound by the database and returns no rows,
   which proves every name, type and aggregate resolves for a fraction of the
   cost of running it.

Whatever survives is scored on those four signals with configured weights, and
the best candidate wins. When nothing is valid, the request fails with the
reasons rather than executing something doubtful.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from nl2sql.config.settings import CostEstimationSettings, RoutingSettings
from nl2sql.core.exceptions import (
    ExecutionError,
    LLMError,
    QueryCostExceededError,
    SQLValidationError,
)
from nl2sql.llm.factory import ProviderRegistry
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.metadata.models import SchemaCatalog
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import (
    CandidateSQL,
    GeneratedSQL,
    QueryFilter,
    RetrievedSchema,
    RoutingDecision,
    SelectionOutcome,
    ValidationIssue,
    ValidationReport,
    VerificationVerdict,
)
from nl2sql.pipeline.router import ModelRouter
from nl2sql.pipeline.sql_executor import SQLExecutor
from nl2sql.pipeline.sql_generator import SQLGenerator
from nl2sql.pipeline.sql_validator import SQLValidator

logger = get_logger(__name__)


class SQLVerifier:
    """Produces candidate queries and chooses the one to run."""

    def __init__(
        self,
        *,
        generator: SQLGenerator,
        validator: SQLValidator,
        router: ModelRouter,
        providers: ProviderRegistry,
        prompts: PromptRegistry,
        settings: RoutingSettings,
        executor: SQLExecutor | None = None,
        cost: CostEstimationSettings | None = None,
    ) -> None:
        self._generator = generator
        self._validator = validator
        self._router = router
        self._providers = providers
        self._prompts = prompts
        self._settings = settings
        self._executor = executor
        self._cost = cost

    async def produce(
        self,
        *,
        question: str,
        intent: str,
        schema: RetrievedSchema,
        catalog: SchemaCatalog,
        decision: RoutingDecision,
        dialect: str,
        dialect_label: str,
        tenant_id: str | None = None,
        filters: Sequence[QueryFilter] = (),
        conversation: str | None = None,
        today: date | None = None,
    ) -> SelectionOutcome:
        """Generate, check and choose a query for one question."""
        context = _Context(
            question=question,
            intent=intent,
            schema=schema,
            catalog=catalog,
            dialect=dialect,
            dialect_label=dialect_label,
            tenant_id=tenant_id,
            filters=tuple(filters),
            conversation=conversation,
            today=today,
        )

        candidates: list[CandidateSQL] = []
        warnings: list[str] = []
        repairs = 0
        escalated = False
        verified = False

        current = await self._generate(decision.primary, context, origin="primary")
        candidates.append(current)

        while not current.valid and repairs < self._settings.max_repair_attempts:
            repaired = await self._repair(decision.primary, current, context)
            repairs += 1
            candidates.append(repaired)
            current = repaired

        verifier_name = decision.verifier
        if not current.valid and verifier_name and self._settings.escalate_on_failure:
            escalated = True
            alternative = await self._generate(verifier_name, context, origin="escalation")
            candidates.append(alternative)
            warnings.append(
                "The first model could not produce a valid query, so a second model was used."
            )
        elif (
            current.valid
            and verifier_name
            and self._router.should_verify(decision, current.generated.confidence)
        ):
            verified = True
            current.verdict = await self._verify(verifier_name, current, context)
            if (
                current.verdict is not None
                and not current.verdict.agrees
                and self._settings.cross_generate_on_disagreement
            ):
                escalated = True
                alternative = await self._generate(verifier_name, context, origin="cross_check")
                candidates.append(alternative)

        await self._dry_run_candidates(candidates, context)
        winner = self._choose(candidates)

        if winner is None or not winner.valid:
            failed = candidates[-1] if candidates else None
            issues = list(failed.report.issues) if failed else []
            detail = failed.generation_error if failed else None
            raise SQLValidationError(
                "No valid query could be produced for this question.",
                details={
                    "issues": [
                        {"code": issue.code, "message": issue.message}
                        for issue in issues
                        if issue.severity == "error"
                    ],
                    "error": detail,
                },
                public_message=_refusal_message(issues, detail),
            )

        await self._check_cost(winner)

        confidence = self._confidence(winner)
        warnings.extend(self._warnings_for(winner))
        logger.info(
            "candidate_selected",
            origin=winner.origin,
            provider=winner.provider,
            candidates=len(candidates),
            repairs=repairs,
            escalated=escalated,
            verified=verified,
            confidence=confidence,
        )
        return SelectionOutcome(
            winner=winner,
            candidates=candidates,
            confidence=confidence,
            warnings=warnings,
            verified=verified,
            escalated=escalated,
            repairs=repairs,
        )

    # -- candidate production ---------------------------------------------
    async def _generate(
        self, provider_name: str, context: _Context, *, origin: str
    ) -> CandidateSQL:
        """Generate one candidate and validate it."""
        provider = self._providers.get(provider_name)
        try:
            result = await self._generator.generate(
                provider,
                question=context.question,
                intent=context.intent,
                schema_context=context.schema.context,
                dialect_label=context.dialect_label,
                conversation=context.conversation,
                today=context.today,
            )
        except LLMError as exc:
            logger.warning(
                "sql_generation_failed",
                provider=provider_name,
                origin=origin,
                error_type=type(exc).__name__,
            )
            return self._failed_candidate(provider, origin, "generation_failed", exc.message)

        return self._validate_candidate(result.output, provider, origin, context)

    async def _repair(
        self, provider_name: str, previous: CandidateSQL, context: _Context
    ) -> CandidateSQL:
        """Ask the same model to correct a query the validator refused."""
        provider = self._providers.get(provider_name)
        try:
            result = await self._generator.repair(
                provider,
                question=context.question,
                schema_context=context.schema.context,
                previous_sql=previous.generated.sql,
                issues=previous.report.errors,
                dialect_label=context.dialect_label,
            )
        except LLMError as exc:
            logger.warning(
                "sql_repair_failed", provider=provider_name, error_type=type(exc).__name__
            )
            return self._failed_candidate(provider, "repair", "repair_failed", exc.message)

        return self._validate_candidate(result.output, provider, "repair", context)

    def _validate_candidate(
        self,
        generated: GeneratedSQL,
        provider: object,
        origin: str,
        context: _Context,
    ) -> CandidateSQL:
        """Validate generated SQL, handling a model that declined to answer."""
        name = getattr(provider, "name", "unknown")
        model = getattr(provider, "model_name", "unknown")

        if not generated.sql.strip():
            report = ValidationReport(valid=False)
            report.issues.append(
                ValidationIssue(
                    "no_sql_produced",
                    "The model did not produce a query for this question.",
                )
            )
            for warning in generated.warnings:
                report.issues.append(ValidationIssue("model_warning", warning, severity="warning"))
            return CandidateSQL(
                generated=generated,
                provider=name,
                model=model,
                report=report,
                origin=origin,
            )

        report = self._validator.validate(
            generated.sql,
            context.catalog,
            dialect=context.dialect,
            tenant_id=context.tenant_id,
            filters=context.filters,
            declared_tables=generated.tables_used,
        )
        return CandidateSQL(
            generated=generated, provider=name, model=model, report=report, origin=origin
        )

    @staticmethod
    def _failed_candidate(provider: object, origin: str, code: str, message: str) -> CandidateSQL:
        report = ValidationReport(valid=False)
        report.issues.append(ValidationIssue(code, message))
        return CandidateSQL(
            generated=GeneratedSQL(),
            provider=getattr(provider, "name", "unknown"),
            model=getattr(provider, "model_name", "unknown"),
            report=report,
            origin=origin,
            generation_error=message,
        )

    # -- verification ------------------------------------------------------
    async def _verify(
        self, provider_name: str, candidate: CandidateSQL, context: _Context
    ) -> VerificationVerdict | None:
        """Have the other model judge whether the query answers the question."""
        provider = self._providers.try_get(provider_name)
        if provider is None:
            return None
        prompt = self._prompts.render(
            "sql_verification",
            dialect=context.dialect_label,
            schema=context.schema.context,
            question=context.question,
            sql=candidate.report.display_sql or candidate.generated.sql,
        )
        try:
            result = await provider.generate_structured(prompt, VerificationVerdict)
        except LLMError as exc:
            logger.warning(
                "sql_verification_failed",
                provider=provider_name,
                error_type=type(exc).__name__,
            )
            return None
        logger.info(
            "sql_verified",
            provider=provider_name,
            agrees=result.output.agrees,
            confidence=result.output.confidence,
        )
        return result.output

    async def _dry_run_candidates(
        self, candidates: Sequence[CandidateSQL], context: _Context
    ) -> None:
        """Bind each valid candidate against the database without reading rows."""
        if not self._settings.dry_run_enabled or self._executor is None:
            return
        for candidate in candidates:
            if not candidate.valid:
                continue
            try:
                candidate.dry_run_ok = await self._executor.dry_run(
                    candidate.report.execution_sql,
                    candidate.report.parameters,
                    tenant_id=context.tenant_id,
                )
            except ExecutionError as exc:
                candidate.dry_run_ok = False
                candidate.report.valid = False
                candidate.report.issues.append(
                    ValidationIssue(
                        "dry_run_failed",
                        f"The database rejected the query: {exc.public_message}",
                    )
                )
                logger.info("dry_run_rejected", origin=candidate.origin, code=exc.code)

    async def _check_cost(self, winner: CandidateSQL) -> None:
        """Refuse a query the optimiser expects to be too expensive."""
        if self._cost is None or not self._cost.enabled or self._executor is None:
            return
        estimate = await self._executor.estimate_cost(
            winner.report.execution_sql, winner.report.parameters
        )
        if estimate is None:
            return
        logger.info("query_cost_estimated", estimated_cost=estimate)
        if estimate > self._cost.max_estimated_cost:
            raise QueryCostExceededError(
                f"The estimated query cost {estimate:.1f} exceeds the ceiling "
                f"{self._cost.max_estimated_cost:.1f}.",
                details={"estimated_cost": estimate},
                public_message=(
                    "The query needed to answer this question is too expensive to run. "
                    "Try narrowing it, for example to a shorter period."
                ),
            )

    # -- selection ---------------------------------------------------------
    def _choose(self, candidates: Sequence[CandidateSQL]) -> CandidateSQL | None:
        """Score every candidate and return the best valid one."""
        weights = self._settings.selection_weights
        best: CandidateSQL | None = None
        for candidate in candidates:
            verdict_score = 0.5
            if candidate.verdict is not None:
                verdict_score = (
                    candidate.verdict.confidence
                    if candidate.verdict.agrees
                    else 1.0 - candidate.verdict.confidence
                )
            dry_run_score = 0.5 if candidate.dry_run_ok is None else float(candidate.dry_run_ok)
            candidate.score = round(
                weights.validation * float(candidate.valid)
                + weights.dry_run * dry_run_score
                + weights.verifier * verdict_score
                + weights.confidence * candidate.generated.confidence,
                4,
            )
            if not candidate.valid:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _confidence(self, winner: CandidateSQL) -> float:
        """Combine the signals into the confidence reported to the caller."""
        confidence = winner.generated.confidence
        if winner.verdict is not None:
            if winner.verdict.agrees:
                confidence = (confidence + winner.verdict.confidence) / 2
            else:
                confidence *= 1.0 - 0.5 * winner.verdict.confidence
        if winner.dry_run_ok is True:
            confidence = min(1.0, confidence + 0.05)
        if winner.report.warnings:
            confidence *= 0.95
        if winner.origin in {"repair", "escalation", "cross_check"}:
            confidence *= 0.95
        return round(max(0.0, min(1.0, confidence)), 3)

    @staticmethod
    def _warnings_for(winner: CandidateSQL) -> list[str]:
        """Collect the warnings a caller should see about the chosen query."""
        warnings = list(winner.generated.warnings)
        warnings.extend(issue.message for issue in winner.report.warnings)
        if winner.verdict is not None and not winner.verdict.agrees:
            detail = "; ".join(winner.verdict.issues[:3])
            warnings.append(
                "A second model questioned whether this query answers the question"
                + (f": {detail}" if detail else ".")
            )
        return warnings


def _refusal_message(issues: Sequence[ValidationIssue], detail: str | None) -> str:
    """Build a caller facing explanation of why no query could be produced."""
    errors = [issue for issue in issues if issue.severity == "error"]
    if not errors:
        return detail or "A safe query could not be produced for this question."
    first = errors[0]
    if first.code in {"no_sql_produced", "generation_failed", "repair_failed"}:
        return (
            "The question could not be turned into a query against the available "
            "tables. Try naming the measure or the period more explicitly."
        )
    return f"A safe query could not be produced: {first.message}"


class _Context:
    """The per request values every candidate needs, gathered once."""

    __slots__ = (
        "catalog",
        "conversation",
        "dialect",
        "dialect_label",
        "filters",
        "intent",
        "question",
        "schema",
        "tenant_id",
        "today",
    )

    def __init__(
        self,
        *,
        question: str,
        intent: str,
        schema: RetrievedSchema,
        catalog: SchemaCatalog,
        dialect: str,
        dialect_label: str,
        tenant_id: str | None,
        filters: tuple[QueryFilter, ...],
        conversation: str | None,
        today: date | None,
    ) -> None:
        self.question = question
        self.intent = intent
        self.schema = schema
        self.catalog = catalog
        self.dialect = dialect
        self.dialect_label = dialect_label
        self.tenant_id = tenant_id
        self.filters = filters
        self.conversation = conversation
        self.today = today
