"""Composition root.

Every component is constructed here and nowhere else, which is what keeps the
rest of the code free of global state and makes each piece testable on its
own. A test builds a container with its own engines and its own model
providers and gets the real pipeline, not a simulation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from nl2sql.config.secrets import resolve_secret
from nl2sql.config.settings import Settings
from nl2sql.conversation.memory_store import InMemoryConversationStore
from nl2sql.conversation.sqlalchemy_store import SQLAlchemyConversationStore
from nl2sql.conversation.store import ConversationService, ConversationStore
from nl2sql.core.masking import Masker
from nl2sql.db.engine import create_app_engine, create_query_engine, sqlglot_dialect
from nl2sql.db.repositories import AuditRepository, ConversationRepository, create_all
from nl2sql.llm.factory import ProviderRegistry, build_provider_registry
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.llm.usage import UsageTracker
from nl2sql.metadata.introspector import SchemaIntrospector
from nl2sql.metadata.service import MetadataService
from nl2sql.observability.audit import AuditLogger
from nl2sql.observability.logging import configure_logging, get_logger
from nl2sql.observability.metrics import MetricsRegistry
from nl2sql.pipeline.answer_generator import AnswerGenerator
from nl2sql.pipeline.intent import IntentAnalyzer
from nl2sql.pipeline.normalizer import QuestionNormalizer
from nl2sql.pipeline.orchestrator import QueryPipeline
from nl2sql.pipeline.result_formatter import ResultFormatter
from nl2sql.pipeline.result_validator import ResultValidator
from nl2sql.pipeline.router import ModelRouter
from nl2sql.pipeline.schema_retriever import SchemaRetriever
from nl2sql.pipeline.sql_executor import SQLExecutor
from nl2sql.pipeline.sql_generator import SQLGenerator
from nl2sql.pipeline.sql_validator import SQLValidator
from nl2sql.pipeline.sql_verifier import SQLVerifier
from nl2sql.security.auth import Authenticator, build_authenticator
from nl2sql.security.policy import AccessPolicy
from nl2sql.security.prompt_injection import PromptInjectionDetector
from nl2sql.security.sql_guard import SQLGuard
from nl2sql.security.tenancy import TenantScoper

logger = get_logger(__name__)

#: Prompts the pipeline cannot run without.
REQUIRED_PROMPTS = (
    "sql_generation",
    "sql_repair",
    "sql_verification",
    "answer_generation",
    "table_selection",
    "followup_rewrite",
    "structured_output",
    "intent_analysis",
)


@dataclass(slots=True)
class Container:
    """Everything the application needs, already wired together."""

    settings: Settings
    query_engine: Engine
    app_engine: Engine
    masker: Masker
    metrics: MetricsRegistry
    prompts: PromptRegistry
    providers: ProviderRegistry
    policy: AccessPolicy
    metadata: MetadataService
    executor: SQLExecutor
    validator: SQLValidator
    pipeline: QueryPipeline
    authenticator: Authenticator
    conversation: ConversationService
    audit: AuditLogger

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        query_engine: Engine | None = None,
        app_engine: Engine | None = None,
        providers: ProviderRegistry | None = None,
        azure_client_factory: Any | None = None,
        configure_logs: bool = True,
        create_app_tables: bool = False,
    ) -> Container:
        """Construct every component from configuration."""
        masker = Masker(
            sensitive_keys=settings.observability.sensitive_keys,
            patterns=settings.observability.mask_patterns,
        )
        if configure_logs:
            configure_logging(
                level=settings.observability.log_level,
                log_format=settings.observability.log_format,
                service_name=settings.observability.service_name,
                environment=settings.env,
                masker=masker,
            )

        metrics = MetricsRegistry()
        usage = UsageTracker(metrics)

        prompts = PromptRegistry(
            Path(settings.prompts.directory),
            active_versions=settings.prompts.versions,
            untrusted_delimiters=tuple(settings.prompts.untrusted_delimiters),
        )
        prompts.require(REQUIRED_PROMPTS)

        engine = query_engine or create_query_engine(settings.database)
        application_engine = app_engine or create_app_engine(settings.app_database)
        if create_app_tables:
            create_all(application_engine)

        policy = AccessPolicy(settings.security)
        introspector = SchemaIntrospector(
            engine,
            include_views=settings.schema_cache.include_views,
            include_indexes=settings.schema_cache.include_indexes,
            schema_filter=policy.is_schema_allowed,
        )
        metadata = MetadataService(introspector, policy, settings.schema_cache)

        guard = SQLGuard(settings.security, policy)
        scoper = TenantScoper(settings.tenancy)
        validator = SQLValidator(
            policy=policy,
            guard=guard,
            scoper=scoper,
            limits=settings.limits,
            security=settings.security,
        )
        executor = SQLExecutor(
            engine,
            settings=settings.execution,
            limits=settings.limits,
            tenancy=settings.tenancy,
            cost=settings.cost_estimation,
        )

        registry = providers or build_provider_registry(
            settings, usage=usage, prompts=prompts, azure_client_factory=azure_client_factory
        )
        router = ModelRouter(
            settings.routing, registry, default_provider=settings.llm.default_provider
        )
        generator = SQLGenerator(prompts, limits=settings.limits)
        verifier = SQLVerifier(
            generator=generator,
            validator=validator,
            router=router,
            providers=registry,
            prompts=prompts,
            settings=settings.routing,
            executor=executor,
            cost=settings.cost_estimation,
        )

        detector = PromptInjectionDetector(settings.security.prompt_injection)
        normalizer = QuestionNormalizer(
            limits=settings.limits,
            detector=detector,
            prompts=prompts,
            conversation=settings.conversation,
        )
        store = cls._build_conversation_store(settings, application_engine)
        conversation = ConversationService(settings.conversation, store, masker)

        audit = AuditLogger(
            repository=AuditRepository(application_engine)
            if settings.observability.audit_enabled
            else None,
            masker=masker,
            metrics=metrics,
            enabled=settings.observability.audit_enabled,
            log_sql=settings.observability.log_generated_sql,
            mask_sql=settings.observability.mask_sql_literals,
            dialect=sqlglot_dialect(engine.dialect.name),
        )

        pipeline = QueryPipeline(
            settings=settings,
            normalizer=normalizer,
            intent_analyzer=IntentAnalyzer(settings.intent, prompts),
            metadata=metadata,
            retriever=SchemaRetriever(settings.retrieval, prompts),
            router=router,
            verifier=verifier,
            executor=executor,
            result_validator=ResultValidator(settings.limits, settings.security),
            formatter=ResultFormatter(settings.answer),
            answer_generator=AnswerGenerator(prompts, settings.answer),
            conversation=conversation,
            audit=audit,
            providers=registry,
            prompts=prompts,
        )

        authenticator = build_authenticator(
            settings.api, resolve_secret(settings.api.api_key.keys_secret)
        )

        logger.info("container_built", **settings.public_summary())
        return cls(
            settings=settings,
            query_engine=engine,
            app_engine=application_engine,
            masker=masker,
            metrics=metrics,
            prompts=prompts,
            providers=registry,
            policy=policy,
            metadata=metadata,
            executor=executor,
            validator=validator,
            pipeline=pipeline,
            authenticator=authenticator,
            conversation=conversation,
            audit=audit,
        )

    @staticmethod
    def _build_conversation_store(
        settings: Settings, app_engine: Engine
    ) -> ConversationStore | None:
        """Build the configured conversation store, or none when disabled."""
        if not settings.conversation.enabled:
            return None
        if settings.conversation.store == "memory":
            return InMemoryConversationStore(
                max_turns=settings.conversation.max_turns,
                ttl_seconds=settings.conversation.ttl_seconds,
            )
        return SQLAlchemyConversationStore(
            ConversationRepository(app_engine),
            max_turns=settings.conversation.max_turns,
            ttl_seconds=settings.conversation.ttl_seconds,
        )

    async def warmup(self) -> None:
        """Do the slow work before the first request rather than during it."""
        if self.settings.schema_cache.warm_on_startup:
            try:
                catalog = await self.metadata.catalog_async()
                logger.info("schema_cache_warmed", table_count=len(catalog))
            except Exception as exc:
                logger.warning("schema_warmup_failed", error_type=type(exc).__name__)
        local = self.providers.try_get("local")
        if local is not None and self.settings.llm.local.preload:
            preload = getattr(local, "preload", None)
            if preload is not None:
                try:
                    await preload()
                except Exception as exc:
                    logger.warning("local_model_preload_failed", error_type=type(exc).__name__)

    def dispose(self) -> None:
        """Close both connection pools."""
        self.query_engine.dispose()
        self.app_engine.dispose()
