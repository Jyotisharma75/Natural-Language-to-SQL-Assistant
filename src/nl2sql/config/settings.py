"""Typed settings.

Every tunable behaviour of the assistant is declared here and populated from
``configs/*.yaml`` plus ``NL2SQL_`` environment variables. Nothing in the
business logic reads a literal limit, table name, model name or credential.

Fields ending in ``_secret`` hold the NAME of an environment variable that
contains the secret, never the secret itself, so a settings dump or a
validation error cannot disclose a credential.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrySettings(_Section):
    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=0.5, ge=0)
    max_backoff_seconds: float = Field(default=8.0, ge=0)
    jitter_seconds: float = Field(default=0.25, ge=0)


class AppInfoSettings(_Section):
    name: str = "nl2sql-assistant"
    version: str = "1.0.0"


class DatabaseSettings(_Section):
    """The database that natural language questions are answered from."""

    url: str | None = Field(
        default=None,
        description="Full SQLAlchemy URL. When set it replaces the Azure SQL fields below.",
    )
    url_secret: str | None = Field(
        default=None, description="Environment variable holding a full SQLAlchemy URL."
    )
    driver: str = "ODBC Driver 18 for SQL Server"
    server: str | None = None
    port: int = 1433
    database: str | None = None
    auth_mode: Literal[
        "sql_password", "entra_default", "entra_managed_identity", "entra_service_principal"
    ] = "sql_password"
    username: str | None = None
    password_secret: str = "NL2SQL_DB_PASSWORD"
    managed_identity_client_id: str | None = None
    encrypt: bool = True
    trust_server_certificate: bool = False
    login_timeout_seconds: int = Field(default=30, ge=1)
    application_intent_read_only: bool = True
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    pool_recycle_seconds: int = Field(default=1800, ge=60)
    pool_pre_ping: bool = True
    odbc_extra: dict[str, str] = Field(default_factory=dict)


class AppDatabaseSettings(_Section):
    """The application owned database for audit and conversation tables."""

    url: str = "sqlite:///./data/nl2sql_app.db"
    url_secret: str | None = None
    echo: bool = False


class SchemaCacheSettings(_Section):
    ttl_seconds: int = Field(default=900, ge=0)
    include_views: bool = True
    include_indexes: bool = True
    serve_stale_on_error: bool = True
    warm_on_startup: bool = True


class PromptInjectionSettings(_Section):
    enabled: bool = True
    action: Literal["reject", "warn"] = "reject"
    patterns: dict[str, str] = Field(default_factory=dict)


class SecuritySettings(_Section):
    allowed_statements: list[str] = Field(default_factory=lambda: ["SELECT"])
    allowed_schemas: list[str] = Field(default_factory=list)
    system_schemas: list[str] = Field(default_factory=list)
    blocked_tables: list[str] = Field(default_factory=list)
    blocked_columns: list[str] = Field(default_factory=list)
    masked_columns: list[str] = Field(default_factory=list)
    mask_token: str = "***"
    blocked_functions: list[str] = Field(default_factory=list)
    blocked_keywords: list[str] = Field(default_factory=list)
    blocked_identifier_prefixes: list[str] = Field(default_factory=list)
    allow_select_star: bool = False
    allow_system_catalogs: bool = False
    allow_cross_database: bool = False
    allow_cross_join: bool = False
    allow_variables: bool = False
    allow_temp_tables: bool = False
    max_sql_chars: int = Field(default=20000, ge=100)
    prompt_injection: PromptInjectionSettings = Field(default_factory=PromptInjectionSettings)

    @field_validator("allowed_statements")
    @classmethod
    def _upper(cls, value: list[str]) -> list[str]:
        return [item.strip().upper() for item in value if item.strip()]


class LimitsSettings(_Section):
    max_rows: int = Field(default=1000, ge=1)
    max_execution_seconds: float = Field(default=30, gt=0)
    max_result_bytes: int = Field(default=5_000_000, ge=1024)
    max_cell_chars: int = Field(default=4000, ge=16)
    max_joins: int = Field(default=6, ge=0)
    max_subquery_depth: int = Field(default=3, ge=0)
    max_tables: int = Field(default=8, ge=1)
    max_question_chars: int = Field(default=1000, ge=10)


class CostEstimationSettings(_Section):
    enabled: bool = False
    max_estimated_cost: float = Field(default=50.0, gt=0)
    timeout_seconds: float = Field(default=10.0, gt=0)


class TenancySettings(_Section):
    enabled: bool = False
    tenant_column_patterns: list[str] = Field(default_factory=list)
    require_tenant: bool = True
    session_context_key: str | None = None


class ExecutionSettings(_Section):
    transient_error_markers: list[str] = Field(default_factory=list)
    timeout_error_markers: list[str] = Field(default_factory=list)
    permission_error_markers: list[str] = Field(default_factory=list)
    invalid_reference_markers: list[str] = Field(default_factory=list)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    fetch_batch_size: int = Field(default=500, ge=1)
    dry_run_timeout_seconds: float = Field(default=10.0, gt=0)


class AzureOpenAISettings(_Section):
    enabled: bool = True
    endpoint: str | None = None
    deployment: str | None = None
    api_version: str = "2024-10-21"
    api_key_secret: str = "AZURE_OPENAI_API_KEY"
    use_managed_identity: bool = False
    managed_identity_client_id: str | None = None
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    token_limit_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    send_temperature: bool = True
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=1200, ge=16)
    timeout_seconds: float = Field(default=60, gt=0)
    health_probe: bool = False
    retry: RetrySettings = Field(default_factory=RetrySettings)


class LocalModelSettings(_Section):
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    revision: str = "main"
    device: str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16", "auto"] = "float32"
    max_new_tokens: int = Field(default=512, ge=16)
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_seconds: float = Field(default=180, gt=0)
    cache_dir: str | None = None
    trust_remote_code: bool = False
    num_threads: int | None = Field(default=None, ge=1)
    max_concurrency: int = Field(default=1, ge=1)
    preload: bool = False
    retry: RetrySettings = Field(default_factory=lambda: RetrySettings(max_attempts=1))


ProviderName = Literal["azure_openai", "local"]

ConversationField = Literal["question", "standalone_question", "sql", "answer", "tables_used"]


def _default_auxiliary_preference() -> list[ProviderName]:
    """Prefer the local model for the cheap auxiliary calls."""
    return ["local", "azure_openai"]


def _default_stored_fields() -> list[ConversationField]:
    """Store enough to resolve a follow up question, and nothing more."""
    return ["standalone_question", "sql", "tables_used"]


class LLMSettings(_Section):
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    local: LocalModelSettings = Field(default_factory=LocalModelSettings)
    default_provider: ProviderName = "azure_openai"
    auxiliary_provider_preference: list[ProviderName] = Field(
        default_factory=_default_auxiliary_preference
    )
    structured_output_repair_attempts: int = Field(default=1, ge=0, le=3)


class RoutingWeights(_Section):
    table_count: float = 0.2
    join_paths: float = 0.2
    intent: float = 0.3
    time_expressions: float = 0.1
    question_length: float = 0.1
    followup: float = 0.05
    ambiguity: float = 0.05


class CandidateSelectionWeights(_Section):
    validation: float = 0.4
    dry_run: float = 0.25
    verifier: float = 0.2
    confidence: float = 0.15


class RoutingSettings(_Section):
    enabled: bool = True
    local_max_complexity: float = Field(default=0.35, ge=0, le=1)
    always_verify_above_complexity: float = Field(default=0.7, ge=0, le=1)
    verify_below_confidence: float = Field(default=0.75, ge=0, le=1)
    local_can_verify: bool = True
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    escalate_on_failure: bool = True
    cross_generate_on_disagreement: bool = True
    dry_run_enabled: bool = True
    table_count_saturation: int = Field(default=5, ge=1)
    join_path_saturation: int = Field(default=4, ge=1)
    question_length_saturation: int = Field(default=40, ge=1)
    time_expression_saturation: int = Field(default=2, ge=1)
    intent_complexity: dict[str, float] = Field(default_factory=dict)
    weights: RoutingWeights = Field(default_factory=RoutingWeights)
    selection_weights: CandidateSelectionWeights = Field(default_factory=CandidateSelectionWeights)


class RetrievalSettings(_Section):
    top_k_tables: int = Field(default=6, ge=1)
    max_tables_in_context: int = Field(default=10, ge=1)
    max_columns_per_table: int = Field(default=40, ge=1)
    fk_expansion_hops: int = Field(default=1, ge=0, le=3)
    llm_table_selection_threshold: int = Field(default=12, ge=1)
    llm_table_selection_enabled: bool = True
    min_score: float = Field(default=0.0, ge=0)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    table_name_weight: int = Field(default=3, ge=1)
    prefix_match_min_chars: int = Field(default=5, ge=3)
    prefix_match_weight: float = Field(default=0.5, ge=0, le=1)
    stopwords: list[str] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)


class IntentSettings(_Section):
    mode: Literal["heuristic", "llm"] = "heuristic"
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    intent_priority: list[str] = Field(default_factory=list)
    default_intent: str = "lookup"
    time_patterns: list[str] = Field(default_factory=list)
    aggregation_words: list[str] = Field(default_factory=list)


class PromptSettings(_Section):
    directory: str = "prompts"
    versions: dict[str, str] = Field(default_factory=dict)
    untrusted_delimiters: list[str] = Field(default_factory=list)


class AnswerSettings(_Section):
    enabled: bool = True
    provider: Literal["primary", "azure_openai", "local"] = "primary"
    send_rows_to_llm: bool = True
    max_rows_in_prompt: int = Field(default=50, ge=0)
    decimal_as: Literal["float", "string"] = "float"


class ConversationSettings(_Section):
    enabled: bool = True
    store: Literal["sqlalchemy", "memory"] = "sqlalchemy"
    stored_fields: list[ConversationField] = Field(default_factory=_default_stored_fields)
    max_turns: int = Field(default=10, ge=1)
    context_turns: int = Field(default=3, ge=0)
    ttl_seconds: int = Field(default=3600, ge=60)
    rewrite_followups: bool = True


class ObservabilitySettings(_Section):
    service_name: str = "nl2sql-assistant"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    log_generated_sql: bool = True
    mask_sql_literals: bool = True
    sensitive_keys: list[str] = Field(default_factory=list)
    mask_patterns: list[str] = Field(default_factory=list)
    audit_enabled: bool = True
    metrics_endpoint_enabled: bool = False


class ApiKeyAuthSettings(_Section):
    header_name: str = "X-API-Key"
    keys_secret: str = "NL2SQL_API_KEYS"


class ApiSettings(_Section):
    title: str = "Natural Language to SQL Assistant"
    auth_mode: Literal["api_key", "none"] = "api_key"
    api_key: ApiKeyAuthSettings = Field(default_factory=ApiKeyAuthSettings)
    anonymous_tenant_id: str | None = None
    cors_origins: list[str] = Field(default_factory=list)
    max_request_bytes: int = Field(default=32_768, ge=1024)
    rate_limit_per_minute: int = Field(default=60, ge=0)
    expose_sql: bool = True
    docs_enabled: bool = True
    max_filters: int = Field(default=10, ge=0)


class EvaluationSettings(_Section):
    float_tolerance: float = Field(default=1e-6, ge=0)
    column_order_sensitive: bool = False
    use_llm_judge: bool = False
    reports_dir: str = "reports"


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_prefix="NL2SQL_",
        env_nested_delimiter="__",
        extra="forbid",
        case_sensitive=False,
    )

    env: str = "development"
    config_dir: str = "configs"
    app: AppInfoSettings = Field(default_factory=AppInfoSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    app_database: AppDatabaseSettings = Field(default_factory=AppDatabaseSettings)
    schema_cache: SchemaCacheSettings = Field(default_factory=SchemaCacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    cost_estimation: CostEstimationSettings = Field(default_factory=CostEstimationSettings)
    tenancy: TenancySettings = Field(default_factory=TenancySettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    intent: IntentSettings = Field(default_factory=IntentSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let environment variables override the YAML layers passed at init."""
        return (env_settings, init_settings, file_secret_settings)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.routing.local_max_complexity > self.routing.always_verify_above_complexity:
            raise ValueError(
                "routing.local_max_complexity must not exceed "
                "routing.always_verify_above_complexity"
            )
        if self.env == "production":
            if self.api.auth_mode == "none":
                raise ValueError("api.auth_mode 'none' is not permitted in production")
            if not self.security.allowed_schemas:
                raise ValueError("security.allowed_schemas must be set explicitly in production")
        if self.tenancy.enabled and not self.tenancy.tenant_column_patterns:
            raise ValueError("tenancy.enabled requires tenancy.tenant_column_patterns")
        return self

    def public_summary(self) -> dict[str, Any]:
        """Return a non sensitive view of the configuration for diagnostics."""
        return {
            "env": self.env,
            "database_dialect_configured": bool(
                self.database.url or self.database.url_secret or self.database.server
            ),
            "azure_openai_enabled": self.llm.azure_openai.enabled,
            "local_model_enabled": self.llm.local.enabled,
            "local_model_id": self.llm.local.model_id,
            "routing_enabled": self.routing.enabled,
            "conversation_enabled": self.conversation.enabled,
            "tenancy_enabled": self.tenancy.enabled,
            "max_rows": self.limits.max_rows,
        }
