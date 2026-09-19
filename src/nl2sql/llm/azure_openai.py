"""Azure OpenAI provider.

Two details of Azure differ from the OpenAI service and are easy to get wrong:

* Azure addresses a **deployment name**, not a model name. The same underlying
  model is reached by different names in different environments, which is why
  it is configuration and never a literal.
* Authentication is either an API key or the Entra ID credential chain.
  Managed identity is preferred in production, because there is no key to
  rotate, leak or find in a log.

Retry is owned by the base class, so the SDK's own retry is switched off.
Otherwise the two would compound and a throttled request would be retried nine
times instead of three.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nl2sql.config.secrets import resolve_secret
from nl2sql.config.settings import AzureOpenAISettings
from nl2sql.core.exceptions import (
    ConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)
from nl2sql.core.retry import RetryPolicy
from nl2sql.llm.base import BaseLLMProvider
from nl2sql.llm.prompts import PromptTemplate
from nl2sql.llm.usage import TokenUsage, UsageTracker
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)

#: Scope requested when authenticating to Azure OpenAI with a token credential.
_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureOpenAIProvider(BaseLLMProvider):
    """Language model access through an Azure OpenAI deployment."""

    def __init__(
        self,
        settings: AzureOpenAISettings,
        *,
        usage: UsageTracker,
        format_prompt: PromptTemplate | None = None,
        repair_attempts: int = 1,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            retry=RetryPolicy(
                name="llm.azure_openai",
                max_attempts=settings.retry.max_attempts,
                initial_backoff_seconds=settings.retry.initial_backoff_seconds,
                max_backoff_seconds=settings.retry.max_backoff_seconds,
                jitter_seconds=settings.retry.jitter_seconds,
            ),
            usage=usage,
            format_prompt=format_prompt,
            repair_attempts=repair_attempts,
        )
        self._settings = settings
        self._client_factory = client_factory
        self._client: Any | None = None
        self.supports_native_schema = settings.structured_output_mode == "json_schema"

    @property
    def name(self) -> str:
        """Return the configuration key of this provider."""
        return "azure_openai"

    @property
    def model_name(self) -> str:
        """Return the deployment name, which Azure uses in place of a model name."""
        return self._settings.deployment or "unconfigured"

    def is_available(self) -> bool:
        """Return whether a deployment and a credential are configured."""
        if not self._settings.enabled:
            return False
        if self._client_factory is not None:
            return bool(self._settings.deployment)
        if not self._settings.endpoint or not self._settings.deployment:
            return False
        return self._settings.use_managed_identity or bool(
            resolve_secret(self._settings.api_key_secret)
        )

    # -- client ------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client

        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:  # pragma: no cover - openai is a hard dependency
            raise LLMError("The openai package is not installed.") from exc

        if not self._settings.endpoint or not self._settings.deployment:
            raise ConfigurationError(
                "Azure OpenAI needs an endpoint and a deployment. Set "
                "NL2SQL_LLM__AZURE_OPENAI__ENDPOINT and NL2SQL_LLM__AZURE_OPENAI__DEPLOYMENT."
            )

        kwargs: dict[str, Any] = {
            "azure_endpoint": self._settings.endpoint,
            "api_version": self._settings.api_version,
            "timeout": self._settings.timeout_seconds,
            # Retry belongs to the base class.
            "max_retries": 0,
        }
        if self._settings.use_managed_identity:
            kwargs["azure_ad_token_provider"] = self._build_token_provider()
        else:
            api_key = resolve_secret(self._settings.api_key_secret)
            if not api_key:
                raise ConfigurationError(
                    f"No Azure OpenAI credential is available. Set the "
                    f"{self._settings.api_key_secret} environment variable, or enable "
                    "llm.azure_openai.use_managed_identity."
                )
            kwargs["api_key"] = api_key

        self._client = AsyncAzureOpenAI(**kwargs)
        logger.info(
            "azure_openai_client_initialised",
            endpoint=self._settings.endpoint,
            deployment=self._settings.deployment,
            api_version=self._settings.api_version,
            managed_identity=self._settings.use_managed_identity,
        )
        return self._client

    def _build_token_provider(self) -> Any:
        """Return a bearer token provider backed by the Entra ID credential chain."""
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:  # pragma: no cover - azure-identity is a hard dependency
            raise ConfigurationError(
                "Managed identity for Azure OpenAI requires the azure-identity package."
            ) from exc

        credential = DefaultAzureCredential(
            managed_identity_client_id=self._settings.managed_identity_client_id
        )
        return get_bearer_token_provider(credential, _COGNITIVE_SERVICES_SCOPE)

    # -- adapter contract --------------------------------------------------
    def _defaults(self) -> tuple[float, int]:
        return self._settings.temperature, self._settings.max_output_tokens

    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None,
        schema_name: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._settings.deployment,
            "messages": messages,
            self._settings.token_limit_parameter: max_tokens,
        }
        if self._settings.send_temperature:
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        elif self._settings.structured_output_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise LLMError(
                "The request was blocked by the Azure OpenAI content filter.",
                public_message="The question could not be processed by the language model.",
            )
        if finish_reason == "length":
            logger.warning(
                "azure_openai_output_truncated",
                deployment=self._settings.deployment,
                max_tokens=max_tokens,
            )
        text = choice.message.content or ""
        raw_usage = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
        )
        return text, usage

    def _translate_error(self, exc: Exception) -> LLMError:
        """Map an SDK exception onto the domain hierarchy, deciding retryability."""
        try:
            import openai
        except ImportError:  # pragma: no cover - openai is a hard dependency
            return LLMError(str(exc))

        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError("Azure OpenAI throttled the request.")
        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError("Azure OpenAI did not respond in time.")
        if isinstance(exc, openai.APIConnectionError | openai.InternalServerError):
            return LLMServiceError(f"Azure OpenAI is unavailable: {type(exc).__name__}")
        if isinstance(exc, openai.AuthenticationError):
            return LLMError(
                "Azure OpenAI rejected the credential.",
                public_message="The language model is not configured correctly.",
            )
        if isinstance(exc, openai.BadRequestError):
            return LLMError(f"Azure OpenAI rejected the request: {exc}")
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status >= 500:
            return LLMServiceError(f"Azure OpenAI returned status {status}.")
        return LLMError(f"{type(exc).__name__}: {exc}")

    async def _probe(self) -> str:
        """Confirm the deployment answers, when probing is enabled."""
        if not self._settings.health_probe:
            return "configured"
        client = self._get_client()
        await client.chat.completions.create(
            model=self._settings.deployment,
            messages=[{"role": "user", "content": "ping"}],
            **{self._settings.token_limit_parameter: 1},
        )
        return "reachable"
