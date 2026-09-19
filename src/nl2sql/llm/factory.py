"""Provider construction and lookup.

The registry is what lets the router treat models as interchangeable. It knows
which providers exist, which are actually usable right now, and how to pick one
by preference order, so a deployment with no local model or with Azure OpenAI
unreachable degrades to what it has rather than failing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from nl2sql.config.settings import Settings
from nl2sql.core.exceptions import LLMUnavailableError
from nl2sql.llm.azure_openai import AzureOpenAIProvider
from nl2sql.llm.base import LLMProvider, ProviderHealth
from nl2sql.llm.huggingface_local import HuggingFaceLocalProvider
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.llm.usage import UsageTracker

AZURE = "azure_openai"
LOCAL = "local"


class ProviderRegistry:
    """Holds the configured providers and answers which can be used."""

    def __init__(self, providers: Mapping[str, LLMProvider]) -> None:
        self._providers = dict(providers)

    def __contains__(self, name: object) -> bool:
        return name in self._providers

    def all(self) -> tuple[LLMProvider, ...]:
        """Return every configured provider, usable or not."""
        return tuple(self._providers.values())

    def get(self, name: str) -> LLMProvider:
        """Return one provider by name, whether or not it is usable."""
        provider = self._providers.get(name)
        if provider is None:
            raise LLMUnavailableError(
                f"No language model provider named {name} is configured.",
                details={"configured": sorted(self._providers)},
            )
        return provider

    def try_get(self, name: str) -> LLMProvider | None:
        """Return one provider only if it is usable right now."""
        provider = self._providers.get(name)
        if provider is None or not provider.is_available():
            return None
        return provider

    def available_names(self) -> tuple[str, ...]:
        """Return the names of every usable provider."""
        return tuple(name for name, p in self._providers.items() if p.is_available())

    def first_available(self, preference: Sequence[str]) -> LLMProvider | None:
        """Return the first usable provider in preference order."""
        for name in preference:
            provider = self.try_get(name)
            if provider is not None:
                return provider
        return None

    def require_any(self) -> LLMProvider:
        """Return any usable provider, or explain that none is."""
        for provider in self._providers.values():
            if provider.is_available():
                return provider
        raise LLMUnavailableError(
            "No language model provider is available. Configure Azure OpenAI or enable "
            "the local model.",
            details={"configured": sorted(self._providers)},
        )

    async def health(self) -> tuple[ProviderHealth, ...]:
        """Report the health of every configured provider concurrently."""
        results = await asyncio.gather(
            *(provider.health() for provider in self._providers.values()),
            return_exceptions=True,
        )
        healths: list[ProviderHealth] = []
        for provider, result in zip(self._providers.values(), results, strict=True):
            if isinstance(result, ProviderHealth):
                healths.append(result)
            else:
                healths.append(
                    ProviderHealth(
                        name=provider.name,
                        model=provider.model_name,
                        available=False,
                        detail=type(result).__name__,
                    )
                )
        return tuple(healths)


def build_provider_registry(
    settings: Settings,
    *,
    usage: UsageTracker,
    prompts: PromptRegistry,
    azure_client_factory: Any | None = None,
) -> ProviderRegistry:
    """Build every provider the configuration declares."""
    format_prompt = prompts.get("structured_output")
    repair_attempts = settings.llm.structured_output_repair_attempts

    providers: dict[str, LLMProvider] = {
        AZURE: AzureOpenAIProvider(
            settings.llm.azure_openai,
            usage=usage,
            format_prompt=format_prompt,
            repair_attempts=repair_attempts,
            client_factory=azure_client_factory,
        ),
        LOCAL: HuggingFaceLocalProvider(
            settings.llm.local,
            usage=usage,
            format_prompt=format_prompt,
            repair_attempts=repair_attempts,
        ),
    }
    return ProviderRegistry(providers)
