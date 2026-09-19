"""Local Hugging Face provider.

Runs a small instruction tuned model in this process. It exists for two
reasons: simple questions can be answered without a network call, and a second
independent model can check the first one's SQL. Which of those it is doing on
any given request is decided by the router.

Practical notes that shaped this adapter:

* the model is loaded lazily and once, behind a lock, because loading is slow
  and memory is finite
* generation is synchronous and CPU bound, so it runs in a worker thread and
  is serialised by a semaphore. Letting several generations share a CPU makes
  every one of them slower without improving throughput
* a generation that outruns its timeout is abandoned by the caller, but the
  thread cannot be interrupted, so the concurrency limit is what stops a slow
  request from piling work up
* there is no native schema enforcement, so the schema is placed in the system
  message by the shared base class and the output is validated the same way
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from typing import Any

from nl2sql.config.settings import LocalModelSettings
from nl2sql.core.exceptions import LLMError, LLMTimeoutError, LLMUnavailableError
from nl2sql.core.retry import RetryPolicy
from nl2sql.llm.base import BaseLLMProvider
from nl2sql.llm.prompts import PromptTemplate
from nl2sql.llm.usage import TokenUsage, UsageTracker
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceLocalProvider(BaseLLMProvider):
    """A local causal language model served from this process."""

    supports_native_schema = False

    def __init__(
        self,
        settings: LocalModelSettings,
        *,
        usage: UsageTracker,
        format_prompt: PromptTemplate | None = None,
        repair_attempts: int = 1,
    ) -> None:
        super().__init__(
            retry=RetryPolicy(
                name="llm.local",
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
        self._lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None

    @property
    def name(self) -> str:
        """Return the configuration key of this provider."""
        return "local"

    @property
    def model_name(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._settings.model_id

    @property
    def is_loaded(self) -> bool:
        """Return whether the weights are in memory."""
        return self._model is not None

    def is_available(self) -> bool:
        """Return whether the model is enabled and its dependencies are installed."""
        if not self._settings.enabled or self._load_error is not None:
            return False
        return bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))

    # -- model loading -----------------------------------------------------
    def _load(self) -> tuple[Any, Any]:
        """Load the tokenizer and weights once."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                self._load_error = str(exc)
                raise LLMUnavailableError(
                    "The local model requires the optional dependencies. Install them with "
                    "pip install '.[local]'."
                ) from exc

            if self._settings.num_threads:
                torch.set_num_threads(self._settings.num_threads)

            common: dict[str, Any] = {
                "revision": self._settings.revision,
                "trust_remote_code": self._settings.trust_remote_code,
            }
            if self._settings.cache_dir:
                common["cache_dir"] = self._settings.cache_dir

            logger.info("local_model_loading", model_id=self._settings.model_id)
            # The loader returns one of many architecture classes, and the
            # methods used below are defined across that hierarchy rather than
            # on a single declared type, so these stay dynamic.
            tokenizer: Any
            model: Any
            try:
                tokenizer = AutoTokenizer.from_pretrained(self._settings.model_id, **common)
                dtype = self._resolve_dtype(torch)
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self._settings.model_id, dtype=dtype, **common
                    )
                except TypeError:
                    # Transformers 4.x spells the argument torch_dtype.
                    model = AutoModelForCausalLM.from_pretrained(
                        self._settings.model_id, torch_dtype=dtype, **common
                    )
            except OSError as exc:
                self._load_error = str(exc)
                raise LLMUnavailableError(
                    f"The local model {self._settings.model_id} could not be loaded. "
                    "Check the model id, the revision and the cache directory.",
                    details={"error": str(exc)[:300]},
                ) from exc

            model.to(self._settings.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            logger.info(
                "local_model_loaded",
                model_id=self._settings.model_id,
                device=self._settings.device,
                dtype=self._settings.dtype,
            )
            return tokenizer, model

    def _resolve_dtype(self, torch: Any) -> Any:
        if self._settings.dtype == "auto":
            return "auto"
        return getattr(torch, self._settings.dtype)

    async def preload(self) -> None:
        """Load the weights ahead of the first request."""
        if not self.is_available():
            return
        await asyncio.to_thread(self._load)

    # -- adapter contract --------------------------------------------------
    def _defaults(self) -> tuple[float, int]:
        return self._settings.temperature, self._settings.max_new_tokens

    async def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None,
        schema_name: str | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        if not self.is_available():
            raise LLMUnavailableError("The local model is not available.")
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._generate, messages, max_tokens, temperature),
                    timeout=self._settings.timeout_seconds,
                )
            except TimeoutError as exc:
                raise LLMTimeoutError(
                    f"The local model did not finish within "
                    f"{self._settings.timeout_seconds} seconds."
                ) from exc

    def _generate(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> tuple[str, TokenUsage]:
        """Run one generation. Executed in a worker thread."""
        import torch

        tokenizer, model = self._load()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(self._settings.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)

        new_tokens = generated[0][prompt_tokens:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return completion, TokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=int(new_tokens.shape[0])
        )

    def _translate_error(self, exc: Exception) -> LLMError:
        """Map a local runtime failure onto the domain hierarchy."""
        if isinstance(exc, TimeoutError):
            return LLMTimeoutError("The local model timed out.")
        if isinstance(exc, MemoryError):
            return LLMUnavailableError("The local model ran out of memory.")
        if isinstance(exc, OSError):
            return LLMUnavailableError(f"The local model could not be read: {exc}")
        return LLMError(f"{type(exc).__name__}: {exc}")

    async def _probe(self) -> str:
        """Confirm the weights are present without generating anything."""
        if self.is_loaded:
            return "loaded"
        return "installed"
