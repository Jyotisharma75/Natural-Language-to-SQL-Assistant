"""Time to live cache with single flight refresh.

Schema introspection is expensive enough that a burst of questions must not
each trigger one. The first caller to find the entry stale performs the
refresh while the others wait on the same lock and then read the fresh value.

When a refresh fails and a stale value exists, the stale value is served and
the failure is logged. A schema that was accurate five minutes ago is a better
answer than an outage.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

from nl2sql.observability.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


class TTLCache(Generic[T]):
    """A single value cache that reloads after ``ttl_seconds``."""

    def __init__(
        self,
        loader: Callable[[], T],
        *,
        ttl_seconds: float,
        serve_stale_on_error: bool = True,
        clock: Callable[[], float] = time.monotonic,
        name: str = "cache",
    ) -> None:
        self._loader = loader
        self._ttl = ttl_seconds
        self._serve_stale = serve_stale_on_error
        self._clock = clock
        self._name = name
        self._lock = threading.Lock()
        self._value: T | None = None
        self._loaded_at: float | None = None

    @property
    def is_warm(self) -> bool:
        """Return whether a value has been loaded at least once."""
        return self._value is not None

    @property
    def age_seconds(self) -> float | None:
        """Return how long ago the current value was loaded."""
        if self._loaded_at is None:
            return None
        return self._clock() - self._loaded_at

    def _is_fresh(self) -> bool:
        if self._value is None or self._loaded_at is None:
            return False
        if self._ttl <= 0:
            return False
        return (self._clock() - self._loaded_at) < self._ttl

    def get(self) -> T:
        """Return the cached value, reloading it when stale."""
        if self._is_fresh():
            return self._cached()
        with self._lock:
            if self._is_fresh():
                return self._cached()
            try:
                value = self._loader()
            except Exception as exc:
                if self._serve_stale and self._value is not None:
                    logger.warning(
                        "cache_refresh_failed_serving_stale",
                        cache=self._name,
                        error_type=type(exc).__name__,
                        age_seconds=self.age_seconds,
                    )
                    return self._cached()
                raise
            self._value = value
            self._loaded_at = self._clock()
            return value

    def _cached(self) -> T:
        value = self._value
        if value is None:  # pragma: no cover - guarded by _is_fresh
            raise RuntimeError("cache value requested before it was loaded")
        return value

    def invalidate(self) -> None:
        """Drop the cached value so the next read reloads it."""
        with self._lock:
            self._value = None
            self._loaded_at = None
