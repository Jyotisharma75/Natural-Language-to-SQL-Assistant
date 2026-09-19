"""Per principal rate limiting.

A sliding window counter held in memory. It protects the database and the
model budget from one caller's runaway loop, which is the common case. It is
per replica, so behind several replicas the effective limit is the configured
number times the replica count; a hard global limit belongs in the gateway in
front of the service, not here.

A limit of zero disables it, which is the default in development.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Allows a fixed number of requests per principal per minute."""

    WINDOW_SECONDS = 60.0

    def __init__(self, per_minute: int) -> None:
        self._limit = per_minute
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    @property
    def enabled(self) -> bool:
        """Return whether limiting is in force."""
        return self._limit > 0

    def check(self, key: str) -> bool:
        """Record one request and return whether it is within the limit."""
        if not self.enabled:
            return True
        now = time.monotonic()
        cutoff = now - self.WINDOW_SECONDS
        with self._lock:
            window = self._hits.setdefault(key, deque())
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self._limit:
                return False
            window.append(now)
            return True

    def reset(self, key: str | None = None) -> None:
        """Clear one principal's window, or every window."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)
