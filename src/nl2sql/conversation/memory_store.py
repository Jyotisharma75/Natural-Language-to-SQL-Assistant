"""In process conversation store.

Used in development and in tests, and anywhere a single replica is acceptable.
History lives in memory, so it is lost on restart and is not shared between
replicas. The SQLAlchemy store is the one to use when either of those matters.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from nl2sql.conversation.store import ConversationStore


class InMemoryConversationStore(ConversationStore):
    """Keeps a bounded, expiring history per conversation and principal."""

    def __init__(self, *, max_turns: int = 10, ttl_seconds: int = 3600) -> None:
        self._max_turns = max_turns
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._turns: dict[tuple[str, str], list[tuple[datetime, str]]] = {}

    async def append(self, conversation_id: str, principal_hash: str, payload: str) -> None:
        """Store one turn, dropping the oldest beyond the limit."""
        key = (conversation_id, principal_hash)
        with self._lock:
            history = self._turns.setdefault(key, [])
            history.append((datetime.now(UTC), payload))
            if len(history) > self._max_turns:
                del history[: len(history) - self._max_turns]

    async def recent(
        self, conversation_id: str, principal_hash: str, limit: int
    ) -> Sequence[tuple[datetime, str]]:
        """Return the most recent turns within the retention window, oldest first."""
        key = (conversation_id, principal_hash)
        cutoff = datetime.now(UTC) - timedelta(seconds=self._ttl)
        with self._lock:
            history = [item for item in self._turns.get(key, []) if item[0] >= cutoff]
            self._turns[key] = history
            return list(history[-limit:]) if limit > 0 else []

    async def clear(self, conversation_id: str, principal_hash: str) -> None:
        """Delete one conversation."""
        with self._lock:
            self._turns.pop((conversation_id, principal_hash), None)
