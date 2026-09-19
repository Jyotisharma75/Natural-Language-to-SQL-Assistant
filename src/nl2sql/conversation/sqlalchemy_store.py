"""Database backed conversation store.

Writes to the application owned tables, so history survives a restart and is
shared between replicas. The driver is synchronous, so each call runs in a
worker thread rather than blocking the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime

from nl2sql.conversation.store import ConversationStore
from nl2sql.db.repositories import ConversationRepository


class SQLAlchemyConversationStore(ConversationStore):
    """Keeps conversation turns in the application database."""

    def __init__(
        self,
        repository: ConversationRepository,
        *,
        max_turns: int = 10,
        ttl_seconds: int = 3600,
    ) -> None:
        self._repository = repository
        self._max_turns = max_turns
        self._ttl = ttl_seconds

    async def append(self, conversation_id: str, principal_hash: str, payload: str) -> None:
        """Store one turn and prune the conversation to its limit."""
        await asyncio.to_thread(
            self._repository.append,
            conversation_id=conversation_id,
            principal_hash=principal_hash,
            payload=payload,
            max_turns=self._max_turns,
        )

    async def recent(
        self, conversation_id: str, principal_hash: str, limit: int
    ) -> Sequence[tuple[datetime, str]]:
        """Return recent turns, oldest first."""
        if limit <= 0:
            return []
        return await asyncio.to_thread(
            self._repository.recent,
            conversation_id=conversation_id,
            principal_hash=principal_hash,
            limit=limit,
            ttl_seconds=self._ttl,
        )

    async def clear(self, conversation_id: str, principal_hash: str) -> None:
        """Delete one conversation."""
        await asyncio.to_thread(
            self._repository.purge,
            conversation_id=conversation_id,
            principal_hash=principal_hash,
        )
