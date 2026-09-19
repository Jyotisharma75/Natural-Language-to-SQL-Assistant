"""Conversation memory.

Optional, bounded and minimal by design. Three rules hold whatever the store:

* only the fields named in ``conversation.stored_fields`` are kept. A
  deployment that wants the SQL but not the question sets exactly that
* every stored string passes through the masker first, so a credential that
  found its way into a question is not persisted
* history is keyed by the conversation and by a hash of the principal, so one
  caller cannot read another caller's turns by guessing an identifier

When the feature is disabled, a store that does nothing is used, and requests
that carry a conversation identifier are answered without history rather than
refused.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from nl2sql.config.settings import ConversationSettings
from nl2sql.core.masking import Masker, hash_identifier
from nl2sql.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One stored turn."""

    created_at: datetime
    fields: dict[str, str]

    def render(self) -> str:
        """Render the turn as the line a prompt shows."""
        parts = [f"{key}: {value}" for key, value in self.fields.items() if value]
        return " | ".join(parts)


class ConversationStore(ABC):
    """Where turns are kept."""

    @abstractmethod
    async def append(self, conversation_id: str, principal_hash: str, payload: str) -> None:
        """Store one turn."""

    @abstractmethod
    async def recent(
        self, conversation_id: str, principal_hash: str, limit: int
    ) -> Sequence[tuple[datetime, str]]:
        """Return recent turns, oldest first."""

    @abstractmethod
    async def clear(self, conversation_id: str, principal_hash: str) -> None:
        """Delete a conversation."""


class ConversationService:
    """Applies the field policy and masking around whichever store is configured."""

    def __init__(
        self,
        settings: ConversationSettings,
        store: ConversationStore | None,
        masker: Masker,
    ) -> None:
        self._settings = settings
        self._store = store
        self._masker = masker

    @property
    def enabled(self) -> bool:
        """Return whether history is being kept."""
        return self._settings.enabled and self._store is not None

    async def history(self, conversation_id: str | None, principal_id: str) -> list[str]:
        """Return the recent turns rendered for a prompt."""
        if not self.enabled or not conversation_id or self._store is None:
            return []
        principal_hash = hash_identifier(principal_id) or "anonymous"
        rows = await self._store.recent(
            conversation_id, principal_hash, self._settings.context_turns
        )
        turns: list[str] = []
        for created_at, payload in rows:
            try:
                fields = json.loads(payload)
            except json.JSONDecodeError:
                continue
            turn = ConversationTurn(created_at=created_at, fields=fields)
            rendered = turn.render()
            if rendered:
                turns.append(rendered)
        return turns

    async def record(
        self,
        conversation_id: str | None,
        principal_id: str,
        values: Mapping[str, object],
    ) -> None:
        """Store the configured fields of one turn."""
        if not self.enabled or not conversation_id or self._store is None:
            return
        allowed = {
            field: self._mask(values.get(field))
            for field in self._settings.stored_fields
            if values.get(field) is not None
        }
        if not allowed:
            return
        principal_hash = hash_identifier(principal_id) or "anonymous"
        try:
            await self._store.append(
                conversation_id, principal_hash, json.dumps(allowed, default=str)
            )
        except Exception as exc:
            logger.warning("conversation_write_failed", error_type=type(exc).__name__)

    def _mask(self, value: object) -> str:
        if isinstance(value, list | tuple):
            value = ", ".join(str(item) for item in value)
        return self._masker.mask_text(str(value))

    async def clear(self, conversation_id: str, principal_id: str) -> None:
        """Delete a conversation's history."""
        if not self.enabled or self._store is None:
            return
        principal_hash = hash_identifier(principal_id) or "anonymous"
        await self._store.clear(conversation_id, principal_hash)


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)
