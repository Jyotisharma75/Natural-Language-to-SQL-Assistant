"""Persistence for the application owned tables.

Both repositories are synchronous, because the drivers are, and are called
from async code through ``asyncio.to_thread``. Failures to write history or an
audit row are logged and swallowed by the caller rather than failing the
user's question: losing an audit row is a problem, but returning an error for
a query that actually succeeded is a worse one, and the same event is also on
the structured log.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from nl2sql.db.app_models import AppBase, ConversationTurnRecord, QueryAuditRecord


def create_all(engine: Engine) -> None:
    """Create the application tables. Alembic owns this in deployed environments."""
    AppBase.metadata.create_all(engine)


class AuditRepository:
    """Stores one row per question."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, values: dict[str, Any]) -> None:
        """Insert an audit row."""
        with Session(self._engine) as session, session.begin():
            session.add(QueryAuditRecord(**values))

    def get(self, query_id: str) -> QueryAuditRecord | None:
        """Return one audit row, used by tests and support tooling."""
        with Session(self._engine) as session:
            return session.get(QueryAuditRecord, query_id)

    def count(self) -> int:
        """Return how many audit rows exist."""
        with Session(self._engine) as session:
            return len(session.execute(select(QueryAuditRecord.query_id)).all())


class ConversationRepository:
    """Stores and prunes conversation turns."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(
        self,
        *,
        conversation_id: str,
        principal_hash: str,
        payload: str,
        max_turns: int,
    ) -> None:
        """Add a turn and drop the oldest ones beyond ``max_turns``."""
        with Session(self._engine) as session, session.begin():
            session.add(
                ConversationTurnRecord(
                    conversation_id=conversation_id,
                    principal_hash=principal_hash,
                    payload=payload,
                )
            )
            session.flush()
            keep_ids = session.scalars(
                select(ConversationTurnRecord.id)
                .where(
                    ConversationTurnRecord.conversation_id == conversation_id,
                    ConversationTurnRecord.principal_hash == principal_hash,
                )
                .order_by(ConversationTurnRecord.id.desc())
                .limit(max_turns)
            ).all()
            if keep_ids:
                session.execute(
                    delete(ConversationTurnRecord).where(
                        ConversationTurnRecord.conversation_id == conversation_id,
                        ConversationTurnRecord.principal_hash == principal_hash,
                        ConversationTurnRecord.id.notin_(keep_ids),
                    )
                )

    def recent(
        self,
        *,
        conversation_id: str,
        principal_hash: str,
        limit: int,
        ttl_seconds: int,
    ) -> Sequence[tuple[datetime, str]]:
        """Return the most recent turns, oldest first, within the retention window.

        The principal is part of the lookup, so one caller cannot read another
        caller's history by guessing a conversation identifier.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
        with Session(self._engine) as session:
            rows = session.execute(
                select(ConversationTurnRecord.created_at, ConversationTurnRecord.payload)
                .where(
                    ConversationTurnRecord.conversation_id == conversation_id,
                    ConversationTurnRecord.principal_hash == principal_hash,
                    ConversationTurnRecord.created_at >= cutoff,
                )
                .order_by(ConversationTurnRecord.id.desc())
                .limit(limit)
            ).all()
        return [(created_at, payload) for created_at, payload in reversed(rows)]

    def purge(self, *, conversation_id: str, principal_hash: str) -> None:
        """Delete every turn of one conversation."""
        with Session(self._engine) as session, session.begin():
            session.execute(
                delete(ConversationTurnRecord).where(
                    ConversationTurnRecord.conversation_id == conversation_id,
                    ConversationTurnRecord.principal_hash == principal_hash,
                )
            )
