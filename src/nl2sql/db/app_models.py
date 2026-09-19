"""Application owned tables.

These are the only tables the service writes to, and they are the only tables
in the system whose names are fixed, because they belong to the application
rather than to the customer's data model. They are created and changed through
Alembic, never at runtime.

Nothing sensitive is stored. The question is masked before it is written, the
principal and tenant are stored as short hashes so history can be grouped
without holding identities, and no credential or row value is recorded.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class AppBase(DeclarativeBase):
    """Declarative base for application owned tables."""


class QueryAuditRecord(AppBase):
    """One row per question, whether it succeeded or failed."""

    __tablename__ = "nl2sql_query_audit"

    query_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    principal_hash: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    tenant_hash: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    question_masked: Mapped[str | None] = mapped_column(Text, default=None)
    question_sha256: Mapped[str | None] = mapped_column(String(64), default=None)
    intent: Mapped[str | None] = mapped_column(String(32), default=None)
    complexity: Mapped[float | None] = mapped_column(Float, default=None)
    primary_provider: Mapped[str | None] = mapped_column(String(32), default=None)
    primary_model: Mapped[str | None] = mapped_column(String(128), default=None)
    verifier_provider: Mapped[str | None] = mapped_column(String(32), default=None)
    verifier_model: Mapped[str | None] = mapped_column(String(128), default=None)
    prompt_versions: Mapped[str | None] = mapped_column(Text, default=None)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    generated_sql: Mapped[str | None] = mapped_column(Text, default=None)
    tables_used: Mapped[str | None] = mapped_column(Text, default=None)
    validation_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_issues: Mapped[str | None] = mapped_column(Text, default=None)
    execution_status: Mapped[str] = mapped_column(String(32), default="not_executed")
    row_count: Mapped[int | None] = mapped_column(Integer, default=None)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    latency_ms: Mapped[float | None] = mapped_column(Float, default=None)
    stage_timings: Mapped[str | None] = mapped_column(Text, default=None)
    error_code: Mapped[str | None] = mapped_column(String(64), default=None, index=True)


class ConversationTurnRecord(AppBase):
    """One stored turn of a conversation, holding only the configured fields."""

    __tablename__ = "nl2sql_conversation_turns"
    __table_args__ = (
        Index(
            "ix_nl2sql_conversation_lookup",
            "conversation_id",
            "principal_hash",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64))
    principal_hash: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    payload: Mapped[str] = mapped_column(Text)
