"""Application owned audit and conversation tables.

Revision ID: 0001
Revises:
Create Date: 2026-02-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "nl2sql_query_audit",
        sa.Column("query_id", sa.String(length=32), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("principal_hash", sa.String(length=32), nullable=True),
        sa.Column("tenant_hash", sa.String(length=32), nullable=True),
        sa.Column("question_masked", sa.Text(), nullable=True),
        sa.Column("question_sha256", sa.String(length=64), nullable=True),
        sa.Column("intent", sa.String(length=32), nullable=True),
        sa.Column("complexity", sa.Float(), nullable=True),
        sa.Column("primary_provider", sa.String(length=32), nullable=True),
        sa.Column("primary_model", sa.String(length=128), nullable=True),
        sa.Column("verifier_provider", sa.String(length=32), nullable=True),
        sa.Column("verifier_model", sa.String(length=128), nullable=True),
        sa.Column("prompt_versions", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("generated_sql", sa.Text(), nullable=True),
        sa.Column("tables_used", sa.Text(), nullable=True),
        sa.Column("validation_passed", sa.Boolean(), nullable=False),
        sa.Column("validation_issues", sa.Text(), nullable=True),
        sa.Column("execution_status", sa.String(length=32), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("stage_timings", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("query_id"),
    )
    op.create_index(
        "ix_nl2sql_query_audit_created_at", "nl2sql_query_audit", ["created_at"], unique=False
    )
    op.create_index(
        "ix_nl2sql_query_audit_principal_hash",
        "nl2sql_query_audit",
        ["principal_hash"],
        unique=False,
    )
    op.create_index(
        "ix_nl2sql_query_audit_tenant_hash", "nl2sql_query_audit", ["tenant_hash"], unique=False
    )
    op.create_index(
        "ix_nl2sql_query_audit_error_code", "nl2sql_query_audit", ["error_code"], unique=False
    )

    op.create_table(
        "nl2sql_conversation_turns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("principal_hash", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_nl2sql_conversation_lookup",
        "nl2sql_conversation_turns",
        ["conversation_id", "principal_hash", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_nl2sql_conversation_lookup", table_name="nl2sql_conversation_turns")
    op.drop_table("nl2sql_conversation_turns")
    op.drop_index("ix_nl2sql_query_audit_error_code", table_name="nl2sql_query_audit")
    op.drop_index("ix_nl2sql_query_audit_tenant_hash", table_name="nl2sql_query_audit")
    op.drop_index("ix_nl2sql_query_audit_principal_hash", table_name="nl2sql_query_audit")
    op.drop_index("ix_nl2sql_query_audit_created_at", table_name="nl2sql_query_audit")
    op.drop_table("nl2sql_query_audit")
