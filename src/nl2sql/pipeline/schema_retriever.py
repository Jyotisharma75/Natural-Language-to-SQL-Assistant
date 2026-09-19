"""Schema retrieval.

A model must be given the tables a question needs and not the whole database.
Sending everything is expensive, pushes the relevant tables into the noise, and
on a large schema does not fit at all.

Retrieval happens in four steps:

1. the question and every table are reduced to word tokens. Identifiers are
   split on case changes and underscores, so ``EnergyConsumption`` and
   ``energy_consumption`` both yield ``energy`` and ``consumption``
2. tables are scored with BM25 over those tokens, with the table's own name
   weighted above its column names and descriptions
3. the best tables are expanded along foreign keys, because a question about
   a measure almost always needs the table holding the name of the thing
   measured, and that table may share no word with the question
4. when too many candidates remain, a model narrows the list, and any name it
   returns that was not a candidate is discarded

The vocabulary that maps question words onto identifier words is configuration,
so a deployment teaches the retriever its own domain language without a code
change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from nl2sql.config.settings import RetrievalSettings
from nl2sql.core.exceptions import LLMError, NoRelevantSchemaError
from nl2sql.llm.base import LLMProvider
from nl2sql.llm.prompts import PromptRegistry
from nl2sql.metadata.models import ColumnInfo, SchemaCatalog, TableInfo
from nl2sql.observability.logging import get_logger
from nl2sql.pipeline.models import IntentAnalysis, RetrievedSchema, TableSelection

logger = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class SchemaRetriever:
    """Chooses the tables that go into the generation prompt."""

    def __init__(self, settings: RetrievalSettings, prompts: PromptRegistry) -> None:
        self._settings = settings
        self._prompts = prompts
        self._stopwords = frozenset(word.casefold() for word in settings.stopwords)
        self._synonyms = {
            key.casefold(): tuple(values) for key, values in settings.synonyms.items()
        }

    # -- tokenisation ------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        """Split text or an identifier into comparable word tokens."""
        if not text:
            return []
        spaced = _CASE_BOUNDARY.sub(" ", text.replace("_", " ").replace(".", " "))
        tokens: list[str] = []
        for raw in _WORD.findall(spaced):
            word = raw.casefold()
            if word in self._stopwords or len(word) < 2:
                continue
            tokens.append(self._stem(word))
        return tokens

    @staticmethod
    def _stem(word: str) -> str:
        """Fold the plural forms that matter for matching identifiers."""
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 4 and word.endswith("sses"):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def _expand(self, tokens: Iterable[str]) -> list[str]:
        """Add the configured synonyms of each question token."""
        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            for synonym in self._synonyms.get(token, ()):
                expanded.extend(self.tokenize(synonym))
        return expanded

    # -- scoring -----------------------------------------------------------
    def _document(self, table: TableInfo) -> list[str]:
        """Build the token document that represents one table."""
        tokens: list[str] = []
        name_tokens = self.tokenize(table.name)
        tokens.extend(name_tokens * self._settings.table_name_weight)
        tokens.extend(self.tokenize(table.schema_name))
        if table.comment:
            tokens.extend(self.tokenize(table.comment))
        for column in table.columns:
            tokens.extend(self.tokenize(column.name))
            if column.comment:
                tokens.extend(self.tokenize(column.comment))
        return tokens

    def score_tables(self, query_tokens: list[str], catalog: SchemaCatalog) -> dict[str, float]:
        """Return a BM25 score per table for the question tokens."""
        documents = {table.key: self._document(table) for table in catalog.tables}
        if not documents:
            return {}
        counters = {key: Counter(tokens) for key, tokens in documents.items()}
        lengths = {key: max(len(tokens), 1) for key, tokens in documents.items()}
        average_length = sum(lengths.values()) / len(lengths)
        total_documents = len(documents)

        k1 = self._settings.bm25_k1
        b = self._settings.bm25_b
        minimum_prefix = self._settings.prefix_match_min_chars
        prefix_weight = self._settings.prefix_match_weight

        document_frequency: dict[str, int] = {}
        for token in set(query_tokens):
            document_frequency[token] = sum(1 for counter in counters.values() if counter[token])

        scores: dict[str, float] = {}
        for key, counter in counters.items():
            score = 0.0
            for token in query_tokens:
                frequency = float(counter[token])
                if frequency == 0 and len(token) >= minimum_prefix:
                    # A partial credit pass, so consumption matches consumed.
                    frequency = prefix_weight * sum(
                        count
                        for word, count in counter.items()
                        if len(word) >= minimum_prefix
                        and (
                            word.startswith(token[:minimum_prefix])
                            or token.startswith(word[:minimum_prefix])
                        )
                    )
                if frequency == 0:
                    continue
                matched_in = document_frequency.get(token, 0) or 1
                idf = math.log(1 + (total_documents - matched_in + 0.5) / (matched_in + 0.5))
                normalisation = k1 * (1 - b + b * lengths[key] / average_length)
                score += idf * (frequency * (k1 + 1)) / (frequency + normalisation)
            if score > 0:
                scores[key] = round(score, 6)
        return scores

    # -- retrieval ---------------------------------------------------------
    async def retrieve(
        self,
        question: str,
        intent: IntentAnalysis,
        catalog: SchemaCatalog,
        *,
        provider: LLMProvider | None = None,
    ) -> RetrievedSchema:
        """Return the tables and the schema context for one question."""
        if not catalog.tables:
            raise NoRelevantSchemaError(
                "No tables are visible to this service.",
                public_message=(
                    "No tables are available to answer questions. Check the schema "
                    "allowlist and the database permissions."
                ),
            )

        tokens = self._expand(
            self.tokenize(question)
            + [token for entity in intent.entities for token in self.tokenize(entity)]
            + [token for metric in intent.metrics for token in self.tokenize(metric)]
        )
        scores = self.score_tables(tokens, catalog)

        by_key = {table.key: table for table in catalog.tables}
        ranked = [
            by_key[key]
            for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score > self._settings.min_score
        ]

        if not ranked:
            if len(catalog.tables) <= self._settings.top_k_tables:
                # A small schema is cheap to describe in full, and a question
                # that shares no word with any identifier is exactly when that
                # helps most.
                ranked = list(catalog.tables)
            else:
                raise NoRelevantSchemaError()

        candidate_count = len(ranked)
        selected = ranked[: self._settings.top_k_tables]
        llm_selected = False

        if (
            provider is not None
            and self._settings.llm_table_selection_enabled
            and candidate_count > self._settings.llm_table_selection_threshold
            and provider.is_available()
        ):
            narrowed = await self._select_with_model(question, ranked, provider, catalog)
            if narrowed:
                selected = narrowed
                llm_selected = True
                logger.info("table_selection_applied", selected=len(selected))

        expanded = self._expand_relationships(selected, catalog)
        join_edges = self._count_edges(expanded)
        context = self.render_context(expanded, tokens, catalog)

        return RetrievedSchema(
            tables=tuple(expanded),
            scores={by_key[key].qualified_name: value for key, value in scores.items()},
            context=context,
            candidate_count=candidate_count,
            join_edges=join_edges,
            llm_selected=llm_selected,
        )

    async def _select_with_model(
        self,
        question: str,
        candidates: list[TableInfo],
        provider: LLMProvider,
        catalog: SchemaCatalog,
    ) -> list[TableInfo]:
        """Ask a model to narrow a long candidate list, ignoring anything invented."""
        listing = "\n".join(
            f"- {table.qualified_name}: {table.comment or 'no description'} "
            f"(columns: {', '.join(table.column_names[:12])})"
            for table in candidates
        )
        try:
            prompt = self._prompts.render("table_selection", question=question, candidates=listing)
            result = await provider.generate_structured(prompt, TableSelection)
        except LLMError as exc:
            logger.warning("table_selection_failed", error_type=type(exc).__name__)
            return []

        allowed = {table.key: table for table in candidates}
        chosen: list[TableInfo] = []
        for name in result.output.tables:
            resolved = catalog.resolve(name, default_schema=catalog.default_schema)
            if resolved is not None and resolved.key in allowed:
                chosen.append(resolved)
        return chosen[: self._settings.max_tables_in_context]

    def _expand_relationships(
        self, selected: list[TableInfo], catalog: SchemaCatalog
    ) -> list[TableInfo]:
        """Add tables reachable by foreign key, so joins have somewhere to land."""
        chosen = {table.key: table for table in selected}
        budget = self._settings.max_tables_in_context

        for _ in range(self._settings.fk_expansion_hops):
            if len(chosen) >= budget:
                break
            additions: dict[str, TableInfo] = {}
            for table in list(chosen.values()):
                # Outgoing keys first: they point at the table that turns an
                # identifier into a name a person recognises.
                for fk in table.foreign_keys:
                    referenced = catalog.get(fk.referred_schema, fk.referred_table)
                    if referenced is not None and referenced.key not in chosen:
                        additions[referenced.key] = referenced
                for neighbour in catalog.neighbours(table):
                    if neighbour.key not in chosen and neighbour.key not in additions:
                        links = sum(
                            1
                            for fk in neighbour.foreign_keys
                            if fk.referred_qualified_name.casefold() in chosen
                        )
                        if links >= 2:
                            # A bridge between two tables already chosen.
                            additions[neighbour.key] = neighbour
            if not additions:
                break
            for key, table in additions.items():
                if len(chosen) >= budget:
                    break
                chosen[key] = table

        return list(chosen.values())[:budget]

    @staticmethod
    def _count_edges(tables: list[TableInfo]) -> int:
        """Count foreign key edges between the chosen tables."""
        keys = {table.key for table in tables}
        return sum(
            1
            for table in tables
            for fk in table.foreign_keys
            if fk.referred_qualified_name.casefold() in keys
        )

    # -- rendering ---------------------------------------------------------
    def render_context(
        self, tables: list[TableInfo], query_tokens: list[str], catalog: SchemaCatalog
    ) -> str:
        """Render the schema block that goes into the prompt."""
        wanted = set(query_tokens)
        blocks: list[str] = []
        for table in sorted(tables, key=lambda t: t.qualified_name):
            header = f"TABLE {table.qualified_name}"
            if table.kind == "view":
                header += " (view)"
            if table.comment:
                header += f"  -- {table.comment}"
            lines = [header]
            for column in self._ordered_columns(table, wanted):
                parts = [f"  {column.name} {column.data_type}"]
                if column.is_primary_key:
                    parts.append("PRIMARY KEY")
                elif not column.nullable:
                    parts.append("NOT NULL")
                reference = self._reference_for(table, column.name)
                if reference:
                    parts.append(f"references {reference}")
                line = " ".join(parts)
                if column.comment:
                    line += f"  -- {column.comment}"
                lines.append(line)
            hidden = len(table.columns) - self._settings.max_columns_per_table
            if hidden > 0:
                lines.append(f"  ... {hidden} further columns are not listed")
            blocks.append("\n".join(lines))

        relationships = self._render_relationships(tables, catalog)
        if relationships:
            blocks.append("RELATIONSHIPS\n" + "\n".join(relationships))
        return "\n\n".join(blocks)

    def _ordered_columns(self, table: TableInfo, wanted: set[str]) -> list[ColumnInfo]:
        """Order columns so the useful ones survive the per table cap."""
        foreign_key_columns = {
            column.casefold() for fk in table.foreign_keys for column in fk.columns
        }

        def rank(column: ColumnInfo) -> tuple[int, int]:
            folded = column.name.casefold()
            if column.is_primary_key:
                priority = 0
            elif folded in foreign_key_columns:
                priority = 1
            elif set(self.tokenize(column.name)) & wanted:
                priority = 2
            else:
                priority = 3
            return priority, table.column_names.index(column.name)

        ordered = sorted(table.columns, key=rank)
        return ordered[: self._settings.max_columns_per_table]

    @staticmethod
    def _reference_for(table: TableInfo, column_name: str) -> str | None:
        folded = column_name.casefold()
        for fk in table.foreign_keys:
            for index, column in enumerate(fk.columns):
                if column.casefold() == folded:
                    target = (
                        fk.referred_columns[index]
                        if index < len(fk.referred_columns)
                        else fk.referred_columns[0]
                        if fk.referred_columns
                        else ""
                    )
                    return f"{fk.referred_qualified_name}.{target}".rstrip(".")
        return None

    @staticmethod
    def _render_relationships(tables: list[TableInfo], catalog: SchemaCatalog) -> list[str]:
        """List the join paths available between the chosen tables."""
        keys = {table.key for table in tables}
        lines: list[str] = []
        for table in sorted(tables, key=lambda t: t.qualified_name):
            for fk in table.foreign_keys:
                if fk.referred_qualified_name.casefold() not in keys:
                    continue
                left = ", ".join(f"{table.qualified_name}.{c}" for c in fk.columns)
                right = ", ".join(f"{fk.referred_qualified_name}.{c}" for c in fk.referred_columns)
                lines.append(f"  {left} = {right}")
        return lines
