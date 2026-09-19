"""Evaluation metrics.

Correctness is measured on several axes because each one misses something:

* **Validity** says the query is safe and every name in it exists. It says
  nothing about meaning.
* **Table and column overlap** says retrieval found the right data, scored as
  precision, recall and F1 so a query that reads twice as much as it needs
  does not score the same as one that reads exactly the right tables.
* **Semantic equivalence** compares the generated query with the reference
  after normalisation. Two correct queries are often written differently, so
  a mismatch here is evidence and not a verdict.
* **Result equality** compares what the two queries actually returned. This is
  the strongest signal available without a human, which is why it is the one
  that decides correctness when it can be computed.
* **Characteristics** catch a result of the wrong shape: empty when it should
  not be, one row when it should be many, or not ordered as asked.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot.errors import SqlglotError

from nl2sql.evaluation.dataset import ExpectedResult


@dataclass(frozen=True, slots=True)
class SetScore:
    """Precision, recall and F1 for one set comparison."""

    precision: float
    recall: float
    f1: float
    matched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()


def normalise_name(name: str) -> str:
    """Reduce a qualified name to a comparable form.

    Only the last two segments are kept, so ``dbo.facilities`` and
    ``main.facilities`` compare equal. A dataset is often written against one
    environment and evaluated in another where the schema is named
    differently, and that difference is not a modelling error.
    """
    cleaned = name.replace("[", "").replace("]", "").replace('"', "").strip().casefold()
    parts = [part for part in cleaned.split(".") if part]
    if not parts:
        return cleaned
    return parts[-1] if len(parts) == 1 else ".".join(parts[-2:])


def _tail(name: str) -> str:
    """Return the final segment of a qualified name."""
    return normalise_name(name).split(".")[-1]


def set_score(expected: Sequence[str], actual: Sequence[str], *, tails: bool = False) -> SetScore:
    """Score how well ``actual`` matches ``expected``.

    Names are compared at the same level of qualification as the less
    qualified side. A dataset that names ``facilities`` is comparing against
    whatever schema the database happens to use, and should not be marked
    wrong for it; a dataset that qualifies its names is taken at its word.
    """
    populated_expected = [item for item in expected if item.strip()]
    populated_actual = [item for item in actual if item.strip()]
    qualified_both = all("." in item for item in populated_expected) and all(
        "." in item for item in populated_actual
    )
    key = _tail if (tails or not qualified_both) else normalise_name
    wanted = {key(item) for item in populated_expected}
    found = {key(item) for item in populated_actual}
    if not wanted:
        return SetScore(precision=1.0, recall=1.0, f1=1.0)
    matched = wanted & found
    precision = len(matched) / len(found) if found else 0.0
    recall = len(matched) / len(wanted)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return SetScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched=tuple(sorted(matched)),
        missing=tuple(sorted(wanted - found)),
        extra=tuple(sorted(found - wanted)),
    )


def normalise_value(value: Any, tolerance: float) -> Any:
    """Return a value in a form two result sets can be compared by."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if tolerance <= 0:
            return float(value)
        digits = max(0, round(-math.log10(tolerance)))
        return round(float(value), digits)
    return str(value).strip()


def results_match(
    expected_rows: Sequence[Sequence[Any]],
    actual_rows: Sequence[Sequence[Any]],
    *,
    tolerance: float = 1e-6,
    column_order_sensitive: bool = False,
) -> bool:
    """Return whether two result sets carry the same values.

    Row order is ignored unless the case asks about ordering, which is checked
    separately, because two queries that differ only in ORDER BY still answer
    the same question.
    """
    if len(expected_rows) != len(actual_rows):
        return False

    def normalise(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
        normalised = []
        for row in rows:
            values = [normalise_value(value, tolerance) for value in row]
            if not column_order_sensitive:
                values = sorted(values, key=lambda item: (item is None, str(item)))
            normalised.append(tuple(values))
        return sorted(normalised, key=lambda item: tuple(str(value) for value in item))

    return normalise(expected_rows) == normalise(actual_rows)


def ast_equivalent(left: str, right: str, *, dialect: str) -> bool:
    """Return whether two statements are the same query once normalised."""
    if not left or not right:
        return False
    try:
        left_tree = sqlglot.parse_one(left, read=dialect)
        right_tree = sqlglot.parse_one(right, read=dialect)
    except SqlglotError:
        return False
    return _canonical(left_tree, dialect) == _canonical(right_tree, dialect)


def _canonical(tree: Any, dialect: str) -> str:
    """Render a statement in a form that ignores case and the row limit.

    The row limit is removed because the validator adds one to every generated
    query, and a reference query written without one is not thereby a
    different question.
    """
    from sqlglot import exp

    normalised = tree.copy()
    for key in ("limit", "offset"):
        if isinstance(normalised, exp.Select) and normalised.args.get(key) is not None:
            normalised.set(key, None)
    for identifier in normalised.find_all(exp.Identifier):
        identifier.set("this", identifier.this.casefold() if identifier.this else identifier.this)
    rendered: str = normalised.sql(dialect=dialect, comments=False, normalize=True, pretty=False)
    return rendered.casefold()


def check_characteristics(
    expected: ExpectedResult, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> list[str]:
    """Return a description of each expected characteristic the result failed."""
    failures: list[str] = []
    count = len(rows)

    if expected.non_empty is True and count == 0:
        failures.append("expected at least one row, got none")
    if expected.non_empty is False and count > 0:
        failures.append(f"expected no rows, got {count}")
    if expected.row_count is not None and count != expected.row_count:
        failures.append(f"expected exactly {expected.row_count} rows, got {count}")
    if expected.min_rows is not None and count < expected.min_rows:
        failures.append(f"expected at least {expected.min_rows} rows, got {count}")
    if expected.max_rows is not None and count > expected.max_rows:
        failures.append(f"expected at most {expected.max_rows} rows, got {count}")

    if expected.columns:
        found = {_tail(name) for name in columns}
        missing = [name for name in expected.columns if _tail(name) not in found]
        if missing:
            failures.append(f"missing expected columns: {', '.join(missing)}")

    if expected.ordered_by and rows:
        index = next(
            (i for i, name in enumerate(columns) if _tail(name) == _tail(expected.ordered_by)),
            None,
        )
        if index is None:
            failures.append(f"the ordering column {expected.ordered_by} is not in the result")
        else:
            values = [row[index] for row in rows if index < len(row)]
            comparable = [value for value in values if isinstance(value, int | float)]
            if len(comparable) == len(values) and len(values) > 1:
                pairs = list(itertools.pairwise(comparable))
                ordered = (
                    all(a >= b for a, b in pairs)
                    if expected.descending
                    else all(a <= b for a, b in pairs)
                )
                if not ordered:
                    direction = "descending" if expected.descending else "ascending"
                    failures.append(f"rows are not ordered {direction} by {expected.ordered_by}")
    return failures


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a percentile using linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[int(position)], 2)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def rate(numerator: int, denominator: int) -> float:
    """Return a rate, or zero when there is nothing to divide."""
    return round(numerator / denominator, 4) if denominator else 0.0
