"""In process metrics with Prometheus text exposition.

Counters and histograms are kept in memory and rendered in the Prometheus text
format on demand, which keeps the service free of an exporter dependency. A
scraper, or the Azure Monitor agent in a container app, reads the endpoint when
``observability.metrics_endpoint_enabled`` is true.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

LabelSet = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS: tuple[float, ...] = (
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    30000,
    60000,
)


def _labels(labels: dict[str, str] | None) -> LabelSet:
    return tuple(sorted((labels or {}).items()))


@dataclass
class _Histogram:
    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self.total += value
        self.observations += 1
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1


class MetricsRegistry:
    """Thread safe counters and histograms."""

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self._lock = threading.Lock()
        self._buckets = buckets
        self._counters: dict[str, dict[LabelSet, float]] = defaultdict(dict)
        self._histograms: dict[str, dict[LabelSet, _Histogram]] = defaultdict(dict)

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Add ``value`` to a counter."""
        key = _labels(labels)
        with self._lock:
            series = self._counters[name]
            series[key] = series.get(key, 0.0) + value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record one observation in a histogram."""
        key = _labels(labels)
        with self._lock:
            series = self._histograms[name]
            if key not in series:
                series[key] = _Histogram(self._buckets)
            series[key].observe(value)

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Return the current value of a counter series."""
        with self._lock:
            return self._counters.get(name, {}).get(_labels(labels), 0.0)

    def render_prometheus(self) -> str:
        """Render every series in the Prometheus text exposition format."""

        def fmt(labels: LabelSet, extra: tuple[tuple[str, str], ...] = ()) -> str:
            pairs = [*labels, *extra]
            if not pairs:
                return ""
            inner = ",".join(
                f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                for key, value in pairs
            )
            return "{" + inner + "}"

        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                for labels, value in series.items():
                    lines.append(f"{name}{fmt(labels)} {value}")
            for name, hist_series in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                for labels, histogram in hist_series.items():
                    for bound, count in zip(histogram.buckets, histogram.counts, strict=True):
                        lines.append(f"{name}_bucket{fmt(labels, (('le', str(bound)),))} {count}")
                    lines.append(
                        f"{name}_bucket{fmt(labels, (('le', '+Inf'),))} {histogram.observations}"
                    )
                    lines.append(f"{name}_sum{fmt(labels)} {histogram.total}")
                    lines.append(f"{name}_count{fmt(labels)} {histogram.observations}")
        return "\n".join(lines) + "\n"
