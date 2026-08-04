"""Evaluation metrics: Fraction, Metric, Outcome, Metrics, and aggregation utilities."""

import math
from collections import Counter
from enum import Enum
from itertools import chain
from typing import Any, Optional, Self, Union

from pydantic import BaseModel, Field, model_serializer, model_validator


class OutcomeKind(str, Enum):
    TP = "TP"        # formula expected, model produced correct formula
    FP = "FP"        # formula expected & wrong, OR abstain expected & model produced formula
    TN = "TN"        # abstain expected, model abstained
    FN = "FN"        # formula expected, model abstained
    Error = "Error"  # infrastructure / workbook-open failure (excluded from accuracy)


class Fraction(BaseModel):
    n: Union[float, int]
    d: Union[float, int]

    @property
    def value(self) -> float:
        if self.d == 0:
            return 0.0
        return float(self.n) / self.d

    def __str__(self) -> str:
        if isinstance(self.n, int) and isinstance(self.d, int):
            return f"{self.value:7.2%} ({self.n}/{self.d})"
        return f"{self.value:7.2%}%"

    @model_serializer(when_used="json")
    def serialize_fraction(self) -> str:
        return f"{self.n}/{self.d}"

    @model_validator(mode="before")
    @classmethod
    def deserialize_fraction(cls, data: Any) -> Any:
        if isinstance(data, str) and "/" in data:
            parts = data.split("/")
            return {
                "n": float(parts[0]) if "." in parts[0] else int(parts[0]),
                "d": float(parts[1]) if "." in parts[1] else int(parts[1]),
            }
        if isinstance(data, dict):
            if "n" in data and "d" in data:
                return data
        return data


class Metric(BaseModel):
    name: str
    value: Union[float, int, bool, Fraction]
    parts: Optional[list["Metric"]] = Field(
        default=None, exclude=True
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, exclude=True
    )

    def model_dump(self, **kwargs) -> dict:
        d = super().model_dump(**kwargs)
        if self.parts:
            d["parts"] = [p.model_dump(**kwargs) for p in self.parts]
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class Metrics(BaseModel):
    """Aggregated evaluation metrics."""

    tp: Fraction
    fp: Fraction
    tn: Fraction
    fn: Fraction
    error: Fraction
    others: dict[str, Union[float, int, bool, Fraction]] = Field(default_factory=dict)

    @property
    def accuracy(self) -> Fraction:
        # Overall correctness = correct formulas + correct abstentions
        d = self.tp.d
        return Fraction(n=self.tp.n + self.tn.n, d=d)

    @property
    def precision(self) -> Fraction:
        # Of predicted formulas (TP+FP), how many were correct? FP includes false alarms.
        d = self.tp.n + self.fp.n
        return Fraction(n=self.tp.n, d=d) if d else Fraction(n=0, d=0)

    @property
    def recall(self) -> Fraction:
        # Of cases where a formula was expected (TP+FP_value+FN), how many did we get right?
        # Approximation: TP / (TP + FN); does not split FP into value-error vs false-alarm.
        d = self.tp.n + self.fn.n
        return Fraction(n=self.tp.n, d=d) if d else Fraction(n=0, d=0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        assert self.tp.d == self.fp.d == self.tn.d == self.fn.d == self.error.d, (
            "Denominators must be equal."
        )
        assert self.tp.n + self.fp.n + self.tn.n + self.fn.n + self.error.n == self.tp.d, (
            "Numerators must sum to denominator."
        )
        return self


class Outcome(BaseModel):
    outcome: OutcomeKind
    metrics: list[Metric] = Field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None


def aggregate(outcomes: list[Outcome]) -> Metrics:
    """Aggregate outcomes into pass@1 metrics."""
    counts = Counter(o.outcome for o in outcomes)
    tp = counts[OutcomeKind.TP]
    fp = counts[OutcomeKind.FP]
    tn = counts[OutcomeKind.TN]
    fn = counts[OutcomeKind.FN]
    error = counts[OutcomeKind.Error]
    total = counts.total()
    others = aggregate_metrics(
        collect_metrics(chain.from_iterable(o.metrics for o in outcomes))
    )
    return Metrics(
        tp=Fraction(n=tp, d=total),
        fp=Fraction(n=fp, d=total),
        tn=Fraction(n=tn, d=total),
        fn=Fraction(n=fn, d=total),
        error=Fraction(n=error, d=total),
        others=others,
    )


def collect_metrics(metrics) -> dict[str, list[Union[float, int, bool]]]:
    """Collect metric values by name, supporting nested parts with dot-notation paths."""
    metrics_dict: dict[str, list] = {}
    queue = [(m, m.name) for m in metrics]
    while queue:
        metric, path = queue.pop()
        if path not in metrics_dict:
            metrics_dict[path] = []
        metrics_dict[path].append(metric.value)
        if metric.parts:
            for part in metric.parts:
                queue.append((part, f"{path}.{part.name}"))
    return {k: v for k, v in metrics_dict.items() if len(v) > 0}


def aggregate_metrics(collected: dict[str, list[Union[float, int, bool]]]) -> dict[str, Union[float, Fraction]]:
    """Aggregate collected metrics into fractions or means."""
    aggregated: dict[str, Union[float, Fraction]] = {}
    for name, values in collected.items():
        if len(values) == 0:
            continue
        if all(isinstance(v, (bool, int)) for v in values):
            aggregated[name] = Fraction(n=sum(values), d=len(values))
        if all(isinstance(v, float) for v in values):
            aggregated[name] = sum(values) / len(values)
    return aggregated


def pass_at_k(total: int, correct: int, k: int) -> float:
    """Pass@k rate."""
    if total - correct < k:
        return 1.0
    if k == 1:
        return correct / total
    return 1.0 - math.prod(1.0 - k / x for x in range(total - correct + 1, total + 1))
