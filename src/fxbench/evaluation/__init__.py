"""Evaluation package for spreadsheet formula completion."""

from fxbench.evaluation.compare import compare
from fxbench.evaluation.evaluator import ExecutionEvaluator, IEvaluator
from fxbench.evaluation.executor import (
    CellLocation,
    CellValue,
    ColumnValue,
    ErrorValue,
    ExcelBook,
    ExcelPool,
    ExecutedValue,
    RangeValue,
    execute,
)
from fxbench.evaluation.formula_refs import (
    FormulaReferenceSet,
    RangeOverlapResult,
    extract_referenced_cells,
    range_overlap,
)
from fxbench.evaluation.formula_sketch import formula_to_metric_sketch
from fxbench.evaluation.metrics import (
    Fraction,
    Metric,
    Metrics,
    Outcome,
    OutcomeKind,
    aggregate,
    aggregate_metrics,
    collect_metrics,
    pass_at_k,
)
from fxbench.evaluation.parser import IResponseParser, JsonParser, ParsedResult

__all__ = [
    "ExecutionEvaluator",
    "IEvaluator",
    "compare",
    "execute",
    "ExcelPool",
    "ExcelBook",
    "CellLocation",
    "CellValue",
    "ColumnValue",
    "RangeValue",
    "ErrorValue",
    "ExecutedValue",
    "FormulaReferenceSet",
    "RangeOverlapResult",
    "extract_referenced_cells",
    "range_overlap",
    "formula_to_metric_sketch",
    "Fraction",
    "Metric",
    "Metrics",
    "Outcome",
    "OutcomeKind",
    "aggregate",
    "aggregate_metrics",
    "collect_metrics",
    "pass_at_k",
    "IResponseParser",
    "JsonParser",
    "ParsedResult",
]
