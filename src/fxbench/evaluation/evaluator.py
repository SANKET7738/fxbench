"""Core evaluator: executes predicted and expected formulas, compares results."""

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from pathlib import Path

from fxbench.data_models import Problem
from fxbench.evaluation.compare import compare
from fxbench.evaluation.executor import ErrorValue, ExecutedValue, can_execute_value, is_infrastructure_error, execute
from fxbench.evaluation.formula_refs import RangeOverlapResult, range_overlap
from fxbench.evaluation.formula_sketch import formula_to_metric_sketch
from fxbench.evaluation.metrics import Metric, Outcome, OutcomeKind


class IEvaluator(ABC):
    @abstractmethod
    def evaluate(self, problem: Problem, predicted_formula: str, workbook_override: Path | None = None) -> Outcome:
        pass


class ExecutionEvaluator(IEvaluator):
    """Evaluates a predicted formula by executing it in the workbook and comparing to ground truth."""

    @staticmethod
    def _is_abstain_prediction(formula: str | None) -> bool:
        """Model abstained per protocol if it returned an empty/bare-`=` formula."""
        if formula is None:
            return False
        return formula.strip().lstrip("=").strip() == ""

    @staticmethod
    def _is_abstain_expected(problem: Problem) -> bool:
        """Ground-truth labels this case as one where the model SHOULD abstain."""
        ann = problem.annotation or {}
        return ann.get("function") == "ABSTAIN"

    def evaluate(self, problem: Problem, predicted_formula: str, workbook_override: Path | None = None) -> Outcome:
        workbook = workbook_override or Path(problem.workbook_path)
        worksheet = problem.worksheet_name
        address = problem.active_cell
        expected_formula = problem.formula

        expected_abstain = self._is_abstain_expected(problem)
        predicted_abstain = self._is_abstain_prediction(predicted_formula)

        # --- Abstention branch (no execution needed) ---
        if expected_abstain or predicted_abstain:
            if expected_abstain and predicted_abstain:
                kind = OutcomeKind.TN
            elif expected_abstain and not predicted_abstain:
                kind = OutcomeKind.FP   # false alarm: model produced a formula on an abstain case
            else:
                kind = OutcomeKind.FN   # miss: model abstained when a formula was expected

            edit = _edit_similarity(predicted_formula or "", "" if expected_abstain else expected_formula)
            overlap = _abstain_range_overlap(kind)
            return Outcome(
                outcome=kind,
                metrics=[
                    Metric(name="ValueShown", value=(kind == OutcomeKind.TN)),
                    Metric(name="Syntax", value=(kind == OutcomeKind.TN)),
                    Metric(name="CanExecute", value=(kind == OutcomeKind.TN)),
                    Metric(name="Edit", value=edit),
                    Metric(name="SketchAccuracy", value=(kind == OutcomeKind.TN)),
                    Metric(name="RangeAccuracy", value=(kind == OutcomeKind.TN)),
                    Metric(name="RangeOverlap", value=overlap.value),
                ],
                metadata={
                    "formula": {
                        "predicted": predicted_formula,
                        "expected": "" if expected_abstain else expected_formula,
                    },
                    "abstain": {
                        "expected": expected_abstain,
                        "predicted": predicted_abstain,
                    },
                    "range_overlap": overlap.metadata(),
                },
            )

        # --- Standard execution branch (formula expected, formula produced) ---
        # Exact string match → skip execution (avoids volatile formula issues)
        if predicted_formula == expected_formula:
            sketch = formula_to_metric_sketch(predicted_formula)
            overlap = range_overlap(
                predicted_formula,
                expected_formula,
                workbook_path=workbook,
                worksheet_name=worksheet,
                active_cell=address,
            )
            return Outcome(
                outcome=OutcomeKind.TP,
                metrics=[
                    Metric(name="ValueShown", value=True),
                    Metric(name="Syntax", value=True),
                    Metric(name="CanExecute", value=True),
                    Metric(name="Edit", value=1.0),
                    Metric(name="SketchAccuracy", value=True),
                    Metric(name="RangeAccuracy", value=True),
                    Metric(name="RangeOverlap", value=overlap.value),
                ],
                metadata={
                    "formula": {
                        "predicted": predicted_formula,
                        "expected": expected_formula,
                    },
                    "sketch": {
                        "predicted": sketch,
                        "expected": sketch,
                    },
                    "value": {
                        "predicted": "exact_match",
                        "expected": "exact_match",
                    },
                    "range_overlap": overlap.metadata(),
                },
            )

        v_predicted = execute(predicted_formula, workbook, worksheet, address)
        v_expected = execute(expected_formula, workbook, worksheet, address)

        # Both formulas failed → try falling back to original workbook
        if isinstance(v_predicted, ErrorValue) and isinstance(v_expected, ErrorValue):
            if v_predicted.error_type == "workbook_error" and v_expected.error_type == "workbook_error":
                original_path = (problem.annotation or {}).get("original_wb_path")
                if original_path:
                    fallback = Path(original_path)
                    if fallback.exists():
                        v_predicted = execute(predicted_formula, fallback, worksheet, address)
                        v_expected = execute(expected_formula, fallback, worksheet, address)

        # Both formulas still failed at the infrastructure/parser level → not a prediction issue.
        # Excel error values like #N/A are legitimate displayed results and are compared below.
        if (
            isinstance(v_predicted, ErrorValue)
            and isinstance(v_expected, ErrorValue)
            and is_infrastructure_error(v_predicted)
            and is_infrastructure_error(v_expected)
        ):
            return Outcome(
                outcome=OutcomeKind.Error,
                metrics=[],
                metadata={
                    "formula": {
                        "predicted": predicted_formula,
                        "expected": expected_formula,
                    },
                    "value": {
                        "predicted": v_predicted.model_dump(mode="json"),
                        "expected": v_expected.model_dump(mode="json"),
                    },
                },
            )

        # Only predicted failed → bad prediction. If both are ErrorValue, Excel
        # displayed-error results are legitimate values and are compared below.
        if isinstance(v_predicted, ErrorValue) and not isinstance(v_expected, ErrorValue):
            kind = OutcomeKind.FP
        else:
            is_correct = compare(v_predicted, v_expected, formatted=True)
            kind = OutcomeKind.TP if is_correct else OutcomeKind.FP

        sketch_predicted = formula_to_metric_sketch(predicted_formula)
        sketch_expected = formula_to_metric_sketch(expected_formula)

        refs_predicted = extract_cell_refs(predicted_formula)
        refs_expected = extract_cell_refs(expected_formula)
        sketch_match = sketch_predicted == sketch_expected
        overlap = range_overlap(
            predicted_formula,
            expected_formula,
            workbook_path=workbook,
            worksheet_name=worksheet,
            active_cell=address,
        )

        # Formula was accepted by Excel even if it returned a displayed error like #N/A.
        syntax_valid = can_execute_value(v_predicted)

        return Outcome(
            outcome=kind,
            metrics=[
                Metric(name="ValueShown", value=compare(v_predicted, v_expected, formatted=True)),
                Metric(name="Syntax", value=syntax_valid),
                Metric(name="CanExecute", value=syntax_valid),
                Metric(name="Edit", value=_edit_similarity(predicted_formula, expected_formula)),
                Metric(name="SketchAccuracy", value=sketch_match),
                Metric(name="RangeAccuracy", value=sketch_match and refs_predicted == refs_expected),
                Metric(name="RangeOverlap", value=overlap.value),
            ],
            metadata={
                "formula": {
                    "predicted": predicted_formula,
                    "expected": expected_formula,
                },
                "sketch": {
                    "predicted": sketch_predicted,
                    "expected": sketch_expected,
                },
                "value": {
                    "predicted": v_predicted.model_dump(mode="json"),
                    "expected": v_expected.model_dump(mode="json"),
                },
                "range_overlap": overlap.metadata(),
            },
        )


# Matches cell references: A1, $A$1, A1:B5, $A$1:$B$5, Sheet1!A1, 'Sheet Name'!A1:B5
_CELL_REF_PATTERN = re.compile(
    r"(?:'[^']*'!|\w+!)?"          # optional sheet reference
    r"\$?[A-Z]{1,3}\$?\d+"         # cell reference (e.g., A1, $B$2)
    r"(?::\$?[A-Z]{1,3}\$?\d+)?"   # optional range end (e.g., :B5)
)


def extract_cell_refs(formula: str) -> list[str]:
    """Extract all cell references from a formula, normalized (no $ signs)."""
    refs = _CELL_REF_PATTERN.findall(formula)
    return [ref.replace("$", "") for ref in refs]


def _edit_similarity(a: str, b: str) -> float:
    """String similarity ratio using SequenceMatcher (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def _abstain_range_overlap(kind: OutcomeKind) -> RangeOverlapResult:
    value = 1.0 if kind == OutcomeKind.TN else 0.0
    return RangeOverlapResult(
        value=value,
        overlap_count=0,
        expected_count=0,
        predicted_count=0,
    )
