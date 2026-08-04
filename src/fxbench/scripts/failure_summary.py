"""Failure-type summary for a single evaluated predictions file.

Reads one evaluated per-record file (the committed ``evaluation.jsonl`` schema:
each line has ``outcome``, ``formula:{predicted,expected}`` and ``metrics``) and
prints a text breakdown of *why* the failures failed — no table, no plot.

Failure rows (``outcome`` not in {TP, TN}) are bucketed into six categories,
following the paper's deterministic failure taxonomy:

  Wrong formula type  — a different outer/nested function strategy.
  Wrong args          — right structure, but arguments/literals/operators differ.
  Wrong range         — right structure, but referenced cells/ranges differ.
  Partial correct     — same outer function, nested subexpressions differ.
  Execution error     — the predicted formula does not execute (syntax/runtime).
  Wrong abstain       — abstained on a real formula, or produced a formula when
                        the answer was to abstain.

The buckets are derived from the per-record metrics already in the file
(``SketchAccuracy``/``RangeAccuracy``/``CanExecute``/``Syntax``) plus a light
formula comparison, so this runs offline with no external dependencies.

Usage:
    uv run python -m fxbench.scripts.failure_summary <path-to-evaluation.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (key, display) in the order shown by the paper's failure figure.
CATEGORIES = [
    ("incorrect_formula_type", "Wrong formula type"),
    ("wrong_args", "Wrong args"),
    ("wrong_range", "Wrong range"),
    ("partial_correct", "Partial correct"),
    ("execution_error", "Execution error"),
    ("abstain_wrong", "Wrong abstain"),
]

_FUNC_PATTERN = re.compile(r"[A-Z_][A-Z0-9_.]*(?=\s*\()", re.IGNORECASE)
_REF_PATTERN = re.compile(r"\$?[A-Za-z]{1,3}\$?[0-9]+(?::\$?[A-Za-z]{1,3}\$?[0-9]+)?")


def _normalize(formula: str | None) -> str:
    if not formula:
        return ""
    s = str(formula).strip()
    if s.startswith("{=") and s.endswith("}"):
        s = "=" + s[2:-1].strip()
    for token in ("_xlfn.", "_XLFN.", "_xlws.", "_XLWS."):
        s = s.replace(token, "")
    return s


def _is_abstain(formula: str | None) -> bool:
    return _normalize(formula).lstrip("=").strip() == ""


def _functions(formula: str) -> Counter:
    return Counter(m.upper() for m in _FUNC_PATTERN.findall(_normalize(formula)))


def _root(formula: str) -> str:
    matches = _FUNC_PATTERN.findall(_normalize(formula))
    return matches[0].upper() if matches else ""


def _refs(formula: str) -> set[str]:
    norm = _normalize(formula)
    # strip function names so they are not mistaken for cell references
    norm = _FUNC_PATTERN.sub("", norm)
    return {r.upper() for r in _REF_PATTERN.findall(norm)}


def _metric(record: dict, name: str):
    for m in record.get("metrics", []):
        if m.get("name") == name:
            return m.get("value")
    return None


def is_failure(record: dict) -> bool:
    return record.get("outcome") not in {"TP", "TN"}


def _counter_overlap(a: Counter, b: Counter) -> float:
    total = sum((a | b).values())
    return 1.0 if total == 0 else sum((a & b).values()) / total


def classify(record: dict) -> str:
    predicted = (record.get("formula") or {}).get("predicted") or ""
    expected = (record.get("formula") or {}).get("expected") or ""
    pred_abstain = _is_abstain(predicted)
    exp_abstain = _is_abstain(expected)

    if (not exp_abstain and pred_abstain) or (exp_abstain and not pred_abstain):
        return "abstain_wrong"

    if record.get("outcome") == "Error":
        return "execution_error"
    if _metric(record, "CanExecute") is False or _metric(record, "Syntax") is False:
        return "execution_error"

    # Executable, wrong value: replicate the paper's structural decision tree using
    # a light function/reference comparison.
    pred_funcs = _functions(predicted)
    exp_funcs = _functions(expected)
    pred_root = _root(predicted)
    exp_root = _root(expected)
    same_strategy = bool(pred_root) and pred_root == exp_root and pred_funcs == exp_funcs
    refs_differ = _refs(predicted) != _refs(expected)

    if same_strategy and refs_differ:
        return "wrong_range"
    if same_strategy:
        return "wrong_args"
    if pred_root and pred_root == exp_root and _counter_overlap(pred_funcs, exp_funcs) >= 0.5:
        return "partial_correct"
    return "incorrect_formula_type"


def load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if records and "outcome" not in records[0]:
        raise SystemExit(
            f"{path} has no 'outcome' field — this script needs an evaluated per-record file "
            "(evaluation.jsonl), not a raw predictions file."
        )
    return records


def summarize(path: Path) -> None:
    records = load_records(path)
    counts = Counter(r.get("outcome") for r in records)
    failures = [r for r in records if is_failure(r)]
    cat_counts = Counter(classify(r) for r in failures)

    n = len(records)
    n_fail = len(failures)
    n_correct = counts.get("TP", 0) + counts.get("TN", 0)
    label = records[0] if records else {}
    builder = label.get("builder", "?")
    model = label.get("model", "?")

    print(f"\nFailure-type summary — {path.name}")
    print(f"  builder={builder}  model={model}")
    print(f"  records={n}  correct(TP+TN)={n_correct}  failures(FP+FN+Error)={n_fail}")
    print("  outcomes: " + ", ".join(f"{k}={counts[k]}" for k in ("TP", "FP", "TN", "FN", "Error") if counts.get(k)))

    width = max(len(name) for _, name in CATEGORIES)
    print(f"\n  Failure types (of {n_fail} failures):")
    if n_fail == 0:
        print("    (no failures)")
        return
    for key, name in CATEGORIES:
        c = cat_counts.get(key, 0)
        bar = "#" * round(c / n_fail * 30)
        print(f"    {name:<{width}}  {c:>4}  ({c / n_fail * 100:5.1f}%)  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a failure-type summary for one evaluated predictions file."
    )
    parser.add_argument("file", type=Path, help="Path to an evaluated per-record file (evaluation.jsonl).")
    args = parser.parse_args()
    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")
    summarize(args.file)


if __name__ == "__main__":
    main()
