"""Context coverage comparison — required-cell exposure coverage and token cost by strategy.

For each context strategy, reports the percentage of FxBench problems where the cells a model
needs (formula references and intent cells) are exposed by that strategy's context, plus the
full-prompt token cost. Coverage is computed from the intent annotations in
``Assets/Annotations/FxBench/intent_cells.jsonl`` and the materialised workbooks.

Prints a formatted table.

Usage:
    uv run python -m fxbench.scripts.context_coverage_comparison            # full run
    uv run python -m fxbench.scripts.context_coverage_comparison --take 5   # smoke test
    uv run python -m fxbench.scripts.context_coverage_comparison --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import tiktoken

from fxbench.analysis.coverage import compute_required_exposed_limits
from fxbench.context_builders import (
    ContextPromptBuilder,
    MiniSheetContextBuilder,
    PipeDelimitedContextBuilder,
    SheetCompressorContextBuilder,
    SpreadsheetCoderContextBuilder,
    ViewportContextBuilder,
)
from fxbench.fxbench import FxBench

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NEUTRAL_ENCODING = "cl100k_base"

UPPER_METRIC = {
    "v2": "theoretical_upper_limit",
    "v3": "theoretical_upper_limit_v3",
    "v4": "theoretical_upper_limit_v4",
}


@dataclass
class Row:
    """One table row: how to source its coverage % and its token cost."""

    label: str
    group: str                          # "main" | "minisheet" (controls separators)
    coverage_strategy: str              # strategy key in the coverage payload
    minisheet_width: int | None         # None => computed in the non-MiniSheet pass
    token_builder: type                 # context-builder class
    token_kwargs: dict = field(default_factory=dict)


ROWS: list[Row] = [
    Row("PipeDelimited", "main", "PipeDelimited", None, PipeDelimitedContextBuilder),
    Row("Viewport (\u00b120r/\u00b110c)", "main", "Windowed", None, ViewportContextBuilder),
    Row("Viewport (\u00b130r/\u00b120c)", "main", "Windowed@30r20c", None,
        ViewportContextBuilder, {"row_margin": 30, "col_margin": 20}),
    Row("Viewport (\u00b140r/\u00b130c)", "main", "Windowed@40r30c", None,
        ViewportContextBuilder, {"row_margin": 40, "col_margin": 30}),
    Row("SpreadsheetCoder", "main", "SpreadsheetCoder", None, SpreadsheetCoderContextBuilder),
    Row("SheetCompressor", "main", "SheetCompressor", None, SheetCompressorContextBuilder),
    Row("MiniSheet (w=1)", "minisheet", "MiniSheet", 1, MiniSheetContextBuilder, {"width": 1}),
    Row("MiniSheet (w=2)", "minisheet", "MiniSheet", 2, MiniSheetContextBuilder, {"width": 2}),
    Row("MiniSheet (w=3)", "minisheet", "MiniSheet", 3, MiniSheetContextBuilder, {"width": 3}),
    Row("MiniSheet (w=4)", "minisheet", "MiniSheet", 4, MiniSheetContextBuilder, {"width": 4}),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def problem_to_record(problem) -> dict:
    return {
        "id": problem.id,
        "formula": problem.formula,
        "cell": problem.active_cell,
        "sheet": problem.worksheet_name,
        "wb_local_path": str(problem.workbook_path),
        "function": (problem.annotation or {}).get("function", ""),
    }


def load_intent_file(path: Path) -> dict[str, dict]:
    """Load an intent_cells JSONL into {id: intent_cells_dict}, matching the shape
    ``context_evidence.load_intent_annotations`` produces (null -> {})."""
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[rec["id"]] = rec.get("intent_cells") or {}
    return by_id


# ---------------------------------------------------------------------------
# Coverage (delegates to compute_required_exposed_limits)
# ---------------------------------------------------------------------------

def _row_pcts(strategy_payload: dict, upper_key: str) -> dict[str, float]:
    return {
        "intent_printed": strategy_payload["intent_printed"]["pct"],
        "intent_mentioned": strategy_payload["intent_mentioned"]["pct"],
        "reference_printed": strategy_payload["reference_printed"]["pct"],
        "reference_mentioned": strategy_payload["reference_mentioned"]["pct"],
        "upper": strategy_payload[upper_key]["pct"],
    }


def compute_coverage(records: list[dict], upper: str, intent_cells_by_id: dict | None = None) -> dict[str, dict[str, float]]:
    upper_key = UPPER_METRIC[upper]
    by_key: dict[tuple[str, int | None], dict[str, float]] = {}

    non_ms = [r.coverage_strategy for r in ROWS if r.minisheet_width is None]
    print(f"  coverage: non-MiniSheet pass -> {non_ms}", flush=True)
    payload, _ = compute_required_exposed_limits(records, strategies=non_ms, intent_cells_by_id=intent_cells_by_id)
    for strat in non_ms:
        by_key[(strat, None)] = _row_pcts(payload[strat], upper_key)

    for width in sorted({r.minisheet_width for r in ROWS if r.minisheet_width is not None}):
        print(f"  coverage: MiniSheet width={width} pass", flush=True)
        payload, _ = compute_required_exposed_limits(
            records, strategies=["MiniSheet"], minisheet_width=width, intent_cells_by_id=intent_cells_by_id
        )
        by_key[("MiniSheet", width)] = _row_pcts(payload["MiniSheet"], upper_key)

    return {row.label: by_key[(row.coverage_strategy, row.minisheet_width)] for row in ROWS}


# ---------------------------------------------------------------------------
# Tokens (full-prompt cl100k via ContextPromptBuilder)
# ---------------------------------------------------------------------------

_tok_state: dict = {}


def _tok_init(builder_cls: type, kwargs: dict) -> None:
    _tok_state["pb"] = ContextPromptBuilder(context_builder=builder_cls(**kwargs))
    _tok_state["enc"] = tiktoken.get_encoding(NEUTRAL_ENCODING)


def _tok_count(problem) -> int:
    try:
        messages = _tok_state["pb"].build(problem)
    except Exception:
        return -1
    text = "\n".join(m.get("content", "") for m in messages)
    return len(_tok_state["enc"].encode(text, disallowed_special=()))


def compute_tokens(problems: list, builder_cls: type, kwargs: dict, workers: int) -> tuple[float, float]:
    counts: list[int] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_tok_init, initargs=(builder_cls, kwargs)) as pool:
        for n in pool.map(_tok_count, problems, chunksize=4):
            if n >= 0:
                counts.append(n)
    if not counts:
        return 0.0, 0.0
    return mean(counts), float(median(counts))


def compute_all_tokens(problems: list, workers: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in ROWS:
        print(f"  tokens: {row.label}  ({row.token_builder.__name__} {row.token_kwargs or ''})", flush=True)
        avg, med = compute_tokens(problems, row.token_builder, row.token_kwargs, workers)
        out[row.label] = {"avg": avg, "median": med}
    return out


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

HEADERS = ["Context", "IntentPr", "IntentMen", "RefPr", "RefMen", "Upper", "Tok Avg", "Tok Med"]


def print_table(coverage: dict, tokens: dict) -> None:
    rows: list[list[str] | None] = []
    prev_group = None
    for row in ROWS:
        if prev_group is not None and row.group != prev_group:
            rows.append(None)  # separator
        cov, tok = coverage[row.label], tokens[row.label]
        rows.append([
            row.label,
            f"{cov['intent_printed']:.1f}%",
            f"{cov['intent_mentioned']:.1f}%",
            f"{cov['reference_printed']:.1f}%",
            f"{cov['reference_mentioned']:.1f}%",
            f"{cov['upper']:.1f}%",
            f"{tok['avg']:,.0f}",
            f"{tok['median']:,.0f}",
        ])
        prev_group = row.group

    widths = [len(h) for h in HEADERS]
    for r in rows:
        if r is None:
            continue
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("\nRequired-cell exposure coverage and token cost by context strategy (FxBench)")
    print(line)
    print(fmt(HEADERS))
    print(line)
    for r in rows:
        print(line if r is None else fmt(r))
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Context coverage comparison (required-exposed + token cost)")
    parser.add_argument("--take", type=int, default=0, help="Limit to first N problems (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for token rendering")
    parser.add_argument("--split", type=str, default="train", help="Benchmark split (default: train)")
    parser.add_argument("--upper", choices=("v2", "v3", "v4"), default="v2", help="Upper-limit definition")
    parser.add_argument(
        "--intent-file",
        type=Path,
        default=None,
        help="Alternate intent_cells JSONL keyed by problem id "
        "(e.g. Assets/Annotations/FxBench/intent_cells_gpt-5.2.jsonl). "
        "Defaults to Assets/Annotations/FxBench/intent_cells.jsonl.",
    )
    args = parser.parse_args()

    problems = list(FxBench.load(split=args.split).problems)
    if args.take:
        problems = problems[: args.take]
    print(f"Loaded {len(problems)} FxBench problems (split={args.split})", flush=True)

    intent_cells_by_id = None
    if args.intent_file is not None:
        intent_cells_by_id = load_intent_file(args.intent_file)
        print(f"Using intent annotations from {args.intent_file} ({len(intent_cells_by_id)} entries)", flush=True)

    records = [problem_to_record(p) for p in problems]
    coverage = compute_coverage(records, args.upper, intent_cells_by_id=intent_cells_by_id)
    tokens = compute_all_tokens(problems, args.workers)

    print_table(coverage, tokens)


if __name__ == "__main__":
    main()
