"""Content baseline comparison — Table 3 (content-strategy baselines on FxBench).

For each content strategy and model, reports three metrics aggregated from the
committed per-record evaluation files:

  ACC   — Pass@1, ``(TP+TN)/N`` (abstain-aware).
  ACC†  — ACC divided by the strategy's context-evidence upper limit (how much of
          the reachable-evidence ceiling the model realizes).
  OVL   — mean expected-reference range overlap (recall over expanded cell sets).

The data is read from the tracked, self-contained evaluation files under
``Assets/Evaluations/FxBench/content_baselines/<Builder>/<Model>/evaluation.jsonl``;
no Excel, workbooks, or benchmark loading are required.

Usage:
    uv run python -m fxbench.scripts.content_baseline_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_REL = Path("Assets") / "Evaluations" / "FxBench" / "content_baselines"

# (model dir name, display name) — column order matches the paper table.
MODELS = [
    ("gpt-5.2", "GPT-5.2"),
    ("anthropic-claude-sonnet-4-6", "Sonnet-4-6"),
    ("Qwen3-8B", "Qwen3-8B"),
]

# (class letter, group caption, [(builder dir, display label)]) — F/W/D groups.
GROUPS = [
    ("F", "Full-sheet strategies", [
        ("HtmlContextBuilder", "HTML"),
        ("CsvContextBuilder", "CSV"),
        ("JsonContextBuilder", "JSON"),
        ("MarkdownContextBuilder", "Markdown"),
        ("PipeDelimitedContextBuilder", "Pipe-delimited"),
    ]),
    ("W", "Windowed approaches", [
        ("WindowedPipeDelimitedContextBuilder", "Viewport"),
        ("SpreadsheetCoderContextBuilder", "SpreadsheetCoder"),
    ]),
    ("D", "Compact strategies", [
        ("SheetCompressorContextBuilder", "SheetCompressor"),
        ("MiniSheetContextBuilder", "MiniSheet"),
    ]),
]

# ACC† upper limits: a strategy's context-evidence ceiling (fraction of problems
# whose required evidence is reachable in that strategy's context). Full-sheet
# strategies render the entire used range, so their ceiling is 1.0. The windowed
# and compact values come from the context-evidence (v4) comparison.
UPPER = {
    "HtmlContextBuilder": 1.0,
    "CsvContextBuilder": 1.0,
    "JsonContextBuilder": 1.0,
    "MarkdownContextBuilder": 1.0,
    "PipeDelimitedContextBuilder": 1.0,
    "WindowedPipeDelimitedContextBuilder": 0.690,
    "SpreadsheetCoderContextBuilder": 0.590,
    "SheetCompressorContextBuilder": 0.946,
    "MiniSheetContextBuilder": 0.883,
}

METRIC_HEADERS = ["ACC", "ACC\u2020", "OVL"]


def repo_root() -> Path:
    """Repo root holding the Assets/ tree (src/fxbench/scripts/<this>.py)."""
    return Path(__file__).resolve().parents[3]


def _load_records(root: Path, builder: str, model: str) -> list[dict]:
    path = root / DATA_REL / builder / model / "evaluation.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cell_metrics(records: list[dict], builder: str) -> tuple[float, float, float]:
    """Return (ACC, ACC-dagger, OVL) as percentages for one builder/model."""
    evaluable = [r for r in records if r["outcome"] in ("TP", "FP", "TN", "FN", "Error")]
    n = len(evaluable)
    correct = sum(1 for r in evaluable if r["outcome"] in ("TP", "TN"))
    acc = correct / n if n else 0.0

    overlaps = [r["range_overlap"] for r in records if r.get("range_overlap") is not None]
    ovl = sum(overlaps) / len(overlaps) if overlaps else 0.0

    upper = UPPER[builder]
    acc_upper = acc / upper if upper else 0.0
    return acc * 100, acc_upper * 100, ovl * 100


def build_matrix(root: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    matrix: dict[tuple[str, str], tuple[float, float, float]] = {}
    for _, _, builders in GROUPS:
        for builder, _label in builders:
            for model_id, _name in MODELS:
                records = _load_records(root, builder, model_id)
                matrix[(builder, model_id)] = _cell_metrics(records, builder)
    return matrix


def print_table(matrix: dict) -> None:
    headers = ["Class", "Representation"]
    for _, name in MODELS:
        headers += [f"{name} {h}" for h in METRIC_HEADERS]

    rows: list[list[str] | None] = []
    for gi, (cls, _caption, builders) in enumerate(GROUPS):
        if gi > 0:
            rows.append(None)
        for ri, (builder, label) in enumerate(builders):
            cells = [cls if ri == 0 else "", label]
            for model_id, _ in MODELS:
                acc, accu, ovl = matrix[(builder, model_id)]
                cells += [f"{acc:.1f}", f"{accu:.1f}", f"{ovl:.1f}"]
            rows.append(cells)

    widths = [len(h) for h in headers]
    for r in rows:
        if r is None:
            continue
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) if i < 2 else c.rjust(widths[i]) for i, c in enumerate(cells))

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("\nContent strategy baselines on FxBench (503 records)")
    print(line)
    print(fmt(headers))
    print(line)
    for r in rows:
        print(line if r is None else fmt(r))
    print(line)
    print("ACC = Pass@1 (TP+TN)/N. ACC\u2020 = ACC / context-evidence upper limit. "
          "OVL = mean range overlap. All over 503 FxBench records.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content strategy baselines (ACC / ACC-dagger / OVL) on FxBench."
    )
    parser.parse_args()
    print_table(build_matrix(repo_root()))


if __name__ == "__main__":
    main()
