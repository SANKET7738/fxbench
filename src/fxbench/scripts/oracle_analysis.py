"""Oracle analysis — Table 4 (oracle results on FxBench, GPT-5.2).

Reports, for each additive Oracle variant, three numbers aggregated from the
committed per-record oracle evaluation files:

  TP   — strict non-abstain execution-equivalent predictions, ``TP / N`` (does not
         include abstention).
  ACC  — execution accuracy, ``(TP + TN) / N`` (execution-equivalent + correct abstain).
  Δ    — ACC improvement over the base variant.

Variants are additive over the base oracle context and ordered by the scope of the
revealed signal (target-local → reference-local → sheet-wide → stacked → out-of-sheet).

The data is read from the tracked, self-contained evaluation files under
``Assets/Evaluations/FxBench/oracle_analysis/<Builder>/gpt-5.2/evaluation.jsonl``;
no Excel, workbooks, or benchmark loading are required.

Usage:
    uv run python -m fxbench.scripts.oracle_analysis
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

DATA_REL = Path("Assets") / "Evaluations" / "FxBench" / "oracle_analysis"
MODEL = "gpt-5.2"

# (builder dir, display label, group) — additive order; `group` controls separators.
# groups: "base" | "additive" | "stacked" | "nl"
VARIANTS = [
    ("OracleContextBuilder", "base", "base"),
    ("OracleWithCellFormatContextBuilder", "+ cell format", "additive"),
    ("OracleWithDepChainContextBuilder", "+ dependency chain", "additive"),
    ("OracleSameColumnContextBuilder", "+ same column", "additive"),
    ("OracleGridNeighborContextBuilder", "+ neighbor", "additive"),
    ("OracleWithAllFormulasContextBuilder", "+ formulas", "additive"),
    ("OracleKitchenSinkContextBuilder", "all of the above", "stacked"),
    ("OracleWithNLPromptContextBuilder", "+ NL", "nl"),
]


def repo_root() -> Path:
    """Repo root holding the Assets/ tree (src/fxbench/scripts/<this>.py)."""
    return Path(__file__).resolve().parents[3]


def _load_records(root: Path, builder: str) -> list[dict]:
    path = root / DATA_REL / builder / MODEL / "evaluation.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cell_metrics(records: list[dict]) -> tuple[float, float]:
    """Return (TP%, ACC%) for one oracle variant."""
    evaluable = [r for r in records if r["outcome"] in ("TP", "FP", "TN", "FN", "Error")]
    n = len(evaluable)
    tp = sum(1 for r in evaluable if r["outcome"] == "TP")
    correct = sum(1 for r in evaluable if r["outcome"] in ("TP", "TN"))
    return (tp / n * 100, correct / n * 100) if n else (0.0, 0.0)


def build_rows(root: Path) -> list[tuple[str, str, float, float, float | None]]:
    """[(group, label, TP%, ACC%, delta-or-None)] in table order."""
    rows = []
    base_acc = None
    for builder, label, group in VARIANTS:
        tp, acc = _cell_metrics(_load_records(root, builder))
        if base_acc is None:
            base_acc = acc
            delta = None
        else:
            delta = acc - base_acc
        rows.append((group, label, tp, acc, delta))
    return rows


def print_table(rows: list) -> None:
    headers = ["Oracle variant", "TP", "ACC", "\u0394"]
    body: list[list[str] | None] = []
    prev_group = None
    for group, label, tp, acc, delta in rows:
        if prev_group is not None and group != prev_group:
            body.append(None)  # separator between groups
        dstr = "\u2014" if delta is None else f"{delta:+.1f}"
        body.append([label, f"{tp:.1f}", f"{acc:.1f}", dstr])
        prev_group = group

    widths = [len(h) for h in headers]
    for r in body:
        if r is None:
            continue
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("\nOracle results on FxBench (GPT-5.2) — variants are additive over base")
    print(line)
    print(fmt(headers))
    print(line)
    for r in body:
        print(line if r is None else fmt(r))
    print(line)
    print("TP = non-abstain execution-equivalent (TP does not include abstention). "
          "ACC = (TP+TN)/N. \u0394 = ACC over base. All over 503 FxBench records.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle results (TP / ACC / Δ) on FxBench (GPT-5.2)."
    )
    parser.parse_args()
    print_table(build_rows(repo_root()))


if __name__ == "__main__":
    main()
