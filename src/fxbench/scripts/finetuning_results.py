"""Fine-tuning results — Table 5 (combined fine-tuning results on FxBench).

Prints sampled pass@k (eight completions per problem) across two comparisons:
model-size scaling on the Curated dataset, and dataset choice at fixed Qwen3-8B
scale, with base-context and a closed-model (GPT-5.2) reference.

  pass@1  — expected fraction of single samples that are execution-equivalent.
  pass@3  — expected fraction solved within three samples.
  Best@8  — fraction of problems solved by at least one of the eight samples.

The numbers are read from the committed aggregate file
``Assets/Evaluations/FxBench/finetuning_results.json``. The underlying sampled
predictions (eight completions per problem) are committed alongside under
``Assets/SampledPredictions/FxBench/<slug>/sampled.json`` as the analysis
reference. No Excel, workbooks, or benchmark loading are required.

Usage:
    uv run python -m fxbench.scripts.finetuning_results
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

RESULTS_REL = Path("Assets") / "Evaluations" / "FxBench" / "finetuning_results.json"
SECTION_ORDER = ["base", "model_size", "dataset", "closed"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_rows() -> list[dict]:
    path = repo_root() / RESULTS_REL
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.sort(key=lambda r: SECTION_ORDER.index(r["section"]))
    return rows


def _best8_pct(r: dict) -> float:
    return r["best_at_8"] / r["total"] * 100


def _records(r: dict) -> str:
    return "--" if r["rows"] is None else f"{r['rows']:,}"


def _training(r: dict) -> str:
    return "--" if r["training_data"] in (None, "none") else r["training_data"]


def print_table(rows: list[dict]) -> None:
    headers = ["Comparison", "Model", "Training data", "Records", "pass@1", "pass@3", "Best@8"]
    body: list[list[str] | None] = []
    prev = None
    for r in rows:
        if prev is not None and r["section"] != prev:
            body.append(None)
        body.append([
            r["comparison"], r["model"], _training(r), _records(r),
            f"{r['pass_at_1'] * 100:.1f}%", f"{r['pass_at_3'] * 100:.1f}%", f"{_best8_pct(r):.1f}%",
        ])
        prev = r["section"]

    widths = [len(h) for h in headers]
    for row in body:
        if row is None:
            continue
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) if i < 3 else c.rjust(widths[i]) for i, c in enumerate(cells))

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("\nCombined fine-tuning results on FxBench (sampled, 8 completions/problem)")
    print(line)
    print(fmt(headers))
    print(line)
    for row in body:
        print(line if row is None else fmt(row))
    print(line)
    print("pass@1/pass@3 are sampled pass@k; Best@8 = fraction solved by at least one of eight "
          "samples. All over 503 FxBench records.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined fine-tuning results (pass@1 / pass@3 / Best@8) on FxBench."
    )
    parser.parse_args()
    print_table(load_rows())


if __name__ == "__main__":
    main()
