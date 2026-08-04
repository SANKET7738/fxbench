from __future__ import annotations

import argparse
from pathlib import Path

from fxbench.benchmark_curation.pipeline_paths import DEFAULT_RECORDS_DIR, read_jsonl, write_jsonl
from fxbench.benchmark_curation.rank_and_select import (
    MAX_PER_FUNCTION,
    RANK_SELECTION_SEED,
    select_top_records,
    strip_internal_fields,
    summarize_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--max-per-function", type=int, default=MAX_PER_FUNCTION)
    parser.add_argument("--seed", type=int, default=RANK_SELECTION_SEED)
    return parser.parse_args()


def load_rank_records(records_dir: Path) -> list[dict]:
    return read_jsonl(records_dir / "deduped_by_intent.jsonl")


def write_selected_records(records_dir: Path, records: list[dict]) -> Path:
    output_path = records_dir / "top100_selected.jsonl"
    write_jsonl(output_path, [strip_internal_fields(record) for record in records])
    return output_path


def print_selection_report(summary: dict, max_per_function: int, output_path: Path) -> None:
    total_after = summary["total_after"]

    print()
    print("=" * 60)
    print("RANK & SELECT REPORT")
    print("=" * 60)
    print(f"Total records before:   {summary['total_before']:,}")
    print(f"Total records after:    {summary['total_after']:,}")
    print(f"Reduction:              {summary['reduction']:,} ({summary['reduction_percent']:.1f}%)")
    print(f"Functions:              {summary['function_count']}")
    print(f"Unique signatures:      {summary['unique_signatures']:,}")
    print(f"Unique sketches:        {summary['unique_sketches']:,}")
    print(f"Unique workbooks:       {summary['unique_workbooks']:,}")

    print()
    print("Intent distribution:")
    for source, count in summary["intent_distribution"].most_common():
        percent = 0.0
        if total_after:
            percent = count / total_after * 100
        print(f"  {source:<22s} {count:>6,} ({percent:5.1f}%)")

    capped_functions = summary["capped_functions"]
    print()
    print(f"Functions capped at {max_per_function} ({len(capped_functions)}):")
    for function_name, before, after in capped_functions:
        print(f"  {function_name:<25s} {before:>6,} -> {after:>4}")

    print()
    print("Records per function distribution:")
    for bucket in summary["records_per_function_distribution"]:
        print(f"  {bucket['label']:>10s}: {bucket['function_count']:>4} functions, {bucket['record_count']:>6,} records")

    print()
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()
    records = load_rank_records(args.records_dir)
    selected_records, stats_before, stats_after = select_top_records(
        records,
        max_per_function=args.max_per_function,
        seed=args.seed,
    )
    output_path = write_selected_records(args.records_dir, selected_records)
    summary = summarize_selection(
        records,
        selected_records,
        stats_before,
        stats_after,
        max_per_function=args.max_per_function,
    )
    print_selection_report(summary, args.max_per_function, output_path)


if __name__ == "__main__":
    main()
