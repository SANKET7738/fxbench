from __future__ import annotations

import argparse
from pathlib import Path

from fxbench import FxBench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and materialize the public FxBench benchmark workbooks.",
    )
    return parser.parse_args()


def setup_fxbench() -> FxBench:
    return FxBench.load()


def workbook_count(benchmark: FxBench) -> int:
    return len({problem.workbook_path for problem in benchmark.problems})


def missing_workbooks(benchmark: FxBench) -> list[Path]:
    missing = []
    for problem in benchmark.problems:
        if not problem.workbook_path.exists():
            missing.append(problem.workbook_path)
    return missing


def main() -> None:
    parse_args()
    benchmark = setup_fxbench()
    missing = missing_workbooks(benchmark)
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise RuntimeError(f"{len(missing)} FxBench workbooks were not materialized. Examples: {examples}")

    print(f"Loaded {len(benchmark.problems):,} FxBench problems.")
    print(f"Materialized {workbook_count(benchmark):,} workbooks in {benchmark.workbook_cache_dir}.")


if __name__ == "__main__":
    main()
