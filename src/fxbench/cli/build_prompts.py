from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from fxbench import FxBench
from fxbench.context_builders import (
    ContextPromptBuilder,
    CsvContextBuilder,
    HtmlContextBuilder,
    JsonContextBuilder,
    MarkdownContextBuilder,
    MiniSheetContextBuilder,
    OracleContextBuilder,
    OracleGridNeighborContextBuilder,
    OracleKitchenSinkContextBuilder,
    OracleSameColumnContextBuilder,
    OracleWithAllFormulasContextBuilder,
    OracleWithCellFormatContextBuilder,
    OracleWithDepChainContextBuilder,
    OracleWithNLPromptContextBuilder,
    PipeDelimitedContextBuilder,
    SheetCompressorContextBuilder,
    SpreadsheetCoderContextBuilder,
    ViewportContextBuilder,
)
from fxbench.context_builders.serializers._utils import sanitize_formula
from fxbench.data_models import Problem
from fxbench.utils import validate_path_component

DEFAULT_OUTPUT_ROOT = Path("Assets") / "Prompts"

_CONTEXT_BUILDERS = {
    "CsvContextBuilder": CsvContextBuilder,
    "HtmlContextBuilder": HtmlContextBuilder,
    "JsonContextBuilder": JsonContextBuilder,
    "MarkdownContextBuilder": MarkdownContextBuilder,
    "MiniSheetContextBuilder": MiniSheetContextBuilder,
    "OracleContextBuilder": OracleContextBuilder,
    "OracleWithCellFormatContextBuilder": OracleWithCellFormatContextBuilder,
    "OracleWithDepChainContextBuilder": OracleWithDepChainContextBuilder,
    "OracleSameColumnContextBuilder": OracleSameColumnContextBuilder,
    "OracleGridNeighborContextBuilder": OracleGridNeighborContextBuilder,
    "OracleWithAllFormulasContextBuilder": OracleWithAllFormulasContextBuilder,
    "OracleKitchenSinkContextBuilder": OracleKitchenSinkContextBuilder,
    "OracleWithNLPromptContextBuilder": OracleWithNLPromptContextBuilder,
    "PipeDelimitedContextBuilder": PipeDelimitedContextBuilder,
    "SheetCompressorContextBuilder": SheetCompressorContextBuilder,
    "SpreadsheetCoderContextBuilder": SpreadsheetCoderContextBuilder,
    "ViewportContextBuilder": ViewportContextBuilder,
}

_CONTEXT_BUILDER_ALIASES = {
    "csv": "CsvContextBuilder",
    "html": "HtmlContextBuilder",
    "json": "JsonContextBuilder",
    "markdown": "MarkdownContextBuilder",
    "minisheet": "MiniSheetContextBuilder",
    "oracle": "OracleContextBuilder",
    "oracle-cell-format": "OracleWithCellFormatContextBuilder",
    "oracle-dep-chain": "OracleWithDepChainContextBuilder",
    "oracle-same-column": "OracleSameColumnContextBuilder",
    "oracle-grid-neighbor": "OracleGridNeighborContextBuilder",
    "oracle-all-formulas": "OracleWithAllFormulasContextBuilder",
    "oracle-kitchen-sink": "OracleKitchenSinkContextBuilder",
    "oracle-nl-prompt": "OracleWithNLPromptContextBuilder",
    "pipe": "PipeDelimitedContextBuilder",
    "pipe-delimited": "PipeDelimitedContextBuilder",
    "sheet-compressor": "SheetCompressorContextBuilder",
    "sheetcompressor": "SheetCompressorContextBuilder",
    "spreadsheetcoder": "SpreadsheetCoderContextBuilder",
    "viewport": "ViewportContextBuilder",
}

_BENCHMARKS = {
    "FxBench": FxBench,
}

_worker_builder: ContextPromptBuilder | None = None
_worker_training = False


def available_context_builders() -> str:
    return ", ".join(sorted(_CONTEXT_BUILDERS))


def get_context_builder(name: str) -> type:
    canonical_name = _CONTEXT_BUILDER_ALIASES.get(name.lower(), name)
    for builder_name, builder_class in _CONTEXT_BUILDERS.items():
        if builder_name.lower() == canonical_name.lower():
            return builder_class
    raise KeyError(f"Unknown context builder {name!r}. Available: {available_context_builders()}")


def get_benchmark(name: str) -> type:
    for benchmark_name, benchmark_class in _BENCHMARKS.items():
        if benchmark_name.lower() == name.lower():
            return benchmark_class
    available = ", ".join(sorted(_BENCHMARKS))
    raise KeyError(f"Unknown benchmark {name!r}. Available: {available}")


def parse_builder_args(raw_args: str | None) -> dict[str, Any]:
    if raw_args is None:
        return {}
    parsed = json.loads(raw_args)
    if not isinstance(parsed, dict):
        raise ValueError("--builder-args must be a JSON object")
    return parsed


def _init_worker(context_builder_name: str, builder_args: dict[str, Any] | None, training: bool) -> None:
    global _worker_builder, _worker_training

    context_builder_class = get_context_builder(context_builder_name)
    context_builder = context_builder_class(**(builder_args or {}))
    _worker_builder = ContextPromptBuilder(context_builder=context_builder)
    _worker_training = training


def _close_worker() -> None:
    global _worker_builder

    if _worker_builder is None:
        return
    context_builder = _worker_builder.context_builder
    close = getattr(context_builder, "close", None)
    if callable(close):
        close()
    _worker_builder = None


def _assistant_formula(problem: Problem) -> str:
    annotation = problem.annotation or {}
    if str(annotation.get("function", "")).upper() == "ABSTAIN":
        return ""
    if not problem.formula:
        return ""
    formula = problem.formula if problem.formula.startswith("=") else f"={problem.formula}"
    return sanitize_formula(formula)


def _build_one(item: tuple[int, Problem]) -> tuple[int, str | None, str | None]:
    if _worker_builder is None:
        raise RuntimeError("Prompt worker was not initialized")

    index, problem = item
    problem_id = problem.id or str(index)
    try:
        messages = _worker_builder.build(problem)
        if _worker_training:
            messages.append({"role": "assistant", "content": json.dumps({"formula": _assistant_formula(problem)})})
        line = json.dumps({"id": str(problem_id), "messages": messages})
        return index, line, None
    except Exception as error:
        return index, None, f"Error on problem {problem_id}: {error}"


def _build_many(items: list[tuple[int, Problem]]) -> list[tuple[int, str | None, str | None]]:
    return [_build_one(item) for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and cache LLM prompts for FxBench or a JSONL dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark", help="Benchmark class name. Currently supported: FxBench.")
    source.add_argument(
        "--dataset",
        type=Path,
        help="Path to a JSONL file with formula records.",
    )
    parser.add_argument("--dataset-name", help="Override the output dataset name when using --dataset.")
    parser.add_argument(
        "--context-builder",
        required=True,
        help=f"Context builder class or alias. Available: {available_context_builders()}",
    )
    parser.add_argument("--builder-args", help='JSON object of context-builder kwargs, e.g. \'{"row_margin": 5}\'.')
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reuse-prompts", type=Path, action="append", default=[])
    parser.add_argument("--force", action="store_true", help="Overwrite an existing prompts.jsonl file.")
    parser.add_argument("--take", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-file-size", type=float, default=None, help="Skip workbooks larger than this size in MB.")
    parser.add_argument("--training", action="store_true", help="Append an assistant message with the ground-truth formula.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory for relative workbook paths in JSONL datasets.")
    return parser.parse_args()


def _field(record: dict[str, Any], names: tuple[str, ...], row_index: int) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise ValueError(f"Dataset row {row_index} is missing one of: {', '.join(names)}")


def _resolve_workbook_path(workbook_value: Any, root: Path) -> Path:
    workbook_path = Path(str(workbook_value))
    if workbook_path.is_absolute():
        return workbook_path
    return (root / workbook_path).resolve()


def _annotation_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    annotation = record.get("annotation")
    if isinstance(annotation, dict):
        return annotation
    if "function" in record:
        return {"function": record["function"]}
    return None


def load_dataset_problems(dataset_path: Path, data_root: Path | None = None) -> tuple[str, list[Problem]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    root = data_root.resolve() if data_root is not None else Path.cwd()
    problems = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            workbook_path = _resolve_workbook_path(
                _field(record, ("wb_local_path", "workbook_path"), row_index),
                root,
            )
            problems.append(
                Problem(
                    workbook_path=workbook_path,
                    worksheet_name=str(_field(record, ("sheet", "target_sheet", "worksheet_name"), row_index)),
                    active_cell=str(_field(record, ("cell", "target_cell", "active_cell"), row_index)),
                    formula=str(_field(record, ("formula",), row_index)),
                    id=str(record.get("id", row_index)),
                    prefix=str(record.get("prefix", "=")),
                    annotation=_annotation_from_record(record),
                )
            )

    dataset_name = dataset_path.stem
    return dataset_name, problems


def load_benchmark_problems(args: argparse.Namespace) -> tuple[str, list[Problem]]:
    benchmark_class = get_benchmark(args.benchmark)
    benchmark = benchmark_class.load()
    return benchmark.spec.name, benchmark.problems


def load_reusable_prompts(paths: list[Path]) -> dict[str, str]:
    reusable = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Reusable prompts file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                prompt_id = json.loads(stripped).get("id")
                if not prompt_id:
                    raise ValueError(f"Missing prompt id in {path}:{line_number}")
                reusable[str(prompt_id)] = stripped
    return reusable


def select_problems(problems: list[Problem], start: int, take: int | None, max_file_size: float | None) -> list[Problem]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if take is not None and take < 0:
        raise ValueError("--take must be >= 0")

    selected = problems
    if max_file_size is not None:
        max_bytes = max_file_size * 1024 * 1024
        selected = [
            problem
            for problem in selected
            if problem.workbook_path.exists() and problem.workbook_path.stat().st_size <= max_bytes
        ]
    if start:
        selected = selected[start:]
    if take is not None:
        selected = selected[:take]
    return selected


def build_prompts(
    problems: list[Problem],
    context_builder_name: str,
    builder_args: dict[str, Any],
    *,
    workers: int,
    chunk_size: int,
    training: bool,
    reusable_prompts: dict[str, str],
) -> tuple[list[tuple[int, str | None, str | None]], int]:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    results: list[tuple[int, str | None, str | None]] = []
    build_candidates: list[tuple[int, Problem]] = []
    reused = 0
    for index, problem in enumerate(problems):
        problem_id = problem.id or str(index)
        reusable_line = reusable_prompts.get(problem_id)
        if reusable_line is None:
            build_candidates.append((index, problem))
        else:
            results.append((index, reusable_line, None))
            reused += 1

    work_items = sorted(build_candidates, key=lambda item: str(item[1].workbook_path))
    work_chunks = [
        work_items[index : index + chunk_size]
        for index in range(0, len(work_items), chunk_size)
    ]
    errors = 0

    if workers == 1:
        _init_worker(context_builder_name, builder_args, training)
        try:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Building prompts", total=len(work_items))
                for chunk in work_chunks:
                    chunk_results = _build_many(chunk)
                    results.extend(chunk_results)
                    for _, _, error in chunk_results:
                        if error:
                            errors += 1
                            progress.console.print(f"[red]WARNING: {error}[/]")
                    progress.advance(task, advance=len(chunk_results))
        finally:
            _close_worker()
        return results, reused

    max_in_flight = max(1, min(len(work_chunks), workers * 2))
    if work_chunks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(context_builder_name, builder_args, training),
        ) as executor:
            work_iter = iter(work_chunks)
            futures = {}

            def submit_next() -> bool:
                try:
                    chunk = next(work_iter)
                except StopIteration:
                    return False
                futures[executor.submit(_build_many, chunk)] = chunk
                return True

            for _ in range(max_in_flight):
                submit_next()

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Building prompts", total=len(work_items))
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        submit_next()
                        chunk_results = future.result()
                        results.extend(chunk_results)
                        for _, _, error in chunk_results:
                            if error:
                                errors += 1
                                progress.console.print(f"[red]WARNING: {error}[/]")
                        progress.advance(task, advance=len(chunk_results))

    return results, reused


def main() -> None:
    args = parse_args()
    builder_args = parse_builder_args(args.builder_args)
    context_builder_class = get_context_builder(args.context_builder)
    builder_name = context_builder_class.__name__

    if args.dataset is not None:
        benchmark_name, problems = load_dataset_problems(args.dataset, args.data_root)
        if args.dataset_name:
            benchmark_name = args.dataset_name
        print(f"Loaded {len(problems):,} problems from dataset: {args.dataset}")
    else:
        benchmark_name, problems = load_benchmark_problems(args)
        print(f"Loaded {len(problems):,} problems from benchmark: {benchmark_name}")

    problems = select_problems(problems, args.start, args.take, args.max_file_size)
    safe_benchmark_name = validate_path_component(benchmark_name, "benchmark or dataset name")
    output_dir = args.output_root / safe_benchmark_name / builder_name
    output_file = output_dir / "prompts.jsonl"

    if output_file.exists() and not args.force:
        print(f"Prompts file already exists: {output_file}")
        print("Use --force to overwrite.")
        return

    reusable_prompts = load_reusable_prompts(args.reuse_prompts)
    print(f"Building prompts with {builder_name}: {len(problems):,} selected, {args.workers} worker(s)")
    results, reused = build_prompts(
        problems,
        args.context_builder,
        builder_args,
        workers=args.workers,
        chunk_size=args.chunk_size,
        training=args.training,
        reusable_prompts=reusable_prompts,
    )

    results.sort(key=lambda result: result[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0
    with output_file.open("w", encoding="utf-8") as file:
        for _, line, error in results:
            if error:
                errors += 1
            if line is not None:
                file.write(line + "\n")
                written += 1

    reuse_message = f", {reused} reused" if reused else ""
    print(f"Saved {written:,} prompts to {output_file}{reuse_message}")
    if errors:
        print(f"{errors:,} problems failed.")


if __name__ == "__main__":
    main()
