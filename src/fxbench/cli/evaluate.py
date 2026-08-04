from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from fxbench.cli.build_prompts import get_benchmark, get_context_builder
from fxbench.data_models import Problem, SavedEvaluation, SavedPrediction
from fxbench.evaluation import ExecutionEvaluator, ExcelPool, JsonParser, Outcome, OutcomeKind, aggregate
from fxbench.utils import validate_path_component

DEFAULT_RESULTS_ROOT = Path("Assets") / "Results"
DEFAULT_EVALUATIONS_ROOT = Path("Assets") / "Evaluations"
DEFAULT_RAW_WORKBOOKS_ROOT = Path("Assets") / "raw_workbooks"


def canonical_context_builder_name(name: str) -> str:
    return get_context_builder(name).__name__


def predictions_file_path(results_root: Path, benchmark: str, context_builder: str, model: str) -> Path:
    return (
        results_root
        / validate_path_component(benchmark, "benchmark")
        / canonical_context_builder_name(context_builder)
        / validate_path_component(model, "model")
        / "predictions.jsonl"
    )


def output_dir_path(evaluations_root: Path, benchmark: str, context_builder: str, model: str) -> Path:
    return (
        evaluations_root
        / validate_path_component(benchmark, "benchmark")
        / canonical_context_builder_name(context_builder)
        / validate_path_component(model, "model")
    )


def load_predictions(path: Path) -> list[SavedPrediction]:
    predictions = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                predictions.append(SavedPrediction.model_validate_json(stripped))
    return predictions


def load_done_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "id" in record:
                done.add(str(record["id"]))
    return done


def load_benchmark_problems(benchmark_name: str) -> dict[str, Problem]:
    benchmark_class = get_benchmark(benchmark_name)
    benchmark = benchmark_class.load()
    return {problem.id: problem for problem in benchmark.problems}


def copy_needed_workbooks(
    predictions: list[SavedPrediction],
    problems_by_id: dict[str, Problem],
    raw_workbook_dir: Path,
) -> int:
    raw_workbook_dir.mkdir(parents=True, exist_ok=True)
    needed_workbooks = {
        Path(problem.workbook_path)
        for prediction in predictions
        if (problem := problems_by_id.get(prediction.id)) is not None
    }
    copied = 0
    for source in needed_workbooks:
        destination = raw_workbook_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
            copied += 1
    return copied


def formula_for_prediction(prediction: SavedPrediction, parser: JsonParser) -> str | None:
    if prediction.answer is not None:
        return prediction.answer
    if prediction.raw_response:
        return parser.parse(prediction.raw_response).formula
    return None


def evaluate_prediction(
    prediction: SavedPrediction,
    problem: Problem,
    *,
    evaluator: ExecutionEvaluator,
    parser: JsonParser,
    raw_workbook_dir: Path,
) -> Outcome:
    formula = formula_for_prediction(prediction, parser)
    if formula is None:
        return Outcome(
            outcome=OutcomeKind.Error,
            metadata={"message": "No formula in prediction"},
        )

    source_workbook = Path(problem.workbook_path)
    evaluation_workbook = raw_workbook_dir / source_workbook.name
    try:
        return evaluator.evaluate(problem, formula, workbook_override=evaluation_workbook)
    except Exception as error:
        return Outcome(
            outcome=OutcomeKind.Error,
            metadata={"message": str(error)},
        )


def saved_evaluation(prediction_id: str, outcome: Outcome) -> SavedEvaluation:
    return SavedEvaluation(
        id=prediction_id,
        outcome=outcome.outcome.value,
        metrics=[metric.model_dump() for metric in outcome.metrics],
        metadata=outcome.metadata or {},
    )


def append_evaluation(path: Path, evaluation: SavedEvaluation) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(evaluation.model_dump_json() + "\n")


def run_evaluations(
    predictions: list[SavedPrediction],
    *,
    problems_by_id: dict[str, Problem],
    evaluator: ExecutionEvaluator,
    parser: JsonParser,
    raw_workbook_dir: Path,
    output_file: Path,
    console: Console,
) -> tuple[list[Outcome], int]:
    outcomes = []
    errors = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(predictions))
        for prediction in predictions:
            problem = problems_by_id.get(prediction.id)
            if problem is None:
                progress.console.print(f"[yellow]WARNING: Problem not found for {prediction.id}, skipping.[/]")
                progress.advance(task)
                continue

            outcome = evaluate_prediction(
                prediction,
                problem,
                evaluator=evaluator,
                parser=parser,
                raw_workbook_dir=raw_workbook_dir,
            )
            if outcome.outcome == OutcomeKind.Error:
                errors += 1
                message = outcome.metadata.get("message", "") if outcome.metadata else ""
                if message and message != "No formula in prediction":
                    progress.console.print(f"[red]WARNING: {prediction.id}: {message}[/]")

            outcomes.append(outcome)
            append_evaluation(output_file, saved_evaluation(prediction.id, outcome))
            progress.advance(task)

    return outcomes, errors


def write_summary(output_dir: Path, outcomes: list[Outcome]) -> Path | None:
    if not outcomes:
        return None
    summary_file = output_dir / "summary.json"
    summary_file.write_text(aggregate(outcomes).model_dump_json(indent=2), encoding="utf-8")
    return summary_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FxBench predictions by executing formulas in Excel workbooks.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name, e.g. FxBench.")
    parser.add_argument("--context-builder", required=True, help="Context builder class name or alias.")
    parser.add_argument("--model", required=True, help="Model folder name used by get_predictions.")
    parser.add_argument("--take", type=int, default=None, help="Evaluate only the first N pending predictions.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing evaluations instead of resuming.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--evaluations-root", type=Path, default=DEFAULT_EVALUATIONS_ROOT)
    parser.add_argument("--raw-workbooks-root", type=Path, default=DEFAULT_RAW_WORKBOOKS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.take is not None and args.take < 0:
        raise ValueError("--take must be >= 0")

    console = Console()
    ExcelPool.setup(size=1, console=console)

    predictions_file = predictions_file_path(args.results_root, args.benchmark, args.context_builder, args.model)
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions not found at {predictions_file}")

    predictions = load_predictions(predictions_file)
    console.print(f"Loaded {len(predictions):,} predictions from {predictions_file}")

    problems_by_id = load_benchmark_problems(args.benchmark)

    output_dir = output_dir_path(args.evaluations_root, args.benchmark, args.context_builder, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "evaluation.jsonl"
    if args.force and output_file.exists():
        output_file.unlink()

    done_ids = load_done_ids(output_file) if not args.force else set()
    todo = [prediction for prediction in predictions if prediction.id not in done_ids]
    if args.take is not None:
        todo = todo[: args.take]

    if not todo:
        console.print("Nothing to do - all predictions already have evaluations.")
        return

    if done_ids:
        console.print(f"Skipping {len(done_ids):,} already evaluated predictions.")

    raw_workbook_dir = args.raw_workbooks_root / validate_path_component(args.benchmark, "benchmark")
    copied = copy_needed_workbooks(todo, problems_by_id, raw_workbook_dir)
    if copied:
        console.print(f"Copied {copied:,} new workbooks to {raw_workbook_dir}")

    console.print(f"Evaluating {len(todo):,} predictions")
    outcomes, errors = run_evaluations(
        todo,
        problems_by_id=problems_by_id,
        evaluator=ExecutionEvaluator(),
        parser=JsonParser(),
        raw_workbook_dir=raw_workbook_dir,
        output_file=output_file,
        console=console,
    )

    summary_file = write_summary(output_dir, outcomes)
    if summary_file is None:
        console.print("No evaluations produced.")
        return

    metrics = aggregate(outcomes)
    console.print()
    console.print("[bold]Evaluation Summary[/]")
    console.print(f"    TP (correct formula):     {metrics.tp}")
    console.print(f"    FP (wrong / false alarm): {metrics.fp}")
    console.print(f"    TN (correct abstention):  {metrics.tn}")
    console.print(f"    FN (missed formula):      {metrics.fn}")
    console.print(f"    Errors:                   {metrics.error}")
    console.print(f"    Overall accuracy (TP+TN): {metrics.accuracy}")
    for name, value in metrics.others.items():
        console.print(f"    {name}: {value}")
    console.print(f"\nSaved evaluations to {output_file}")
    console.print(f"Saved summary to {summary_file}")
    if errors:
        console.print(f"{errors:,} predictions failed during evaluation.")


if __name__ == "__main__":
    main()
