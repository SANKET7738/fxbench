from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from fxbench.cli.build_prompts import get_context_builder
from fxbench.data_models import SavedPrediction
from fxbench.utils import validate_path_component

DEFAULT_RESULTS_ROOT = Path("Assets") / "Results"

_FORMULA_JSON_RE = re.compile(r'\{[^{}]*?"formula"\s*:\s*"((?:\\.|[^"\\])*)"[^{}]*?\}', re.DOTALL)
_CODE_BLOCK_FORMULA_RE = re.compile(r"```(?:excel|formula)?\s*\n?(=[^\n`]+)", re.IGNORECASE)


def extract_formula(raw_response: str | None) -> str | None:
    if not raw_response:
        return None

    clean = raw_response.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        if len(lines) > 2 and lines[-1].strip().startswith("```"):
            clean = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "formula" in data:
            return data["formula"]
    except (json.JSONDecodeError, TypeError):
        pass

    matches = _FORMULA_JSON_RE.findall(raw_response)
    if matches:
        try:
            return json.loads(f'"{matches[-1]}"')
        except json.JSONDecodeError:
            return matches[-1]

    code_match = _CODE_BLOCK_FORMULA_RE.search(raw_response)
    if code_match:
        return code_match.group(1).strip()

    return None


_extract_formula = extract_formula


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest external prediction results into FxBench result assets.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name, e.g. FxBench.")
    parser.add_argument("--context-builder", required=True, help="Context builder class name or alias.")
    parser.add_argument("--model", required=True, help="Model name for the output folder.")
    parser.add_argument("--path", required=True, type=Path, help="Path to an external JSON or JSONL predictions file.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing predictions.jsonl file.")
    return parser.parse_args()


def load_external_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        return load_external_jsonl(path), {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        results = data
        parameters = {}
    elif isinstance(data, dict):
        raw_results = data.get("results")
        if raw_results is None:
            raise ValueError("Input JSON object must contain a 'results' list")
        if not isinstance(raw_results, list):
            raise ValueError("Input field 'results' must be a list")
        results = raw_results
        raw_parameters = data.get("parameters", {})
        parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
    else:
        raise ValueError("Input must be a JSON object or list")

    if not results:
        raise ValueError("No results found in input file")
    return [normalize_result_record(record, path, index) for index, record in enumerate(results, start=1)], parameters


def load_external_jsonl(path: Path) -> list[dict[str, Any]]:
    results = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            results.append(normalize_result_record(record, path, line_number))

    if not results:
        raise ValueError("No results found in input file")
    return results


def normalize_result_record(record: Any, path: Path, row_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Result row {row_number} in {path} must be a JSON object")
    if "id" not in record:
        raise ValueError(f"Result row {row_number} in {path} is missing 'id'")
    return record


def raw_response_for_result(result: dict[str, Any]) -> str | None:
    predictions = result.get("predictions")
    if isinstance(predictions, list) and predictions:
        raw_response = predictions[0]
    else:
        raw_response = result.get("prediction", result.get("response"))
    if raw_response is None or isinstance(raw_response, str):
        return raw_response
    return json.dumps(raw_response)


def prediction_from_result(
    result: dict[str, Any],
    *,
    source_file: Path,
    source_parameters: dict[str, Any],
) -> SavedPrediction:
    raw_response = raw_response_for_result(result)
    usage = result.get("usage", {})
    return SavedPrediction(
        id=str(result["id"]),
        raw_response=raw_response,
        answer=extract_formula(raw_response),
        metadata={
            "source": "ingested",
            "source_file": source_file.name,
            "source_parameters": source_parameters,
            "tokens": usage if isinstance(usage, dict) else {},
        },
    )


def write_predictions(
    results: list[dict[str, Any]],
    *,
    source_file: Path,
    source_parameters: dict[str, Any],
    output_file: Path,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_file.open("w", encoding="utf-8") as file:
        for result in results:
            prediction = prediction_from_result(
                result,
                source_file=source_file,
                source_parameters=source_parameters,
            )
            file.write(prediction.model_dump_json() + "\n")
            written += 1
    return written


def main() -> None:
    args = parse_args()
    results, source_parameters = load_external_results(args.path)

    output_file = predictions_file_path(args.results_root, args.benchmark, args.context_builder, args.model)
    if output_file.exists() and not args.force:
        raise FileExistsError(f"Output file already exists: {output_file}. Use --force to overwrite.")

    written = write_predictions(
        results,
        source_file=args.path,
        source_parameters=source_parameters,
        output_file=output_file,
    )
    print(f"Saved {written:,} predictions to {output_file}")


if __name__ == "__main__":
    main()
