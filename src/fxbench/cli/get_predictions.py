from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from fxbench.cli.build_prompts import get_context_builder
from fxbench.data_models import SavedPrediction, SavedPrompt
from fxbench.llm_utils import LLMClient
from fxbench.utils import validate_path_component

DEFAULT_PROMPTS_ROOT = Path("Assets") / "Prompts"
DEFAULT_RESULTS_ROOT = Path("Assets") / "Results"


def load_model_aliases() -> dict[str, str]:
    raw = os.environ.get("MODEL_ALIASES", "{}")
    try:
        aliases = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(aliases, dict):
        return {}
    return {str(alias).lower(): str(deployment) for alias, deployment in aliases.items()}


def deployment_name(model_alias: str) -> str:
    return load_model_aliases().get(model_alias.lower(), model_alias)


def model_aliases_for_help() -> str:
    aliases = sorted(load_model_aliases())
    return ", ".join(aliases) if aliases else "none configured"


def load_prompts(path: Path) -> list[SavedPrompt]:
    prompts = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                prompts.append(SavedPrompt.model_validate_json(stripped))
    return prompts


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


def canonical_context_builder_name(name: str) -> str:
    return get_context_builder(name).__name__


def prompts_file_path(prompts_root: Path, benchmark: str, context_builder: str) -> Path:
    return prompts_root / validate_path_component(benchmark, "benchmark") / canonical_context_builder_name(context_builder) / "prompts.jsonl"


def predictions_file_path(results_root: Path, benchmark: str, context_builder: str, model_alias: str) -> Path:
    return (
        results_root
        / validate_path_component(benchmark, "benchmark")
        / canonical_context_builder_name(context_builder)
        / validate_path_component(model_alias, "model")
        / "predictions.jsonl"
    )


def build_prompts_if_missing(prompts_file: Path, benchmark: str, context_builder: str, prompts_root: Path) -> None:
    if prompts_file.exists():
        return
    command = [
        sys.executable,
        "-m",
        "fxbench.cli.build_prompts",
        "--benchmark",
        benchmark,
        "--context-builder",
        context_builder,
        "--output-root",
        str(prompts_root),
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"build_prompts failed with exit code {result.returncode}")
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file still missing after build: {prompts_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference over pre-built FxBench prompts.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name, e.g. FxBench.")
    parser.add_argument("--context-builder", required=True, help="Context builder class name or alias.")
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model alias or deployment name. Aliases: {model_aliases_for_help()}",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel LLM calls.")
    parser.add_argument("--take", type=int, default=None, help="Process only the first N pending prompts.")
    parser.add_argument("--prompts-root", type=Path, default=DEFAULT_PROMPTS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing predictions instead of resuming.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4000)
    return parser.parse_args()


def result_raw_response(result: dict) -> str | None:
    raw_response = result.get("response")
    if raw_response is None and result.get("raw_json"):
        raw_response = result["raw_json"].get("formula")
    if raw_response is None and result.get("validated_response"):
        raw_response = str(result["validated_response"])
    if raw_response is not None and not isinstance(raw_response, str):
        raw_response = json.dumps(raw_response)
    return raw_response


def run_predictions(
    prompts: list[SavedPrompt],
    *,
    client: LLMClient,
    model: str,
    temperature: float,
    max_tokens: int,
    workers: int,
    output_file: Path,
) -> int:
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    write_lock = threading.Lock()

    def append_result(prediction: SavedPrediction) -> None:
        line = prediction.model_dump_json()
        with write_lock:
            with output_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def process(prompt: SavedPrompt) -> SavedPrediction:
        try:
            result = client.prompt_llm(
                model=model,
                message_history=prompt.messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_response = result_raw_response(result)
            prediction = SavedPrediction(
                id=prompt.id,
                raw_response=raw_response,
                answer=extract_formula(raw_response),
                metadata=result.get("metadata", {}),
            )
        except Exception as error:
            prediction = SavedPrediction(
                id=prompt.id,
                metadata={"error": str(error)},
            )
        append_result(prediction)
        return prediction

    errors = 0
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Predicting", total=len(prompts))
        if workers == 1:
            for prompt in prompts:
                prediction = process(prompt)
                if prediction.metadata.get("error"):
                    errors += 1
                    progress.console.print(f"[red]WARNING: {prompt.id}: {prediction.metadata['error']}[/]")
                progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process, prompt): prompt for prompt in prompts}
                for future in as_completed(futures):
                    prediction = future.result()
                    prompt = futures[future]
                    if prediction.metadata.get("error"):
                        errors += 1
                        progress.console.print(f"[red]WARNING: {prompt.id}: {prediction.metadata['error']}[/]")
                    progress.advance(task)
    return errors


def main() -> None:
    args = parse_args()
    model_alias = args.model
    model = deployment_name(model_alias)

    prompts_file = prompts_file_path(args.prompts_root, args.benchmark, args.context_builder)
    build_prompts_if_missing(prompts_file, args.benchmark, args.context_builder, args.prompts_root)

    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts):,} prompts from {prompts_file}")

    output_file = predictions_file_path(args.results_root, args.benchmark, args.context_builder, model_alias)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output_file.exists():
        output_file.unlink()

    done_ids = load_done_ids(output_file) if not args.force else set()
    todo = [prompt for prompt in prompts if prompt.id not in done_ids]
    if args.take is not None:
        if args.take < 0:
            raise ValueError("--take must be >= 0")
        todo = todo[: args.take]

    if not todo:
        print("Nothing to do - all prompts already have predictions.")
        return

    if done_ids:
        print(f"Skipping {len(done_ids):,} already completed prompts.")

    client = LLMClient()
    print(f"Running inference on {len(todo):,} prompts with {model} ({args.workers} worker(s))")
    errors = run_predictions(
        todo,
        client=client,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
        output_file=output_file,
    )

    print(f"Saved {len(todo) - errors:,}/{len(todo):,} predictions to {output_file}")
    if errors:
        print(f"{errors:,} prompts failed.")


if __name__ == "__main__":
    main()
