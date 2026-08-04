"""Intent Cells Annotator — LLM-as-judge for formula intent classification.

For each FxBench problem, uses the ground-truth formula to:
1. Extract cell/range references
2. Collect surrounding cells (+/-width) for context
3. Prompt an LLM to classify intent cells into:
   - direct_intent: cells describing what the OUTPUT represents
   - data_description: cells describing the INPUT data
   - data_range: range notations from the formula
   - intent: concise NL description of what the formula does

Problems are loaded from the published FxBench benchmark via ``FxBench.load`` and the
annotations are written to ``Assets/Annotations/FxBench/intent_cells.jsonl`` (one JSON
object per line), with resume support.

Usage:
    uv run python -m fxbench.scripts.intent_cells_annotator --dry-run --take 2
    uv run python -m fxbench.scripts.intent_cells_annotator --workers 8 --model your-alias
    uv run python -m fxbench.scripts.intent_cells_annotator --split test --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from fxbench.data_models import Problem
from fxbench.fxbench import FxBench
from fxbench.llm_utils import LLMClient

load_dotenv()

DEFAULT_OUTPUT = Path("Assets") / "Annotations" / "FxBench" / "intent_cells.jsonl"

# ---------------------------------------------------------------------------
# Cell reference extraction
# ---------------------------------------------------------------------------

_CELL_REF_PATTERN = re.compile(
    r"(?:'[^']*'!|\w+!)?"
    r"(\$?[A-Z]{1,3}\$?\d+"
    r"(?::\$?[A-Z]{1,3}\$?\d+)?)"
)


def _normalize_ref(ref: str) -> str:
    return ref.replace("$", "")


def _extract_range_notations(formula: str) -> list[str]:
    """Extract range notations (e.g., A1:B10) from formula, ignoring cross-sheet."""
    ranges = []
    for m in _CELL_REF_PATTERN.finditer(formula):
        full_match = m.group(0)
        if "!" in full_match:
            continue
        ref = _normalize_ref(m.group(1))
        if ":" in ref:
            ranges.append(ref)
    return ranges


def _extract_cell_refs(formula: str) -> set[str]:
    """Extract individual cell refs (not ranges) from formula."""
    cells: set[str] = set()
    for m in _CELL_REF_PATTERN.finditer(formula):
        full_match = m.group(0)
        if "!" in full_match:
            continue
        ref = _normalize_ref(m.group(1))
        if ":" not in ref:
            cells.add(ref)
    return cells


# ---------------------------------------------------------------------------
# Surrounding cells
# ---------------------------------------------------------------------------

def _surrounding_coords(row: int, col: int, width: int = 4) -> list[tuple[int, int]]:
    """Get cell coordinates within +/-width of (row, col)."""
    coords = []
    for r in range(max(1, row - width), row + width + 1):
        for c in range(max(1, col - width), col + width + 1):
            coords.append((r, c))
    return coords


def _get_surrounding_cells(formula: str, target_cell: str, width: int = 4) -> list[str]:
    """Get addresses of cells surrounding formula refs + target cell."""
    coords: set[tuple[int, int]] = set()

    # Surrounding target cell
    t_row, t_col = coordinate_to_tuple(target_cell)
    coords.update(_surrounding_coords(t_row, t_col, width))

    # Surrounding individual cell refs
    for ref in _extract_cell_refs(formula):
        try:
            r, c = coordinate_to_tuple(ref)
            coords.update(_surrounding_coords(r, c, width))
        except (ValueError, TypeError):
            continue

    # Surrounding range corners
    for rng in _extract_range_notations(formula):
        start, end = rng.split(":")
        try:
            r1, c1 = coordinate_to_tuple(start)
            r2, c2 = coordinate_to_tuple(end)
        except (ValueError, TypeError):
            continue
        corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
        for row, col in corners:
            coords.update(_surrounding_coords(row, col, width))

    return sorted(
        [f"{get_column_letter(c)}{r}" for r, c in coords],
        key=lambda a: (coordinate_to_tuple(a)[0], coordinate_to_tuple(a)[1]),
    )


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _build_context(ws, surrounding_addresses: list[str]) -> str:
    """Read cell values for surrounding addresses, format as 'ADDRESS | value'."""
    lines = []
    for address in surrounding_addresses:
        try:
            r, c = coordinate_to_tuple(address)
            cell = ws.cell(row=r, column=c)
            if cell.value is not None:
                val = str(cell.value).replace("\n", " ").strip()
                if len(val) > 32:
                    val = val[:29] + "..."
                lines.append(f"{address} | {val}")
        except Exception:
            continue
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert at identifying cells in Excel that help understand formula intent.

Given a formula, its target cell, and surrounding cell context, identify three types of cells and the formula intent:

1. **Direct Intent**: Cells describing what the OUTPUT/RESULT represents
   - Column header above the target cell (where the formula lives)
   - Row labels next to the target cell
   - Focus: What is being CALCULATED or PRODUCED
   - Example: If target is G6 with header "Percent Rank" at G2, return ["G2"]

2. **Data Description**: Cells describing the INPUT data being used
   - Column headers for ranges/cells referenced IN the formula
   - Row labels for data being consumed
   - Focus: What data is being READ or CONSUMED
   - Example: For =F6/SUM(F6:F23), if F2 is "Percent Male", return ["F2"]

3. **Data Range**: Full range notations used in the formula
   - Use colon notation: "F6:F23" not separate boundaries
   - Include all ranges found in the formula
   - Example: For SUM(F6:F23), return ["F6:F23"]

4. **Intent**: A concise description (at most 10 words) of what the formula calculates.
   - Example: "Total sales for east region"

Return cell addresses as a JSON object with four fields.
Example response format:
```json
{
  "direct_intent": ["G2"],
  "data_description": ["F2"],
  "data_range": ["F6:F23"],
  "intent": "Percentage of male respondents in the survey"
}
```

Important guidelines:
- Return actual cell addresses (e.g., "G2", "F6:F23"), not descriptions
- Direct intent describes the target cell's meaning (OUTPUT)
- Data description describes the referenced data's meaning (INPUT)
- Data range uses colon notation for ranges
- Use the context to identify relevant headers and labels
- Consider the target cell's position relative to headers
- You need to return all categories, use empty lists if nothing applies
""".strip()


def _build_user_prompt(target_cell: str, formula: str, context: str, data_ranges: list[str]) -> str:
    return f"""# Target Cell
{target_cell}

# Formula
{formula}

# Context
The following cells surround the formula target and referenced cells:

{context}

# Data Ranges
The formula operates on these ranges: {', '.join(data_ranges) if data_ranges else 'None (individual cell references only)'}""".strip()


# ---------------------------------------------------------------------------
# Annotation worker
# ---------------------------------------------------------------------------

def _annotate_one(
    problem: Problem,
    client: LLMClient,
    deployment: str | None,
    width: int = 4,
) -> dict | None:
    """Annotate a single FxBench problem. Returns annotation dict or None on error."""
    formula = problem.formula
    target_cell = problem.active_cell

    if not formula or formula.strip() == "":
        return {"id": problem.id, "intent_cells": None, "reason": "empty_formula"}

    try:
        wb = load_workbook(str(problem.workbook_path), read_only=False, data_only=False)
        ws = wb[problem.worksheet_name]
    except Exception as e:
        return {"id": problem.id, "intent_cells": None, "reason": f"workbook_error: {e}"}

    try:
        surrounding = _get_surrounding_cells(formula, target_cell, width=width)
        context = _build_context(ws, surrounding)
        data_ranges = _extract_range_notations(formula)

        user_prompt = _build_user_prompt(target_cell, formula, context, data_ranges)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result = client.prompt_llm(
            model=deployment,
            message_history=messages,
            temperature=0.0,
            max_tokens=500,
        )

        raw = result.get("response", "")
        try:
            clean = client.extract_json_from_response(raw)
            parsed = json.loads(clean)
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw_response": raw, "parse_error": True}

        tokens = result.get("metadata", {}).get("tokens", {})
        return {
            "id": problem.id,
            "intent_cells": {
                "direct_intent": parsed.get("direct_intent", []),
                "data_description": parsed.get("data_description", []),
                "data_range": parsed.get("data_range", []),
                "intent": parsed.get("intent", ""),
            },
            "metadata": {
                "prompt_tokens": tokens.get("prompt_tokens"),
                "completion_tokens": tokens.get("completion_tokens"),
            },
        }
    except Exception as e:
        return {"id": problem.id, "intent_cells": None, "reason": f"error: {e}"}
    finally:
        wb.close()


# ---------------------------------------------------------------------------
# Model alias resolution
# ---------------------------------------------------------------------------

def _load_model_aliases() -> dict[str, str]:
    raw = os.environ.get("MODEL_ALIASES", "{}")
    try:
        aliases = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(aliases, dict):
        return {}
    return {str(alias).lower(): str(deployment) for alias, deployment in aliases.items()}


def _resolve_deployment(model: str | None) -> str | None:
    if not model:
        return os.getenv("DEPLOYMENT_NAME")
    return _load_model_aliases().get(model.lower(), model)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _load_done_ids(output_file: Path) -> set[str]:
    done: set[str] = set()
    if not output_file.exists():
        return done
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    done.add(rec["id"])
            except json.JSONDecodeError:
                continue
    return done


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _dry_run(problems: list[Problem], width: int) -> None:
    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)
    for problem in problems:
        print("\n" + "=" * 70)
        print(f"id: {problem.id}")
        print(f"workbook: {problem.workbook_path}")
        print(f"sheet: {problem.worksheet_name}  target: {problem.active_cell}")
        if not problem.formula or problem.formula.strip() == "":
            print("(empty formula - would be skipped)")
            continue
        try:
            wb = load_workbook(str(problem.workbook_path), read_only=False, data_only=False)
            ws = wb[problem.worksheet_name]
        except Exception as e:
            print(f"(workbook_error: {e})")
            continue
        try:
            surrounding = _get_surrounding_cells(problem.formula, problem.active_cell, width=width)
            context = _build_context(ws, surrounding)
            data_ranges = _extract_range_notations(problem.formula)
            print("\n--- USER PROMPT ---")
            print(_build_user_prompt(problem.active_cell, problem.formula, context, data_ranges))
        finally:
            wb.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Intent cells annotation via LLM-as-judge")
    parser.add_argument("--take", type=int, default=0, help="Limit to first N problems (0 = all)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel LLM calls")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model alias or deployment name (resolved via MODEL_ALIASES; "
        "falls back to DEPLOYMENT_NAME)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotations")
    parser.add_argument("--width", type=int, default=4, help="Surrounding cell width (default: 4)")
    parser.add_argument("--split", type=str, default="train", help="Benchmark split (default: train)")
    parser.add_argument("--repo-id", type=str, default=None, help="Override the FxBench HF repo id")
    parser.add_argument("--revision", type=str, default=None, help="Optional dataset revision")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--dry-run", action="store_true", help="Build and print prompts without calling the LLM")
    args = parser.parse_args()

    # Load problems from the published benchmark.
    load_kwargs: dict = {"split": args.split}
    if args.repo_id is not None:
        load_kwargs["repo_id"] = args.repo_id
    if args.revision is not None:
        load_kwargs["revision"] = args.revision
    benchmark = FxBench.load(**load_kwargs)
    problems = benchmark.problems
    if args.take:
        problems = problems[: args.take]
    print(f"Loaded {len(problems)} FxBench problems (split={args.split})")

    if args.dry_run:
        _dry_run(problems, args.width)
        return

    deployment = _resolve_deployment(args.model)
    if not deployment:
        parser.error("No model resolved: pass --model or set DEPLOYMENT_NAME in the environment.")

    output_file = args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.force and output_file.exists():
        output_file.unlink()
        done_ids: set[str] = set()
    else:
        done_ids = _load_done_ids(output_file)

    todo = [p for p in problems if p.id not in done_ids]
    print(f"  Already done: {len(done_ids)}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    client = LLMClient()
    write_lock = threading.Lock()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    errors = 0

    with progress:
        task_id = progress.add_task("Annotating", total=len(todo))

        def _process(problem: Problem) -> dict | None:
            return _annotate_one(problem, client, deployment, width=args.width)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process, p): p for p in todo}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    with write_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if result.get("reason"):
                        errors += 1
                progress.advance(task_id)

    total_done = len(_load_done_ids(output_file))
    print(f"\nAnnotations saved to {output_file}")
    print(f"    Total: {total_done}, Errors this run: {errors}")


if __name__ == "__main__":
    main()
