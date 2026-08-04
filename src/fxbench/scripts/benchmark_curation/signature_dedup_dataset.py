import json
import re
from collections import defaultdict
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


INPUT_DIR = Path("Assets/records/valid_records")
INPUT_PATH = INPUT_DIR / "sketch_map_deduped.json"
OUTPUT_PATH = INPUT_DIR / "selected_sketches.json"

Signature = tuple[tuple[str, ...], tuple[str, ...], int]


def extract_functions(sketch: str) -> list[str]:
    """Extract ordered function chain from a sketch string."""
    functions = []
    s = sketch
    i = 1 if s.startswith("=") else 0
    while i < len(s):
        if s[i].isupper():
            j = i
            while j < len(s) and (s[j].isupper() or s[j].isdigit() or s[j] in "._"):
                j += 1
            word = s[i:j]
            if j < len(s) and s[j] == "(":
                functions.append(word)
            i = j
        else:
            i += 1
    return functions


def _is_value_like(ch: str) -> bool:
    return ch in (")", ">") or ch.isalnum() or ch in "_."


def extract_operators(sketch: str, include_colon: bool = False) -> set[str]:
    operators = set()
    s = sketch
    i = 1 if s.startswith("=") else 0
    while i < len(s):
        two = s[i : i + 2] if i + 1 < len(s) else ""
        if two in (">=", "<=", "<>"):
            operators.add(two)
            i += 2
        elif s[i] in ("+", "*", "/", ">", "<", "&"):
            if s[i] == "<":
                m = re.match(r"<[a-z]+>", s[i:])
                if m:
                    i += m.end()
                    continue
            if s[i] == ">":
                if i > 0 and s[i - 1].islower():
                    i += 1
                    continue
            operators.add(s[i])
            i += 1
        elif s[i] == "=":
            k = i - 1
            while k >= 0 and s[k] == " ":
                k -= 1
            if k >= 1 and _is_value_like(s[k]):
                operators.add("=")
            i += 1
        elif s[i] == "-":
            k = i - 1
            while k >= 0 and s[k] == " ":
                k -= 1
            if k >= 0 and _is_value_like(s[k]):
                operators.add("-")
            i += 1
        elif s[i] == ":" and include_colon:
            operators.add(":")
            i += 1
        else:
            i += 1
    return operators


def nesting_depth(sketch: str) -> int:
    max_depth = 0
    current = 0
    for ch in sketch:
        if ch == "(":
            current += 1
            max_depth = max(max_depth, current)
        elif ch == ")":
            current -= 1
    return max_depth


def extract_signature(sketch: str) -> Signature:
    """Functions + operator groups (arith/cmp/concat) + depth capped at 3."""
    s = sketch.upper()
    funcs = tuple(dict.fromkeys(extract_functions(s)))[:3]
    ops = extract_operators(s, include_colon=False)
    groups = []
    if ops & {"+", "-", "*", "/"}:
        groups.append("arith")
    if ops & {">", "<", "=", ">=", "<=", "<>"}:
        groups.append("cmp")
    if "&" in ops:
        groups.append("concat")
    depth = min(nesting_depth(s), 3)
    return (funcs, tuple(sorted(groups)), depth)


def signature_to_str(sig: Signature) -> str:
    funcs, ops, depth = sig
    return f"{','.join(funcs)}|{','.join(ops)}|{depth}"


def select_representatives(sketches: list[str], sketch_record_counts: dict[str, int]) -> list[str]:
    if len(sketches) == 1:
        return sketches[:]

    n_reps = 2 if len(sketches) <= 5 else 3
    by_freq = sorted(sketches, key=lambda s: -sketch_record_counts.get(s, 0))
    by_len = sorted(sketches, key=len)

    selected = [by_freq[0]]

    if by_len[0] != selected[0]:
        selected.append(by_len[0])
    else:
        for sketch in by_freq[1:]:
            if sketch not in selected:
                selected.append(sketch)
                break

    if n_reps >= 3 and len(selected) < 3:
        has_range = any("<range>" in sketch for sketch in selected)
        if not has_range:
            range_candidates = [
                sketch for sketch in sketches if "<range>" in sketch and sketch not in selected
            ]
            if range_candidates:
                selected.append(range_candidates[0])
        if len(selected) < 3:
            remaining = [sketch for sketch in by_len if sketch not in selected]
            if remaining:
                mid = len(remaining) // 2
                selected.append(remaining[mid])

    return selected


def main() -> None:
    print(f"Loading {INPUT_PATH} ...", flush=True)
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_functions = sorted(data.keys())
    print(f"  Total functions: {len(all_functions):,}")

    output: dict[str, dict[str, list[str]]] = {}
    grand_sketches_before = 0
    grand_sketches_after = 0
    grand_sigs = 0
    selected_records = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Selecting signature representatives", total=len(all_functions))

        for func_name in all_functions:
            func_map = data[func_name]
            all_sketches = list(func_map.keys())
            sketch_record_counts = {
                sketch: len(records) for sketch, records in func_map.items()
            }
            n_before = len(all_sketches)

            sig_to_sketches: dict[Signature, list[str]] = defaultdict(list)
            for sketch in all_sketches:
                sig = extract_signature(sketch)
                sig_to_sketches[sig].append(sketch)

            selected_map: dict[str, list[str]] = {}
            n_after = 0
            for sig, sketches in sig_to_sketches.items():
                reps = select_representatives(sketches, sketch_record_counts)
                selected_map[signature_to_str(sig)] = reps
                n_after += len(reps)
                selected_records += sum(sketch_record_counts[rep] for rep in reps)

            output[func_name] = selected_map
            grand_sketches_before += n_before
            grand_sketches_after += n_after
            grand_sigs += len(sig_to_sketches)
            progress.advance(task)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    reduction = (
        (1 - grand_sketches_after / grand_sketches_before) * 100
        if grand_sketches_before > 0
        else 0
    )

    print("Signature dedup completed.")
    print(f"Functions: {len(all_functions):,}")
    print(f"Signatures: {grand_sigs:,}")
    print(f"Sketches before deduplication: {grand_sketches_before:,}")
    print(f"Sketches after deduplication: {grand_sketches_after:,}")
    print(f"Selected records: {selected_records:,}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
