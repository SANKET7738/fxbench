from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from fxbench.benchmark_curation.pipeline_paths import (
    DEFAULT_RECORDS_DIR,
    DEFAULT_WORKBOOK_DIR,
    append_jsonl,
    read_json,
    read_jsonl,
    write_jsonl,
)
from fxbench.benchmark_curation.programmatic_annotations import (
    fallback_annotations_for_sheet_items,
    process_workbook,
    select_records_by_diversity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--workbook-dir", type=Path, default=DEFAULT_WORKBOOK_DIR)
    return parser.parse_args()


def load_done_ids(annotation_path: Path) -> set[str]:
    if not annotation_path.exists():
        return set()
    return {row["id"] for row in read_jsonl(annotation_path)}


def load_selected_work_items(records_dir: Path, done_ids: set[str]) -> tuple[list[dict], int]:
    selected_sketches = read_json(records_dir / "selected_sketches.json")
    sketch_map = read_json(records_dir / "sketch_map_deduped.json")

    work_items = []
    skipped_done = 0
    for function_name, signature_map in selected_sketches.items():
        function_sketch_map = sketch_map.get(function_name, {})
        for signature, sketches in signature_map.items():
            for sketch in sketches:
                for record in function_sketch_map.get(sketch, []):
                    if record["id"] in done_ids:
                        skipped_done += 1
                        continue
                    work_items.append(
                        {
                            "function": function_name,
                            "signature": signature,
                            "sketch": sketch,
                            "id": record["id"],
                            "formula": record["formula"],
                            "cell": record["cell"],
                            "sheet": record["sheet"],
                            "wb_local_path": record["wb_local_path"],
                        }
                    )
    return work_items, skipped_done


def resolve_workbook_path(workbook_path: str, workbook_dir: Path) -> str:
    path = Path(workbook_path)
    if path.is_absolute():
        return str(path)
    if path.parent == DEFAULT_WORKBOOK_DIR:
        return str(workbook_dir / path.name)
    if path.parent == Path("."):
        return str(workbook_dir / path.name)
    return str(path)


def group_work_items_by_workbook(
    work_items: list[dict],
    workbook_dir: Path,
) -> dict[str, list[tuple[str, list[dict]]]]:
    workbook_groups = defaultdict(list)
    workbook_sheet_indexes = {}

    for index, item in enumerate(work_items):
        workbook_path = resolve_workbook_path(item["wb_local_path"], workbook_dir)
        workbook_sheet_key = (workbook_path, item["sheet"])
        if workbook_sheet_key not in workbook_sheet_indexes:
            workbook_sheet_indexes[workbook_sheet_key] = len(workbook_groups[workbook_path])
            workbook_groups[workbook_path].append((item["sheet"], []))

        sheet_index = workbook_sheet_indexes[workbook_sheet_key]
        workbook_groups[workbook_path][sheet_index][1].append(
            {
                "idx": index,
                "cell": item["cell"],
                "formula": item["formula"],
            }
        )

    return dict(workbook_groups)


def annotate_work_items(
    work_items: list[dict],
    workbook_dir: Path,
    workers: int,
) -> tuple[dict[int, dict], Counter, int, int]:
    workbook_groups = group_work_items_by_workbook(work_items, workbook_dir)
    if not workbook_groups:
        return {}, Counter(), 0, 0

    available_workers = max(1, (os.cpu_count() or 2) - 1)
    worker_count = max(1, min(workers, available_workers, len(workbook_groups)))
    annotations_by_index = {}
    fallback_counts = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Annotating workbooks", total=len(workbook_groups))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_group = {
                executor.submit(process_workbook, workbook_path, sheet_items): (workbook_path, sheet_items)
                for workbook_path, sheet_items in workbook_groups.items()
            }
            for future in as_completed(future_to_group):
                _, sheet_items = future_to_group[future]
                progress.update(task_id, advance=1)
                try:
                    workbook_annotations = future.result()
                except Exception as error:
                    fallback_reason = f"worker_error:{type(error).__name__}"
                    workbook_annotations = fallback_annotations_for_sheet_items(sheet_items, fallback_reason)

                for annotation in workbook_annotations:
                    annotations_by_index[annotation["idx"]] = annotation
                    fallback_reason = annotation.get("fallback_reason")
                    if fallback_reason:
                        fallback_counts[fallback_reason] += 1

    return annotations_by_index, fallback_counts, len(workbook_groups), worker_count


def normalized_formula(formula: str) -> str:
    if formula.startswith("="):
        return formula
    return f"={formula}"


def build_annotation_rows(work_items: list[dict], annotations_by_index: dict[int, dict]) -> list[dict]:
    rows = []
    missing_indexes = []

    for index, item in enumerate(work_items):
        annotation = annotations_by_index.get(index)
        if annotation is None:
            missing_indexes.append(index)
            continue
        rows.append(
            {
                "id": item["id"],
                "function": item["function"],
                "sketch": item["sketch"],
                "signature": item["signature"],
                "formula": normalized_formula(item["formula"]),
                "cell": item["cell"],
                "sheet": item["sheet"],
                "wb_local_path": item["wb_local_path"],
                "location": annotation["location"],
                "formula_ref_types": annotation["formula_ref_types"],
                "sheet_size": annotation["sheet_size"],
            }
        )

    if missing_indexes:
        raise RuntimeError(f"Missing annotations for {len(missing_indexes)} work items")

    return rows


def write_selected_records(records_dir: Path) -> tuple[int, int]:
    annotation_path = records_dir / "programmatic_annotations.jsonl"
    selected_records_path = records_dir / "selected_records.jsonl"

    annotated_records = read_jsonl(annotation_path)
    selected_records = select_records_by_diversity(annotated_records)
    write_jsonl(selected_records_path, selected_records)
    return len(annotated_records), len(selected_records)


def main() -> None:
    args = parse_args()
    records_dir = args.records_dir
    workbook_dir = args.workbook_dir
    annotation_path = records_dir / "programmatic_annotations.jsonl"

    records_dir.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(annotation_path)
    print(f"Resuming: {len(done_ids):,} records already annotated")

    work_items, skipped_done = load_selected_work_items(records_dir, done_ids)
    print(f"Work items: {len(work_items):,}")
    print(f"Skipped already annotated: {skipped_done:,}")

    if work_items:
        annotations_by_index, fallback_counts, workbook_count, worker_count = annotate_work_items(
            work_items,
            workbook_dir,
            args.workers,
        )
        annotation_rows = build_annotation_rows(work_items, annotations_by_index)
        append_jsonl(annotation_path, annotation_rows)
        print(f"Workbooks processed: {workbook_count:,}")
        print(f"Workers used: {worker_count:,}")
        print(f"Annotations written: {len(annotation_rows):,}")

        fallback_total = sum(fallback_counts.values())
        print(f"Fallback annotations: {fallback_total:,}")
        for reason, count in sorted(fallback_counts.items()):
            print(f"  {reason}: {count:,}")
    elif not annotation_path.exists():
        write_jsonl(annotation_path, [])
        print("Annotations written: 0")

    annotated_count, selected_count = write_selected_records(records_dir)
    reduction = (1 - selected_count / annotated_count) * 100 if annotated_count else 0

    print("Programmatic annotations completed.")
    print(f"Records before diversity selection: {annotated_count:,}")
    print(f"Records after diversity selection: {selected_count:,}")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Output: {records_dir / 'programmatic_annotations.jsonl'}")
    print(f"Output: {records_dir / 'selected_records.jsonl'}")


if __name__ == "__main__":
    main()
