from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

from fxbench.benchmark_curation.llm_annotations import (
    ERROR_PATTERNS,
    RateLimiter,
    call_llm_for_annotation,
    extract_markdown_tables_from_workbook,
    normalized_formula,
)
from fxbench.llm_utils import LLMClient
from fxbench.benchmark_curation.pipeline_paths import (
    DEFAULT_RECORDS_DIR,
    DEFAULT_WORKBOOK_DIR,
    read_json,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--rpm", type=int, default=60)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--workbook-dir", type=Path, default=DEFAULT_WORKBOOK_DIR)
    return parser.parse_args()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["id"] for record in read_jsonl(path)}


def load_work_items(records_dir: Path, done_ids: set[str]) -> tuple[list[dict], int, int]:
    records = read_jsonl(records_dir / "selected_records.jsonl")
    work_items = []
    skipped_done = 0
    skipped_errors = 0

    for record in records:
        if record["id"] in done_ids:
            skipped_done += 1
            continue
        if ERROR_PATTERNS.search(record["formula"]):
            skipped_errors += 1
            continue
        work_items.append(record)

    return work_items, skipped_done, skipped_errors


def resolve_workbook_path(workbook_path: str, workbook_dir: Path) -> str:
    path = Path(workbook_path)
    if path.is_absolute():
        return str(path)
    if path.parent == DEFAULT_WORKBOOK_DIR:
        return str(workbook_dir / path.name)
    if path.parent == Path("."):
        return str(workbook_dir / path.name)
    return str(path)


def group_items_by_workbook(
    work_items: list[dict],
    indexes: list[int],
    workbook_dir: Path,
) -> dict[str, list[tuple[str, list[dict]]]]:
    workbook_groups = defaultdict(list)
    workbook_sheet_indexes = {}

    for index in indexes:
        item = work_items[index]
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
            }
        )

    return dict(workbook_groups)


def load_dense_cache(path: Path, work_items: list[dict]) -> tuple[dict[int, str | None], list[int], dict[str, str | None]]:
    dense_cache = read_json(path) if path.exists() else {}
    dense_tables = {}
    missing_indexes = []

    for index, item in enumerate(work_items):
        if item["id"] in dense_cache:
            dense_tables[index] = dense_cache[item["id"]]
        else:
            dense_tables[index] = None
            missing_indexes.append(index)

    return dense_tables, missing_indexes, dense_cache


def build_missing_dense_tables(
    work_items: list[dict],
    missing_indexes: list[int],
    workbook_dir: Path,
    workers: int,
) -> tuple[dict[int, str | None], Counter]:
    if not missing_indexes:
        return {}, Counter()

    workbook_groups = group_items_by_workbook(work_items, missing_indexes, workbook_dir)
    available_workers = max(1, (os.cpu_count() or 2) - 1)
    worker_count = max(1, min(workers, available_workers, len(workbook_groups)))
    dense_tables = {}
    stats = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Extracting context", total=len(missing_indexes))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_group = {
                executor.submit(extract_markdown_tables_from_workbook, workbook_path, sheet_items): sheet_items
                for workbook_path, sheet_items in workbook_groups.items()
            }
            for future in as_completed(future_to_group):
                sheet_items = future_to_group[future]
                try:
                    results = future.result()
                except Exception as error:
                    results = [
                        {"idx": cell["idx"], "md_table": None, "reason": f"worker_error:{type(error).__name__}"}
                        for _, cells in sheet_items
                        for cell in cells
                    ]

                for result in results:
                    dense_tables[result["idx"]] = result["md_table"]
                    if result["md_table"] is None:
                        stats[result.get("reason", "context_missing")] += 1
                    progress.update(task_id, advance=1)

    return dense_tables, stats


def write_dense_cache(path: Path, work_items: list[dict], dense_tables: dict[int, str | None]) -> None:
    dense_cache = {
        item["id"]: dense_tables.get(index)
        for index, item in enumerate(work_items)
    }
    write_json(path, dense_cache)


def prepared_items(work_items: list[dict], dense_tables: dict[int, str | None]) -> list[tuple[dict, str]]:
    items = []
    for index, item in enumerate(work_items):
        markdown_table = dense_tables.get(index)
        if markdown_table is not None:
            items.append((item, markdown_table))
    return items


def annotation_output_row(record: dict, annotation: dict) -> dict:
    return {
        "id": record["id"],
        "function": record["function"],
        "sketch": record["sketch"],
        "signature": record["signature"],
        "formula": normalized_formula(record["formula"]),
        "cell": record["cell"],
        "sheet": record["sheet"],
        "wb_local_path": record["wb_local_path"],
        "annotation": annotation,
    }


def annotate_prepared_items(
    items: list[tuple[dict, str]],
    models: list[str],
    workers_per_model: int,
    rpm: int,
    output_path: Path,
) -> Counter:
    if not items:
        return Counter()

    client = LLMClient()
    write_lock = threading.Lock()
    blocked_workbooks = set()
    rate_limiters = {model: RateLimiter(rpm) for model in models}
    model_by_index = [models[index % len(models)] for index in range(len(items))]
    worker_count = max(1, workers_per_model * len(models))
    stats = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def annotate_one(index: int, item: tuple[dict, str], output_file) -> None:
        record, markdown_table = item
        model = model_by_index[index]
        workbook_path = record["wb_local_path"]

        with write_lock:
            if workbook_path in blocked_workbooks:
                stats["content_filter"] += 1
                return

        annotation = None
        for attempt in range(3):
            rate_limiters[model].wait()
            annotation = call_llm_for_annotation(
                client,
                model,
                markdown_table,
                record["cell"],
                record["formula"],
            )
            if annotation.get("skipped") and annotation.get("reason") == "rate_limit":
                time.sleep(min(10 * (3 ** attempt), 60))
                continue
            break

        if annotation is None:
            with write_lock:
                stats["missing_annotation"] += 1
            return

        if annotation.get("skipped"):
            reason = annotation.get("reason", "skipped")
            with write_lock:
                if reason == "content_filter":
                    blocked_workbooks.add(workbook_path)
                stats[reason] += 1
            return

        with write_lock:
            row = annotation_output_row(record, annotation)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["done"] += 1

    with output_path.open("a", encoding="utf-8") as output_file:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Annotating", total=len(items))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(annotate_one, index, item, output_file): index
                    for index, item in enumerate(items)
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        with write_lock:
                            stats[f"llm_error:{type(error).__name__}"] += 1
                    progress.update(task_id, advance=1)

    return stats


def main() -> None:
    args = parse_args()
    records_dir = args.records_dir
    output_path = records_dir / "annotations_v2.jsonl"
    dense_cache_path = records_dir / "dense_cache_v2.json"

    records_dir.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)
    work_items, skipped_done, skipped_errors = load_work_items(records_dir, done_ids)

    print(f"Resuming: {len(done_ids):,} records already annotated")
    print(f"Work items: {len(work_items):,}")
    print(f"Skipped already annotated: {skipped_done:,}")
    print(f"Skipped Excel errors: {skipped_errors:,}")

    dense_tables, missing_indexes, _ = load_dense_cache(dense_cache_path, work_items)
    print(f"Dense cache hits: {len(work_items) - len(missing_indexes):,}")
    print(f"Dense cache misses: {len(missing_indexes):,}")

    missing_dense_tables, context_stats = build_missing_dense_tables(
        work_items,
        missing_indexes,
        args.workbook_dir,
        args.workers,
    )
    dense_tables.update(missing_dense_tables)
    write_dense_cache(dense_cache_path, work_items, dense_tables)

    items = prepared_items(work_items, dense_tables)
    print(f"Prepared for LLM: {len(items):,}")
    for reason, count in sorted(context_stats.items()):
        print(f"{reason}: {count:,}")

    stats = annotate_prepared_items(items, args.models, args.workers, args.rpm, output_path)
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    print("LLM annotations completed.")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value:,}")
    print(f"Output: {output_path}")
    print(f"Output: {dense_cache_path}")


if __name__ == "__main__":
    main()
