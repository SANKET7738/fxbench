from __future__ import annotations

import argparse
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fxbench.benchmark_curation.pipeline_paths import DEFAULT_RECORDS_DIR, DEFAULT_WORKBOOK_DIR, read_jsonl, write_json, write_jsonl
from fxbench.snapshot_generator import create_dense_representation, generate_snapshot


DEFAULT_SNAPSHOT_DIR = Path("Assets") / "Snapshots" / "benchmark"
DEFAULT_INPUT_NAME = "top100_selected.jsonl"
DEFAULT_OUTPUT_NAME = "final_records.jsonl"
MAX_DENSE_CHARS = 200_000

_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--workbook-dir", type=Path, default=DEFAULT_WORKBOOK_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--take", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--max-dense-chars", type=int, default=MAX_DENSE_CHARS)
    return parser.parse_args()


def safe_record_id(record_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", record_id)


def path_for_record(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def record_with_original_workbook_path(record: dict) -> dict:
    updated_record = dict(record)
    if "original_wb_path" not in updated_record and "wb_local_path" in updated_record:
        updated_record["original_wb_path"] = updated_record["wb_local_path"]
    return updated_record


def source_workbook_value(record: dict) -> str:
    value = record.get("original_wb_path") or record.get("wb_local_path")
    if not value:
        raise FileNotFoundError("Record has neither original_wb_path nor wb_local_path")
    return str(value)


def resolve_workbook_path(record: dict, workbook_dir: Path) -> Path:
    source_value = source_workbook_value(record)
    source_path = Path(source_value)

    candidates = []
    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.extend(
            [
                Path.cwd() / source_path,
                workbook_dir / source_path,
                workbook_dir / source_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Workbook not found for {source_value}. Checked: {checked_paths}")


def dense_representation_too_large(workbook_path: Path, sheet_name: str, max_dense_chars: int) -> tuple[bool, int]:
    dense_representation = create_dense_representation(workbook_path, sheet_name)
    dense_length = len(dense_representation)
    return dense_length > max_dense_chars, dense_length


def status_for_exception(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, FileNotFoundError):
        return "workbook_not_found"
    if "worksheet" in message and "not found" in message:
        return "sheet_not_found"
    if "sheet" in message and "not found" in message:
        return "sheet_not_found"
    if "segmentation" in message and "json" in message:
        return "segmentation_parse_error"
    if "snapshot cleaning" in message and "json" in message:
        return "cleaning_parse_error"
    if "regions_to_delete" in message:
        return "cleaning_parse_error"
    return f"snapshot_error:{type(error).__name__}"


def snapshot_destination(record: dict, snapshot_dir: Path) -> Path:
    return snapshot_dir / f"{safe_record_id(str(record['id']))}.xlsx"


def mark_snapshot_record(record: dict, destination_path: Path) -> dict:
    updated_record = record_with_original_workbook_path(record)
    updated_record["wb_local_path"] = path_for_record(destination_path)
    return updated_record


def generate_snapshot_for_record(record: dict, args: argparse.Namespace) -> tuple[dict, str, dict | None]:
    updated_record = record_with_original_workbook_path(record)
    destination_path = snapshot_destination(updated_record, args.snapshot_dir)

    if destination_path.exists() and not args.force:
        return mark_snapshot_record(updated_record, destination_path), "skipped_existing", None

    try:
        workbook_path = resolve_workbook_path(updated_record, args.workbook_dir)
        too_large, dense_length = dense_representation_too_large(workbook_path, updated_record["sheet"], args.max_dense_chars)
        if too_large:
            detail = f"sheet too large: {dense_length:,} chars (limit {args.max_dense_chars:,})"
            return updated_record, "too_large", {"id": updated_record["id"], "reason": "too_large", "detail": detail}

        output_dir = args.snapshot_dir / "_intermediate" / safe_record_id(str(updated_record["id"]))
        cleaned_path = generate_snapshot(
            filepath=workbook_path,
            sheet_name=updated_record["sheet"],
            active_cell=updated_record["cell"],
            output_dir=output_dir,
            model=args.model,
        )

        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cleaned_path, destination_path)
        return mark_snapshot_record(updated_record, destination_path), "ok", None
    except Exception as error:
        status = status_for_exception(error)
        return updated_record, status, {"id": updated_record.get("id"), "reason": status, "detail": str(error)}


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def status_label(status: str) -> str:
    if status == "ok":
        return "OK"
    if status == "skipped_existing":
        return "SKIP (exists)"
    if status == "too_large":
        return "SKIP (too large)"
    if status == "workbook_not_found":
        return "FAIL (workbook not found)"
    if status == "sheet_not_found":
        return "FAIL (sheet not found)"
    return f"FAIL ({status})"


def worker(index: int, total: int, record: dict, args: argparse.Namespace) -> tuple[int, dict, str, dict | None]:
    updated_record, status, manifest_entry = generate_snapshot_for_record(record, args)
    log(f"[{index:3d}/{total}] {str(record.get('id', ''))[:70]}... {status_label(status)}")
    return index, updated_record, status, manifest_entry


def selected_record_indexes(records: list[dict], take: int | None) -> list[int]:
    limit = len(records) if take is None else min(take, len(records))
    return list(range(limit))


def process_records(records: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict[str, int], list[dict]]:
    output_records = list(records)
    summary = {"ok": 0, "skipped_existing": 0, "too_large": 0, "failed": 0}
    manifest_entries = []
    indexes = selected_record_indexes(records, args.take)
    total = len(indexes)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(worker, index + 1, total, records[index], args): index
                for index in indexes
            }
            for future in as_completed(futures):
                index, updated_record, status, manifest_entry = future.result()
                output_records[index - 1] = updated_record
                update_summary(summary, status)
                if manifest_entry is not None:
                    manifest_entries.append(manifest_entry)
    else:
        for position, record_index in enumerate(indexes, start=1):
            _, updated_record, status, manifest_entry = worker(position, total, records[record_index], args)
            output_records[record_index] = updated_record
            update_summary(summary, status)
            if manifest_entry is not None:
                manifest_entries.append(manifest_entry)
            if position < total and args.delay > 0:
                time.sleep(args.delay)

    return output_records, summary, manifest_entries


def update_summary(summary: dict[str, int], status: str) -> None:
    if status in summary:
        summary[status] += 1
    else:
        summary["failed"] += 1


def write_skipped_manifest(snapshot_dir: Path, entries: list[dict]) -> Path | None:
    if not entries:
        return None
    manifest_path = snapshot_dir / "skipped_records.json"
    write_json(manifest_path, entries)
    return manifest_path


def main() -> None:
    args = parse_args()
    input_path = args.records_dir / args.input_name
    output_path = args.records_dir / args.output_name
    records = read_jsonl(input_path)
    output_records, summary, manifest_entries = process_records(records, args)
    write_jsonl(output_path, output_records)
    manifest_path = write_skipped_manifest(args.snapshot_dir, manifest_entries)

    print()
    print("Snapshot generation completed.")
    print(f"Records loaded: {len(records):,}")
    print(f"Records processed: {len(selected_record_indexes(records, args.take)):,}")
    print(f"OK: {summary['ok']:,}")
    print(f"Skipped existing: {summary['skipped_existing']:,}")
    print(f"Too large: {summary['too_large']:,}")
    print(f"Failed: {summary['failed']:,}")
    print(f"Output: {output_path}")
    if manifest_path is not None:
        print(f"Skipped manifest: {manifest_path}")


if __name__ == "__main__":
    main()
