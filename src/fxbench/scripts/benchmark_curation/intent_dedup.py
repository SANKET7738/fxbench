from __future__ import annotations

import argparse
from pathlib import Path

from fxbench.benchmark_curation.intent_dedup import dedupe_by_intent
from fxbench.benchmark_curation.pipeline_paths import DEFAULT_RECORDS_DIR, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    return parser.parse_args()


def load_annotation_records(records_dir: Path) -> list[dict]:
    return read_jsonl(records_dir / "annotations_v2.jsonl")


def write_deduped_records(records_dir: Path, records: list[dict]) -> None:
    write_jsonl(records_dir / "deduped_by_intent.jsonl", records)


def main() -> None:
    args = parse_args()
    records = load_annotation_records(args.records_dir)
    deduped_records, dropped = dedupe_by_intent(records)
    write_deduped_records(args.records_dir, deduped_records)

    print("Intent dedup completed.")
    print(f"Usable records: {len(records) - dropped:,}")
    print(f"Dropped records: {dropped:,}")
    print(f"Records before deduplication: {len(records) - dropped:,}")
    print(f"Records after deduplication: {len(deduped_records):,}")
    print(f"Output: {args.records_dir / 'deduped_by_intent.jsonl'}")


if __name__ == "__main__":
    main()
