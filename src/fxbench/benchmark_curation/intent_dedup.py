from __future__ import annotations

import random
from collections import defaultdict


INTENT_SELECTION_SEED = 42
INTENT_K_BY_SOURCE = {
    "col_header": 3,
    "row_label": 3,
    "adjacent_formula": 3,
    "combination": 3,
    "adjacent_values": 5,
    "no_hint": 5,
}
INTENT_K_DEFAULT = 3


def get_intent_source(annotation: dict) -> str | None:
    if annotation.get("parse_error") or annotation.get("skipped"):
        return None
    intent_hint = annotation.get("intent_hint", {})
    if isinstance(intent_hint, dict):
        return intent_hint.get("source")
    return None


def select_diverse_workbooks(records: list[dict], limit: int, rng: random.Random) -> list[dict]:
    if len(records) <= limit:
        return records

    records_by_workbook = defaultdict(list)
    for record in records:
        records_by_workbook[record["wb_local_path"]].append(record)

    selected = []
    selected_ids = set()
    workbook_paths = list(records_by_workbook.keys())
    rng.shuffle(workbook_paths)

    for workbook_path in workbook_paths:
        if len(selected) >= limit:
            break
        record = rng.choice(records_by_workbook[workbook_path])
        selected.append(record)
        selected_ids.add(record["id"])

    if len(selected) < limit:
        remaining = [record for record in records if record["id"] not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])

    return selected


def group_records_by_intent(records: list[dict]) -> tuple[dict[tuple[str, str], list[dict]], int]:
    groups = defaultdict(list)
    dropped = 0

    for record in records:
        source = get_intent_source(record.get("annotation", {}))
        if source is None:
            dropped += 1
            continue
        groups[(record["signature"], source)].append(record)

    return dict(groups), dropped


def dedupe_by_intent(
    records: list[dict],
    seed: int = INTENT_SELECTION_SEED,
) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    groups, dropped = group_records_by_intent(records)
    deduped = []

    for (_, source), group_records in sorted(groups.items()):
        limit = INTENT_K_BY_SOURCE.get(source, INTENT_K_DEFAULT)
        deduped.extend(select_diverse_workbooks(group_records, limit, rng))

    return deduped, dropped
