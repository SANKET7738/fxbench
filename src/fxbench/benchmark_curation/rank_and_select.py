from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from fxbench.benchmark_curation.intent_dedup import get_intent_source


RANK_SELECTION_SEED = 42
MAX_PER_FUNCTION = 100

_RE_FUNC_CALL = re.compile(r"[A-Z][A-Z0-9_.]*\(")


def nesting_depth(sketch: str) -> int:
    return len(_RE_FUNC_CALL.findall(sketch))


def count_if_occurrences(sketch: str) -> int:
    return len(re.findall(r"\bIF\(", sketch))


def if_subcategory(sketch: str) -> str:
    if_count = count_if_occurrences(sketch)
    if if_count >= 3:
        return "IF-nested-d3plus"
    if if_count == 2:
        return "IF-nested-d2"

    inner_sketch = sketch[3:]
    other_functions = [match for match in _RE_FUNC_CALL.findall(inner_sketch) if match != "IF("]
    if other_functions:
        return "IF-function"

    return "IF-simple"


def expression_subcategory(formula: str, sketch: str) -> str:
    if "$" in formula or "!" in formula:
        return "EXPR-reference"

    operator_count = sum(1 for character in sketch if character in "+-*/")
    if operator_count > 1 or "(" in sketch:
        return "EXPR-multi"

    return "EXPR-simple"


def allocate_quotas(pools: dict[str, list[dict]], total_quota: int, floor: int = 10) -> dict[str, int]:
    total_pool = sum(len(records) for records in pools.values())
    if total_pool == 0:
        return {key: 0 for key in pools}

    quotas = {}
    for key, records in pools.items():
        quotas[key] = max(floor, round(total_quota * len(records) / total_pool))

    diff = sum(quotas.values()) - total_quota
    if diff > 0:
        for key in sorted(quotas, key=lambda item: -quotas[item]):
            trim = min(diff, quotas[key] - floor)
            quotas[key] -= trim
            diff -= trim
            if diff == 0:
                break
    elif diff < 0:
        for key in sorted(quotas, key=lambda item: quotas[item]):
            quotas[key] += 1
            diff += 1
            if diff == 0:
                break

    return quotas


def rank_intent_source(record: dict) -> str:
    source = get_intent_source(record.get("annotation", {}))
    if source is None:
        return "no_hint"
    return source


def prepare_rank_record(record: dict) -> dict:
    rank_record = dict(record)
    rank_record["_intent_source"] = rank_intent_source(record)
    rank_record["_nesting_depth"] = nesting_depth(record["sketch"])
    return rank_record


def greedy_select(pool: list[dict], quota: int, rng: random.Random) -> list[dict]:
    if len(pool) <= quota:
        return list(pool)

    shuffled_pool = list(pool)
    rng.shuffle(shuffled_pool)

    selected = []
    selected_signatures = Counter()
    selected_intents = Counter()
    selected_workbooks = set()
    remaining_indexes = set(range(len(shuffled_pool)))

    for _ in range(quota):
        if not remaining_indexes:
            break

        best_score = -1.0
        best_index = -1

        for index in remaining_indexes:
            record = shuffled_pool[index]
            signature = record["signature"]
            intent = record["_intent_source"]
            workbook_path = record["wb_local_path"]
            depth = record["_nesting_depth"]

            signature_bonus = 10.0 if signature not in selected_signatures else 1.0 / (1.0 + selected_signatures[signature])
            intent_bonus = 5.0 / (1.0 + selected_intents[intent])
            workbook_bonus = 3.0 if workbook_path not in selected_workbooks else 0.0
            complexity_bonus = min(depth, 10) * 0.1
            score = signature_bonus + intent_bonus + workbook_bonus + complexity_bonus

            if score > best_score:
                best_score = score
                best_index = index

        selected_record = shuffled_pool[best_index]
        selected.append(selected_record)
        selected_signatures[selected_record["signature"]] += 1
        selected_intents[selected_record["_intent_source"]] += 1
        selected_workbooks.add(selected_record["wb_local_path"])
        remaining_indexes.remove(best_index)

    return selected


def select_top_records(
    records: list[dict],
    max_per_function: int = MAX_PER_FUNCTION,
    seed: int = RANK_SELECTION_SEED,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    rng = random.Random(seed)
    rank_records = [prepare_rank_record(record) for record in records]

    records_by_function = defaultdict(list)
    for record in rank_records:
        records_by_function[record["function"]].append(record)

    selected_records = []
    stats_before = {}
    stats_after = {}

    for function_name in sorted(records_by_function.keys()):
        function_records = records_by_function[function_name]
        stats_before[function_name] = len(function_records)

        if len(function_records) <= max_per_function:
            selected_records.extend(function_records)
            stats_after[function_name] = len(function_records)
            continue

        if function_name == "IF":
            sub_pools = defaultdict(list)
            for record in function_records:
                sub_pools[if_subcategory(record["sketch"])].append(record)
            quotas = allocate_quotas(sub_pools, max_per_function, floor=10)
            selected_for_function = []
            for subcategory, sub_pool in sub_pools.items():
                selected_for_function.extend(greedy_select(sub_pool, quotas[subcategory], rng))

        elif function_name == "EXPRESSION":
            sub_pools = defaultdict(list)
            for record in function_records:
                sub_pools[expression_subcategory(record["formula"], record["sketch"])].append(record)
            quotas = allocate_quotas(sub_pools, max_per_function, floor=10)
            selected_for_function = []
            for subcategory, sub_pool in sub_pools.items():
                selected_for_function.extend(greedy_select(sub_pool, quotas[subcategory], rng))

        else:
            selected_for_function = greedy_select(function_records, max_per_function, rng)

        selected_records.extend(selected_for_function)
        stats_after[function_name] = len(selected_for_function)

    return selected_records, stats_before, stats_after


def strip_internal_fields(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def summarize_selection(
    records_before: list[dict],
    records_after: list[dict],
    stats_before: dict[str, int],
    stats_after: dict[str, int],
    max_per_function: int = MAX_PER_FUNCTION,
) -> dict:
    total_before = sum(stats_before.values())
    total_after = sum(stats_after.values())
    reduction_percent = 0.0
    if total_before:
        reduction_percent = (1 - total_after / total_before) * 100

    after_counts = list(stats_after.values())
    records_per_function_distribution = []
    for label, low, high in [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-50", 11, 50), ("51-100", 51, 100)]:
        function_count = sum(1 for count in after_counts if low <= count <= high)
        record_count = sum(count for count in after_counts if low <= count <= high)
        records_per_function_distribution.append(
            {
                "label": label,
                "function_count": function_count,
                "record_count": record_count,
            }
        )

    capped_functions = [
        (function_name, stats_before[function_name], stats_after[function_name])
        for function_name in stats_before
        if stats_before[function_name] > max_per_function
    ]

    return {
        "total_before": total_before,
        "total_after": total_after,
        "reduction": total_before - total_after,
        "reduction_percent": reduction_percent,
        "function_count": len(stats_after),
        "unique_signatures": len({record["signature"] for record in records_after}),
        "unique_sketches": len({record["sketch"] for record in records_after}),
        "unique_workbooks": len({record["wb_local_path"] for record in records_after}),
        "intent_distribution": Counter(record.get("_intent_source", rank_intent_source(record)) for record in records_after),
        "capped_functions": sorted(capped_functions, key=lambda item: -item[1]),
        "records_per_function_distribution": records_per_function_distribution,
    }
