from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import openpyxl
from datasets import load_dataset
from tqdm import tqdm

from fxbench.benchmark_curation.filters import (
    file_level_filters,
    is_sheet_formulas_valid,
    sheet_level_filters,
)

HF_DATASET = "tianzl66/Sheetpedia_xlsx"
OUT_DIR = Path("Assets/records/valid_records")
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_file_level(dataset) -> list[dict]:
    valid: list[dict] = []
    for i in tqdm(range(len(dataset)), desc="file-level"):
        rec = dataset[i]
        xlsx = rec.get("xlsx")
        if not xlsx:
            continue
        ok, _, metadata = file_level_filters(xlsx)
        if ok:
            valid.append({"index": i, "key": rec["__key__"], "file_metadata": metadata})
    return valid


def run_sheet_level(dataset, file_valid: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for entry in tqdm(file_valid, desc="sheet-level"):
        sheets = sheet_level_filters(dataset[entry["index"]]["xlsx"])
        if sheets:
            valid.append({**entry, "valid_sheets": sheets})
    return valid


def _function_worker(xlsx_bytes: bytes, sheet_names: list[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    try:
        wb = openpyxl.load_workbook(BytesIO(xlsx_bytes), read_only=True, data_only=False)
    except Exception:
        return {n: False for n in sheet_names}
    for name in sheet_names:
        if name not in wb.sheetnames:
            out[name] = False
            continue
        try:
            ok, _ = is_sheet_formulas_valid(wb[name])
        except Exception:
            ok = False
        out[name] = ok
    wb.close()
    return out


def run_function_level(dataset, sheet_valid: list[dict]) -> list[dict]:
    valid: list[dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for entry in sheet_valid:
            try:
                xlsx_bytes = dataset[entry["index"]]["xlsx"]
            except Exception:
                continue
            names = [s["sheet_name"] for s in entry["valid_sheets"]]
            futures[executor.submit(_function_worker, xlsx_bytes, names)] = entry

        for fut in tqdm(as_completed(futures), total=len(futures), desc="function-level"):
            entry = futures[fut]
            try:
                results = fut.result()
            except Exception:
                continue
            kept = [s for s in entry["valid_sheets"] if results.get(s["sheet_name"])]
            if kept:
                valid.append({**entry, "valid_sheets": kept})
    return valid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET)["train"]
    print(f"  {len(ds):,} workbooks")

    file_valid = run_file_level(ds)
    write_jsonl(OUT_DIR / "file_level.jsonl", file_valid)
    print(f"  file-level valid:     {len(file_valid):,}")

    sheet_valid = run_sheet_level(ds, file_valid)
    write_jsonl(OUT_DIR / "sheet_level.jsonl", sheet_valid)
    sheet_count = sum(len(r["valid_sheets"]) for r in sheet_valid)
    print(f"  sheet-level valid:    {len(sheet_valid):,} workbooks, {sheet_count:,} sheets")

    func_valid = run_function_level(ds, sheet_valid)
    write_jsonl(OUT_DIR / "function_level.jsonl", func_valid)
    func_sheet_count = sum(len(r["valid_sheets"]) for r in func_valid)
    print(f"  function-level valid: {len(func_valid):,} workbooks, {func_sheet_count:,} sheets")


if __name__ == "__main__":
    main()
