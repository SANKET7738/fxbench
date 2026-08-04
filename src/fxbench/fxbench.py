from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from fxbench.data_models import Benchmark, BenchmarkSpec, Problem


DEFAULT_REPO_ID = "FormulaComletion/fxbench"
DEFAULT_SPLIT = "train"
DEFAULT_WORKBOOK_DIR = Path("Assets") / "workbooks"
REQUIRED_COLUMNS = {
    "id",
    "function",
    "formula",
    "target_cell",
    "target_sheet",
    "workbook",
}


def safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "fxbench"


def workbook_bytes(value: Any, row_id: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, dict):
        bytes_value = value.get("bytes")
        if isinstance(bytes_value, (bytes, bytearray, memoryview)):
            return workbook_bytes(bytes_value, row_id)
    raise TypeError(f"Expected workbook bytes for row {row_id}, got {type(value).__name__}")


def materialize_workbook(workbook_dir: Path, row_id: str, workbook_data: bytes) -> Path:
    workbook_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = workbook_dir / f"{safe_path_component(row_id)}.xlsx"
    if not workbook_path.exists() or workbook_path.read_bytes() != workbook_data:
        workbook_path.write_bytes(workbook_data)
    return workbook_path


def load_hf_dataset(
    repo_id: str,
    *,
    split: str,
    cache_dir: Path | None,
    revision: str | None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Loading FxBench requires the 'datasets' package.") from error

    kwargs: dict[str, Any] = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(repo_id, **kwargs)


def problem_from_row(row: dict[str, Any], row_index: int, workbook_dir: Path) -> Problem:
    missing_columns = REQUIRED_COLUMNS - set(row)
    if missing_columns:
        raise ValueError(f"FxBench row {row_index} is missing required columns: {sorted(missing_columns)}")

    row_id = str(row["id"])
    workbook_path = materialize_workbook(workbook_dir, row_id, workbook_bytes(row["workbook"], row_id))
    function_name = str(row["function"])

    return Problem(
        workbook_path=workbook_path,
        worksheet_name=str(row["target_sheet"]),
        active_cell=str(row["target_cell"]),
        formula=str(row["formula"]),
        id=row_id,
        annotation={"function": function_name},
    )


class FxBench(Benchmark):
    repo_id: str = DEFAULT_REPO_ID
    split: str = DEFAULT_SPLIT
    revision: str | None = None
    workbook_cache_dir: Path = Field(default_factory=lambda: DEFAULT_WORKBOOK_DIR)

    @classmethod
    def load(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        split: str = DEFAULT_SPLIT,
        cache_dir: str | Path | None = None,
        workbook_cache_dir: str | Path | None = None,
        revision: str | None = None,
    ) -> FxBench:
        resolved_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        resolved_workbook_dir = (
            Path(workbook_cache_dir).expanduser().resolve()
            if workbook_cache_dir is not None
            else DEFAULT_WORKBOOK_DIR.expanduser().resolve()
        )
        dataset = load_hf_dataset(
            repo_id,
            split=split,
            cache_dir=resolved_cache_dir,
            revision=revision,
        )
        problems = [
            problem_from_row(dict(row), row_index, resolved_workbook_dir)
            for row_index, row in enumerate(dataset)
        ]

        return cls(
            problems=problems,
            spec=BenchmarkSpec(version=1),
            repo_id=repo_id,
            split=split,
            revision=revision,
            workbook_cache_dir=resolved_workbook_dir,
        )
