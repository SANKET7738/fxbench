from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Problem:
    workbook_path: Path
    worksheet_name: str
    active_cell: str
    formula: str
    id: str = ""
    prefix: str = "="
    annotation: dict | None = None
