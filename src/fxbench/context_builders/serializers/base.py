from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import openpyxl

from fxbench.data_models import Problem


class IContextBuilder(ABC):
    _read_only: bool = True

    @abstractmethod
    def build(self, problem: Problem) -> str:
        ...

    def _open_worksheet(self, problem: Problem):
        workbook_path = Path(problem.workbook_path)
        if not workbook_path.exists():
            raise FileNotFoundError(f"Workbook not found: {workbook_path}")

        workbook = openpyxl.load_workbook(
            workbook_path,
            read_only=self._read_only,
            data_only=False,
        )
        if problem.worksheet_name not in workbook.sheetnames:
            workbook.close()
            raise ValueError(f"Sheet '{problem.worksheet_name}' not in {workbook_path.name}")

        return workbook, workbook[problem.worksheet_name]
