from __future__ import annotations

import csv
import io

from fxbench.context_builders.serializers._utils import cell_display, col_letter, parse_cell_address
from fxbench.context_builders.serializers.base import IContextBuilder
from fxbench.data_models import Problem


class CsvContextBuilder(IContextBuilder):
    def build(self, problem: Problem) -> str:
        workbook, worksheet = self._open_worksheet(problem)
        try:
            return self._build_csv(worksheet, problem.active_cell)
        finally:
            workbook.close()

    def _build_csv(self, worksheet, target_cell: str) -> str:
        target_column, target_row = parse_cell_address(target_cell)
        if target_column is None:
            raise ValueError(f"Invalid cell address: {target_cell}")

        min_row = worksheet.min_row or 1
        max_row = worksheet.max_row or 1
        min_column = 1
        max_column = worksheet.max_column or 1

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        column_headers = [col_letter(column) for column in range(min_column, max_column + 1)]
        writer.writerow([""] + column_headers)

        for row_index, row in enumerate(
            worksheet.iter_rows(
                min_row=min_row,
                max_row=max_row,
                min_col=min_column,
                max_col=max_column,
            )
        ):
            row_number = min_row + row_index
            values = [str(row_number)]
            for column_index, cell in enumerate(row):
                column_number = min_column + column_index
                if row_number == target_row and column_number == target_column:
                    values.append("")
                else:
                    values.append(cell_display(cell))
            writer.writerow(values)

        return buffer.getvalue().rstrip("\r\n")
