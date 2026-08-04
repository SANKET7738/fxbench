from __future__ import annotations

from fxbench.context_builders.serializers._utils import cell_display, col_letter, parse_cell_address
from fxbench.context_builders.serializers.base import IContextBuilder
from fxbench.data_models import Problem


class MarkdownContextBuilder(IContextBuilder):
    def __init__(self, row_margin: int | None = None, col_margin: int | None = None) -> None:
        self.row_margin = row_margin
        self.col_margin = col_margin

    def build(self, problem: Problem) -> str:
        workbook, worksheet = self._open_worksheet(problem)
        try:
            return self._build_table(worksheet, problem.active_cell)
        finally:
            workbook.close()

    def _build_table(self, worksheet, target_cell: str) -> str:
        target_column, target_row = parse_cell_address(target_cell)
        if target_column is None:
            raise ValueError(f"Invalid cell address: {target_cell}")

        if self.row_margin is None:
            min_row = 1
            max_row = worksheet.max_row or target_row
        else:
            min_row = max(1, target_row - self.row_margin)
            max_row = min(worksheet.max_row or target_row, target_row + self.row_margin)

        if self.col_margin is None:
            min_column = 1
            max_column = worksheet.max_column or target_column
        else:
            min_column = max(1, target_column - self.col_margin)
            max_column = min(worksheet.max_column or target_column, target_column + self.col_margin)

        column_headers = [col_letter(column) for column in range(min_column, max_column + 1)]
        output_rows = [
            "| Row | " + " | ".join(column_headers) + " |",
            "|-----|" + "|".join(["---"] * len(column_headers)) + "|",
        ]

        for row_index, row in enumerate(
            worksheet.iter_rows(
                min_row=min_row,
                max_row=max_row,
                min_col=min_column,
                max_col=max_column,
            )
        ):
            row_number = min_row + row_index
            values = []
            for column_index, cell in enumerate(row):
                column_number = min_column + column_index
                if row_number == target_row and column_number == target_column:
                    values.append("")
                else:
                    values.append(cell_display(cell))
            output_rows.append(f"| {row_number} | " + " | ".join(values) + " |")

        return "\n".join(output_rows)
