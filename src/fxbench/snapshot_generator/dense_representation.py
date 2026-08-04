from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_excel_column_name(column_index: int) -> str:
    result = ""
    while column_index >= 0:
        result = chr(column_index % 26 + ord("A")) + result
        column_index = column_index // 26 - 1
        if column_index < 0:
            break
    return result


def first_sheet_name(filepath: str | Path) -> str:
    with pd.ExcelFile(filepath, engine="openpyxl") as excel_file:
        return excel_file.sheet_names[0]


def create_dense_representation(filepath: str | Path, sheet_name: str | None = None) -> str:
    selected_sheet = sheet_name or first_sheet_name(filepath)
    dataframe = pd.read_excel(filepath, sheet_name=selected_sheet, engine="openpyxl")
    dataframe = dataframe.astype(object).fillna("")

    header_row = pd.DataFrame([dataframe.columns.tolist()], columns=dataframe.columns)
    dataframe_with_header = pd.concat([header_row, dataframe], ignore_index=True)

    dataframe_with_header.index = range(1, len(dataframe_with_header) + 1)
    dataframe_with_header.index.name = "Row"
    dataframe_with_header.columns = [get_excel_column_name(index) for index in range(len(dataframe.columns))]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        return dataframe_with_header.to_markdown(index=True)
