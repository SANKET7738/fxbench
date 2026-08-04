from __future__ import annotations

import re
from datetime import date, datetime, time

_EXCEL_INTERNAL_PREFIX_RE = re.compile(r"_xlfn\.|_xlws\.|_xlpm\.", re.IGNORECASE)


def sanitize_formula(formula: str) -> str:
    return _EXCEL_INTERNAL_PREFIX_RE.sub("", formula)


def col_letter(column_index: int) -> str:
    result = ""
    while column_index > 0:
        column_index, remainder = divmod(column_index - 1, 26)
        result = chr(remainder + ord("A")) + result
    return result


def parse_cell_address(address: str) -> tuple[int | None, int | None]:
    match = re.match(r"([A-Z]+)(\d+)", address.upper())
    if not match:
        return None, None

    column_label, row_number = match.group(1), int(match.group(2))
    column_index = 0
    for character in column_label:
        column_index = column_index * 26 + (ord(character) - ord("A") + 1)

    return column_index, row_number


def cell_display(cell) -> str:
    data_type = getattr(cell, "data_type", "n")
    if data_type == "f":
        from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula

        value = cell.value
        if value is None or isinstance(value, DataTableFormula):
            return ""
        if isinstance(value, ArrayFormula):
            text = value.text
            if text and not str(text).startswith("="):
                return sanitize_formula(f"={text}")
            return sanitize_formula(str(text)) if text else ""
        if hasattr(value, "text"):
            return sanitize_formula(f"={value.text}")

        formula = str(value)
        if not formula.startswith("="):
            formula = f"={formula}"
        return sanitize_formula(formula)

    if cell.value is None:
        return ""
    return str(cell.value)


def cell_value_str(cell) -> str:
    value = cell.value
    if value is None:
        return ""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


def normalize_value(cell):
    from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula

    value = cell.value
    if value is None or isinstance(value, DataTableFormula):
        return ""
    if isinstance(value, ArrayFormula):
        text = value.text
        if text and not str(text).startswith("="):
            return sanitize_formula(f"={text}")
        return sanitize_formula(str(text)) if text else ""
    if hasattr(value, "text") and not isinstance(value, (str, int, float, bool)):
        text = value.text
        if text and not str(text).startswith("="):
            return sanitize_formula(f"={text}")
        return sanitize_formula(str(text)) if text else ""
    if isinstance(value, str) and value.startswith("="):
        return sanitize_formula(value)
    return value
