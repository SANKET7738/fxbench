"""Value comparison logic for executed formula results."""

from fxbench.evaluation.executor import (
    CellValue,
    ColumnValue,
    ErrorValue,
    ExecutedValue,
    RangeValue,
    Scalar,
    is_excel_error_value,
)


def compare(a: ExecutedValue, b: ExecutedValue, formatted: bool = False) -> bool:
    """Compare two executed values for equality."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ErrorValue):
        return is_excel_error_value(a) and a.error_type == b.error_type

    to_compare: list[tuple[CellValue, CellValue]] = []

    if isinstance(a, CellValue):
        to_compare.append((a, b))
    elif isinstance(a, ColumnValue):
        if len(a.cells) != len(b.cells):
            return False
        for cell_a, cell_b in zip(a.cells, b.cells):
            to_compare.append((cell_a, cell_b))
    elif isinstance(a, RangeValue):
        if len(a.cells) != len(b.cells):
            return False
        for row_a, row_b in zip(a.cells, b.cells):
            if len(row_a) != len(row_b):
                return False
            for cell_a, cell_b in zip(row_a, row_b):
                to_compare.append((cell_a, cell_b))

    for cell_a, cell_b in to_compare:
        val_a = cell_a.formatted if formatted else cell_a.value
        val_b = cell_b.formatted if formatted else cell_b.value
        if not _compare_scalar(val_a, val_b):
            return False
    return True


def _compare_scalar(a: Scalar, b: Scalar) -> bool:
    """Compare two scalar values with float tolerance of 1e-6."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < 1e-6
    return a == b
