"""Theoretical Limit Analysis.

For each content strategy, compute what % of FCBench problems have all
formula-referenced cells (+ intent header cells) visible in the context.
This is the ceiling — if a required cell isn't in the context, no model
can produce the correct formula.

Usage:
    python -m analysis.theoretical_limit [--take N] [--minisheet-width 2]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter, rows_from_range

# Paths are resolved relative to the current working directory (run from the repo
# root), mirroring the rest of fxbench's Assets/ conventions.
REPO_ROOT = Path.cwd()
INTENT_CELLS_FILE = REPO_ROOT / "Assets" / "Annotations" / "FxBench" / "intent_cells.jsonl"

STRATEGIES = [
    "PipeDelimited",
    "SheetCompressor",
    "Windowed",
    "SpreadsheetCoder",
    "MiniSheet",
]

STRATEGY_LABELS = {
    "PipeDelimited": "PipeDelimited (Full Sheet)",
    "SheetCompressor": "SheetCompressor (Full Sheet)",
    "Windowed": "Windowed (±20r/±10c)",
    "SpreadsheetCoder": "SpreadsheetCoder (±10/±10)",
    "MiniSheet": "MiniSheet (blob edges)",
}

DEFAULT_MINISHEET_WIDTH = 2

METRICS = (
    "visible",
    "mentioned",
    "printed",
    "intent_visible",
    "intent_mentioned",
    "intent_printed",
)

STRUCTURE_ALL_METRIC = "structure_annotated_all"
STRUCTURE_AVG_METRIC = "structure_annotated_avg"
TYPE_ALL_METRIC = "type_annotated_all"
TYPE_AVG_METRIC = "type_annotated_avg"

ANNOTATION_METRIC_SPECS = (
    ("structure_annotated", STRUCTURE_ALL_METRIC, STRUCTURE_AVG_METRIC, "structured"),
    ("type_annotated", TYPE_ALL_METRIC, TYPE_AVG_METRIC, "typed"),
)


@dataclass
class CellEvidence:
    """Strategy-level evidence for address availability and prominence."""

    visible: set[str]
    mentioned: set[str]
    printed: set[str]
    structured: set[str] = field(default_factory=set)
    typed: set[str] = field(default_factory=set)


_MINISHEET_CELL_CACHE: dict[tuple[int, str, str | None], dict] = {}
_INTENT_ANNOTATION_CACHE: dict[str, dict] | None = None

_RE_INFER_EMAIL = re.compile(r"^[\w.\-]+@[\w.\-]+\.\w+$")
_RE_INFER_CURRENCY = re.compile(
    r"^[+-]?[$€£¥]\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$"
)
_RE_INFER_PERCENTAGE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%$")
_RE_INFER_SCIENTIFIC = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$"
)
_RE_INFER_FLOAT = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)$")
_RE_INFER_INT = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)$")
_RE_INFER_ISO_DATE = re.compile(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?$")
_RE_INFER_DATE_WITH_YEAR = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")
_RE_INFER_TIME = re.compile(
    r"^(?:2[0-3]|[01]?\d):[0-5]\d(?::[0-5]\d)?(?:\s?[AaPp][Mm])?$"
)
_EXCEL_ERROR_VALUES = {
    "#DIV/0!",
    "#N/A",
    "#NAME?",
    "#NULL!",
    "#NUM!",
    "#REF!",
    "#VALUE!",
    "#SPILL!",
    "#CALC!",
    "#FIELD!",
    "#GETTING_DATA",
    "#BLOCKED!",
    "#UNKNOWN!",
}

# ---------------------------------------------------------------------------
# Cell reference extraction
# ---------------------------------------------------------------------------

_CELL_REF_PATTERN = re.compile(
    r"(?:'[^']*'!|\w+!)?"           # optional sheet prefix (cross-sheet)
    r"(\$?[A-Z]{1,3}\$?\d+"         # cell ref start
    r"(?::\$?[A-Z]{1,3}\$?\d+)?)"   # optional range end
)


def _normalize_ref(ref: str) -> str:
    return ref.replace("$", "")


def _expand_range(start: str, end: str) -> set[str]:
    """Expand A1:B3 → {A1, A2, A3, B1, B2, B3}."""
    r1, c1 = coordinate_to_tuple(start)
    r2, c2 = coordinate_to_tuple(end)
    cells = set()
    for r in range(min(r1, r2), max(r1, r2) + 1):
        for c in range(min(c1, c2), max(c1, c2) + 1):
            cells.add(f"{get_column_letter(c)}{r}")
    return cells


def _expand_ref(ref: str) -> set[str]:
    if ":" not in ref:
        return {ref}
    start, end = ref.split(":", 1)
    return _expand_range(start, end)


def _singletons_after_compression(addresses: list[str]) -> set[str]:
    """Return addresses that MiniSheet would render as individual addresses."""
    from fxbench.context_builders.serializers.minisheet import _compress

    printed: set[str] = set()
    for ref in _compress(addresses):
        if ":" not in ref:
            printed.add(ref)
    return printed


def _has_ambiguous_leading_zero(cleaned: str) -> bool:
    unsigned = cleaned.lstrip("+-")
    if not unsigned or unsigned == "0":
        return False
    if unsigned.startswith("0."):
        return False
    return unsigned.startswith("0")


def _looks_like_datetime(value: str) -> bool:
    for separator in ("T", " "):
        if separator not in value:
            continue
        date_part, time_part = value.split(separator, 1)
        if (
            (_RE_INFER_ISO_DATE.match(date_part) or _RE_INFER_DATE_WITH_YEAR.match(date_part))
            and _RE_INFER_TIME.match(time_part.strip())
        ):
            return True
    return False


def _infer_rendered_type(value: object) -> str | None:
    """Return the type exposed by a non-empty rendered context value."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    if isinstance(value, int):
        return "year" if 1900 <= value <= 2099 else "int"
    if isinstance(value, float):
        return "float"

    text = str(value).strip()
    if not text:
        return "string" if isinstance(value, str) else None
    text_upper = text.upper()
    if text.startswith("="):
        return "formula"
    if text_upper in _EXCEL_ERROR_VALUES:
        return "error"
    if text_upper in {"TRUE", "FALSE"}:
        return "bool"
    if _RE_INFER_EMAIL.match(text):
        return "email"
    if _RE_INFER_CURRENCY.match(text):
        return "currency"
    if _RE_INFER_PERCENTAGE.match(text):
        return "percentage"
    if _looks_like_datetime(text):
        return "datetime"
    if _RE_INFER_ISO_DATE.match(text) or _RE_INFER_DATE_WITH_YEAR.match(text):
        return "date"
    if _RE_INFER_TIME.match(text):
        return "time"

    numeric = text.replace(",", "")
    if not _has_ambiguous_leading_zero(numeric):
        if _RE_INFER_SCIENTIFIC.match(text):
            return "scientific"
        if _RE_INFER_FLOAT.match(numeric):
            return "float"
        if _RE_INFER_INT.match(text):
            try:
                integer = int(numeric)
            except ValueError:
                return "int"
            return "year" if 1900 <= integer <= 2099 else "int"

    return "string"


def extract_required_refs(formula: str) -> tuple[set[str], bool]:
    """Extract all same-sheet cell refs from formula, expanding ranges.

    Returns (cell_set, has_cross_sheet_refs).
    """
    cells: set[str] = set()
    has_cross = False

    for m in _CELL_REF_PATTERN.finditer(formula):
        full_match = m.group(0)  # includes sheet prefix if present
        ref = _normalize_ref(m.group(1))  # cell ref without sheet prefix
        if "!" in full_match:
            has_cross = True
            continue
        if ":" in ref:
            start, end = ref.split(":")
            cells.update(_expand_range(start, end))
        else:
            cells.add(ref)
    return cells, has_cross


# ---------------------------------------------------------------------------
# Intent (header / label) cells
# ---------------------------------------------------------------------------

def _find_header_bounds(ws, target_row: int, target_col: int) -> tuple[int, int, int] | None:
    """Return (header_row, min_col, max_col) for the table containing target."""
    try:
        for table in ws.tables.values():
            idx_rows = list(rows_from_range(table.ref))
            if not idx_rows:
                continue
            fc = idx_rows[0][0]
            lc = idx_rows[-1][-1]
            fr, fc_ = coordinate_to_tuple(fc)
            lr, lc_ = coordinate_to_tuple(lc)
            if fc_ <= target_col <= lc_ and fr <= target_row <= lr:
                return fr, fc_, lc_
    except Exception:
        pass
    return None


def _find_header_row(ws, target_row: int, target_col: int) -> int | None:
    """Return the header row number if the target cell belongs to a table."""
    bounds = _find_header_bounds(ws, target_row, target_col)
    return bounds[0] if bounds is not None else None


def _find_table_label_col(ws, target_row: int, target_col: int) -> int | None:
    """Return the label column (first col of the table containing target)."""
    try:
        for table in ws.tables.values():
            idx_rows = list(rows_from_range(table.ref))
            if not idx_rows:
                continue
            fc = idx_rows[0][0]
            lc = idx_rows[-1][-1]
            fr, fc_ = coordinate_to_tuple(fc)
            lr, lc_ = coordinate_to_tuple(lc)
            if fc_ <= target_col <= lc_ and fr <= target_row <= lr:
                return fc_
    except Exception:
        pass
    return None


def _nearest_string_above(ws, start_row: int, col: int) -> str | None:
    for row in range(start_row - 1, 0, -1):
        cell = ws.cell(row=row, column=col)
        if isinstance(cell.value, str) and cell.value.strip():
            return f"{get_column_letter(col)}{row}"
    return None


def _nearest_string_left(ws, row: int, start_col: int) -> str | None:
    for col in range(start_col - 1, 0, -1):
        cell = ws.cell(row=row, column=col)
        if isinstance(cell.value, str) and cell.value.strip():
            return f"{get_column_letter(col)}{row}"
    return None


def get_intent_cells(
    referenced_cells: set[str],
    target_cell: str,
    ws,
    annotation: dict | None,
) -> set[str]:
    """Determine header / label cells based on FCBench annotation + ws.tables."""
    intent: set[str] = set()
    hint = (annotation or {}).get("intent_hint", {})
    source = hint.get("source", "") if hint else ""
    if not source:
        return intent

    t_row, t_col = coordinate_to_tuple(target_cell)

    if source == "col_header":
        header_row = _find_header_row(ws, t_row, t_col)
        if header_row is not None:
            cols = set()
            for ref in referenced_cells:
                _, c = coordinate_to_tuple(ref)
                cols.add(c)
            cols.add(t_col)
            for c in cols:
                intent.add(f"{get_column_letter(c)}{header_row}")
        else:
            col_start_rows = {t_col: t_row}
            for ref in referenced_cells:
                r, c = coordinate_to_tuple(ref)
                col_start_rows[c] = min(col_start_rows.get(c, r), r)
            for c, start_row in col_start_rows.items():
                if header_cell := _nearest_string_above(ws, start_row, c):
                    intent.add(header_cell)

    elif source == "row_label":
        label_col = _find_table_label_col(ws, t_row, t_col)
        if label_col is not None:
            rows = set()
            for ref in referenced_cells:
                r, _ = coordinate_to_tuple(ref)
                rows.add(r)
            rows.add(t_row)
            for r in rows:
                intent.add(f"{get_column_letter(label_col)}{r}")
        else:
            row_start_cols = {t_row: t_col}
            for ref in referenced_cells:
                r, c = coordinate_to_tuple(ref)
                row_start_cols[r] = min(row_start_cols.get(r, c), c)
            for r, start_col in row_start_cols.items():
                if label_cell := _nearest_string_left(ws, r, start_col):
                    intent.add(label_cell)

    return intent


def load_intent_annotations() -> dict[str, dict]:
    """Load LLM-annotated intent cell addresses keyed by problem id."""
    global _INTENT_ANNOTATION_CACHE
    if _INTENT_ANNOTATION_CACHE is not None:
        return _INTENT_ANNOTATION_CACHE

    cache: dict[str, dict] = {}
    if INTENT_CELLS_FILE.exists():
        with open(INTENT_CELLS_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["id"]] = rec.get("intent_cells") or {}
    _INTENT_ANNOTATION_CACHE = cache
    return cache


def _resolve_intent_cells(
    rec_id: str,
    referenced_cells: set[str],
    target_cell: str,
    ws,
    annotation: dict | None,
    intent_cells_by_id: dict[str, dict],
    *,
    allow_fallback: bool,
) -> set[str]:
    if rec_id in intent_cells_by_id:
        return _intent_cells_from_annotation(intent_cells_by_id[rec_id])
    if allow_fallback:
        return get_intent_cells(referenced_cells, target_cell, ws, annotation)
    return set()


def _intent_cells_from_annotation(intent_data: dict | None) -> set[str]:
    """Return explicit intent cell addresses from intent_cells.jsonl.

    ``data_range`` is intentionally excluded: formula/range references are
    already extracted from the ground-truth formula and are not intent cells.
    """
    intent: set[str] = set()
    if not intent_data:
        return intent
    for field in ("direct_intent", "data_description"):
        for ref in intent_data.get(field, []) or []:
            if not isinstance(ref, str):
                continue
            ref = _normalize_ref(ref.strip().upper())
            if not ref:
                continue
            try:
                intent.update(_expand_ref(ref))
            except ValueError:
                continue
    return intent


# ---------------------------------------------------------------------------
# Visible-cell-set computation per strategy
# ---------------------------------------------------------------------------

def _visible_full_sheet(ws) -> set[str]:
    """PipeDelimited / SheetCompressor: all addresses in the used range."""
    cells: set[str] = set()
    for r in range(ws.min_row or 1, (ws.max_row or 1) + 1):
        for c in range(ws.min_column or 1, (ws.max_column or 1) + 1):
            cells.add(f"{get_column_letter(c)}{r}")
    return cells


def _visible_windowed(ws, target_cell: str, row_margin: int = 20, col_margin: int = 10) -> set[str]:
    """Windowed strategy: ±row_margin / ±col_margin around target."""
    t_row, t_col = coordinate_to_tuple(target_cell)
    min_r = max(1, t_row - row_margin)
    max_r = min(ws.max_row or t_row, t_row + row_margin)
    min_c = max(1, t_col - col_margin)
    max_c = min(ws.max_column or t_col, t_col + col_margin)
    cells: set[str] = set()
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            cells.add(f"{get_column_letter(c)}{r}")
    return cells


def _visible_spreadsheetcoder(ws, target_cell: str) -> set[str]:
    """SpreadsheetCoder: ±10/±10 + containing-table header row."""
    cells = _visible_windowed(ws, target_cell, row_margin=10, col_margin=10)
    t_row, t_col = coordinate_to_tuple(target_cell)
    header_bounds = _find_header_bounds(ws, t_row, t_col)
    if header_bounds is not None:
        header_row, header_min_col, header_max_col = header_bounds
        for c in range(header_min_col, header_max_col + 1):
            cells.add(f"{get_column_letter(c)}{header_row}")
    return cells


def _visible_sheetcompressor(ws, target_cell: str) -> set[str]:
    """SheetCompressor: all non-empty cells in the used range.

    SheetCompressor iterates the entire sheet but excludes empty cells
    from output (type -1 is dropped).  Non-empty cells appear either
    individually or inside region range-notations like (IntNum|A1:A50).
    """
    cells: set[str] = set()
    for row in ws.iter_rows(min_row=ws.min_row, max_row=ws.max_row,
                            min_col=ws.min_column, max_col=ws.max_column):
        for cell in row:
            if cell.value is not None and cell.value != "":
                cells.add(f"{get_column_letter(cell.column)}{cell.row}")
    # Target cell address is always present (shown as blank placeholder)
    cells.add(target_cell)
    return cells


def _visible_minisheet(
    ws,
    target_cell: str,
    wb_path: str = None,
    width: int = DEFAULT_MINISHEET_WIDTH,
    include_strings: bool = True,
    include_clusters: bool = True,
    include_formulas: bool = True,
    include_tables: bool = True,
) -> set[str]:
    """MiniSheet: table header+first data row, blob-edge strings,
    all non-string non-empty cells (clusters), all formula cells.

    Interior string cells are counted as visible if the blob edges
    expose the same string value — the model can infer the cell's
    value from the edge address groupings.
    """
    from fxbench.context_builders.serializers.minisheet import (
        _Cell as _MSCell,
        _extract_cells,
        _find_blobs,
        _edge_wrapped,
    )

    sheet_cells = _extract_cells(ws, workbook_path=wb_path)
    if target_cell in sheet_cells:
        sheet_cells[target_cell] = _MSCell(
            address=target_cell, value=None, formula=None, format=None,
        )

    visible: set[str] = set()
    exclude: set[str] = set()

    # 1. Tables: header + first data row
    try:
        for table in ws.tables.values():
            idx_rows = list(rows_from_range(table.ref))
            if include_tables:
                for row in idx_rows[:2]:
                    for addr in row:
                        visible.add(addr)
            for row in idx_rows:
                exclude.update(row)
    except Exception:
        pass

    # 2. Blob-edge string cells
    if include_strings:
        blobs = _find_blobs(sheet_cells)
        for blob in blobs:
            edge_cells = _edge_wrapped(blob, width, width, width, width)

            # Collect unique string values shown on edges
            edge_string_values: set[str] = set()
            for cell in edge_cells:
                if (cell.value not in {None, ""}
                        and cell.address not in exclude
                        and cell.formula is None
                        and isinstance(cell.value, str)):
                    visible.add(cell.address)
                    edge_string_values.add(cell.value)

            # Interior string cells with a value already shown on edges
            # are effectively covered — the model knows which rows have
            # which value from the edge address groupings
            for cell in blob:
                if (cell.address not in visible
                        and cell.address not in exclude
                        and cell.formula is None
                        and isinstance(cell.value, str)
                        and cell.value in edge_string_values):
                    visible.add(cell.address)

    # 3. ALL non-string, non-empty cells in sheet (clusters)
    if include_clusters:
        for cell in sheet_cells.values():
            if cell.value in {None, ""} or cell.address in exclude:
                continue
            if isinstance(cell.value, str):
                continue
            visible.add(cell.address)

    # 4. ALL formula cells in sheet
    if include_formulas:
        for cell in sheet_cells.values():
            if cell.formula is None or cell.address in exclude:
                continue
            visible.add(cell.address)

    return visible




def _all_printed_evidence(cells: set[str]) -> CellEvidence:
    cells = set(cells)
    return CellEvidence(
        visible=cells,
        mentioned=set(cells),
        printed=set(cells),
        structured=set(),
        typed=set(),
    )


def _typed_from_rendered_values(ws, target_cell: str, addresses: set[str]) -> set[str]:
    from fxbench.context_builders.serializers._utils import normalize_value

    typed: set[str] = set()
    for addr in addresses:
        if addr == target_cell:
            continue
        r, c = coordinate_to_tuple(addr)
        cell = ws.cell(row=r, column=c)
        value = normalize_value(cell)
        if value is None or value == "":
            if getattr(cell, "data_type", "n") == "f" and cell.value is not None:
                typed.add(addr)
            continue
        if _infer_rendered_type(value) is not None:
            typed.add(addr)
    return typed


def _all_printed_evidence_with_inferred_types(
    ws,
    target_cell: str,
    cells: set[str],
) -> CellEvidence:
    evidence = _all_printed_evidence(cells)
    evidence.typed.update(_typed_from_rendered_values(ws, target_cell, evidence.visible))
    return evidence


def _evidence_spreadsheetcoder(ws, target_cell: str) -> CellEvidence:
    evidence = _all_printed_evidence_with_inferred_types(
        ws,
        target_cell,
        _visible_spreadsheetcoder(ws, target_cell),
    )
    t_row, t_col = coordinate_to_tuple(target_cell)
    header_bounds = _find_header_bounds(ws, t_row, t_col)
    if header_bounds is None:
        return evidence

    header_row, header_min_col, header_max_col = header_bounds
    min_row = max(1, t_row - 10)
    max_row = min(ws.max_row or t_row, t_row + 10)
    min_col = max(1, t_col - 10)
    max_col = min(ws.max_column or t_col, t_col + 10)
    header_needs_section = (
        header_row < min_row
        or header_row > max_row
        or header_min_col < min_col
        or header_max_col > max_col
    )
    if header_needs_section:
        evidence.structured.update(
            f"{get_column_letter(col)}{header_row}"
            for col in range(header_min_col, header_max_col + 1)
        )
    return evidence


def _evidence_sheetcompressor(ws, target_cell: str) -> CellEvidence:
    from fxbench.context_builders.serializers._utils import normalize_value
    from fxbench.context_builders.serializers.sheet_compressor import classify_cell, _find_regions

    visible = _visible_sheetcompressor(ws, target_cell)
    mentioned = set(visible)
    printed: set[str] = set()
    structured: set[str] = set()
    typed: set[str] = set()

    t_row, t_col = coordinate_to_tuple(target_cell)
    min_row = ws.min_row or 1
    max_row = ws.max_row or 1
    min_col = ws.min_column or 1
    max_col = ws.max_column or 1
    nrows = max_row - min_row + 1
    ncols = max_col - min_col + 1

    type_grid: list[list[int]] = [[-1] * ncols for _ in range(nrows)]
    formula_grid: list[list[bool]] = [[False] * ncols for _ in range(nrows)]
    value_grid: list[list[object]] = [[None] * ncols for _ in range(nrows)]

    for r_idx, row in enumerate(
        ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col)
    ):
        for c_idx, cell in enumerate(row):
            row_num = min_row + r_idx
            col_num = min_col + c_idx
            if row_num == t_row and col_num == t_col:
                continue

            raw = normalize_value(cell)
            value_grid[r_idx][c_idx] = raw
            data_type = getattr(cell, "data_type", "n")
            is_formula = data_type == "f" or (
                isinstance(raw, str) and str(raw).startswith("=")
            )
            if is_formula:
                formula_grid[r_idx][c_idx] = True
            else:
                type_grid[r_idx][c_idx] = classify_cell(raw)[0]

    _, in_region = _find_regions(type_grid, nrows, ncols)
    for r in range(nrows):
        for c in range(ncols):
            row_num = min_row + r
            col_num = min_col + c
            addr = f"{get_column_letter(col_num)}{row_num}"
            if formula_grid[r][c]:
                printed.add(addr)
                typed.add(addr)
            elif not in_region[r][c] and type_grid[r][c] != -1:
                printed.add(addr)
                if _infer_rendered_type(value_grid[r][c]) is not None:
                    typed.add(addr)
            if in_region[r][c]:
                structured.add(addr)
                typed.add(addr)

    return CellEvidence(
        visible=visible,
        mentioned=mentioned,
        printed=printed,
        structured=structured,
        typed=typed,
    )


def _blanked_minisheet_cells(ws, target_cell: str, wb_path: str | None):
    from fxbench.context_builders.serializers.minisheet import _Cell as _MSCell, _extract_cells

    key = (id(ws), target_cell, wb_path)
    if key in _MINISHEET_CELL_CACHE:
        return _MINISHEET_CELL_CACHE[key]

    sheet_cells = _extract_cells(ws, workbook_path=wb_path)
    if target_cell in sheet_cells:
        sheet_cells[target_cell] = _MSCell(
            address=target_cell, value=None, formula=None, format=None,
        )
    _MINISHEET_CELL_CACHE[key] = sheet_cells
    return sheet_cells


def _infer_minisheet_cell_type(cell, *, count_strings: bool = False) -> str | None:
    if cell is None:
        return None
    if cell.formula is not None:
        return _infer_rendered_type(cell.formula)
    if count_strings and isinstance(cell.value, str) and cell.value != "":
        return "string"
    return _infer_rendered_type(cell.value)


def _minisheet_table_exclude(ws) -> set[str]:
    exclude: set[str] = set()
    try:
        for table in ws.tables.values():
            exclude.update(_expand_ref(table.ref))
    except Exception:
        pass
    return exclude


def _evidence_minisheet(
    ws,
    target_cell: str,
    wb_path: str | None = None,
    width: int = DEFAULT_MINISHEET_WIDTH,
    include_strings: bool = True,
    include_clusters: bool = True,
    include_formulas: bool = True,
    include_tables: bool = True,
) -> CellEvidence:
    from fxbench.context_builders.serializers.minisheet import (
        _edge_wrapped,
        _find_blobs,
        _format_name,
        _format_value,
        _group_formula,
    )

    sheet_cells = _blanked_minisheet_cells(ws, target_cell, wb_path)

    visible: set[str] = set()
    mentioned: set[str] = set()
    printed: set[str] = set()
    structured: set[str] = set()
    typed: set[str] = set()
    exclude: set[str] = set()

    try:
        for table in ws.tables.values():
            idx_rows = list(rows_from_range(table.ref))
            table_cells = _expand_ref(table.ref)
            if include_tables:
                for row in idx_rows[:2]:
                    visible.update(row)
                    printed.update(row)
                    for addr in row:
                        if _infer_minisheet_cell_type(sheet_cells.get(addr)) is not None:
                            typed.add(addr)
                mentioned.update(table_cells)
                structured.update(table_cells)
            exclude.update(table_cells)
    except Exception:
        pass

    blobs = _find_blobs(sheet_cells)

    if include_strings:
        value_groups: dict[str, list[str]] = defaultdict(list)
        for blob in blobs:
            edge_cells = _edge_wrapped(blob, width, width, width, width)
            edge_string_values: set[str] = set()
            for cell in edge_cells:
                if (
                    cell.value not in {None, ""}
                    and cell.address not in exclude
                    and cell.formula is None
                    and isinstance(cell.value, str)
                ):
                    visible.add(cell.address)
                    edge_string_values.add(cell.value)
                    value_groups[_format_value(cell.value)].append(cell.address)
            for cell in blob:
                if (
                    cell.address not in visible
                    and cell.address not in exclude
                    and cell.formula is None
                    and isinstance(cell.value, str)
                    and cell.value in edge_string_values
                ):
                    visible.add(cell.address)
        for addresses in value_groups.values():
            mentioned.update(addresses)
            printed.update(_singletons_after_compression(addresses))
            structured.update(addresses)
            typed.update(addresses)

    if include_clusters:
        cluster_groups: dict[str, list[str]] = defaultdict(list)
        for cell in sheet_cells.values():
            if cell.value in {None, ""} or cell.address in exclude:
                continue
            if isinstance(cell.value, str):
                continue
            visible.add(cell.address)
            cluster_groups[_format_name(cell.format or "")].append(cell.address)
        for addresses in cluster_groups.values():
            mentioned.update(addresses)
            structured.update(addresses)
            typed.update(addresses)

    if include_formulas:
        formula_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for cell in sheet_cells.values():
            if cell.formula is None or cell.address in exclude:
                continue
            visible.add(cell.address)
            formula_groups[_group_formula(cell.formula)].append((cell.address, cell.formula))
        for entries in formula_groups.values():
            addresses = [addr for addr, _ in entries]
            mentioned.update(addresses)
            structured.update(addresses)
            typed.update(addresses)
            printed.update(addresses)

    return CellEvidence(
        visible=visible,
        mentioned=mentioned,
        printed=printed,
        structured=structured,
        typed=typed,
    )




def _evidence_fns(
    minisheet_width: int = DEFAULT_MINISHEET_WIDTH,
    minisheet_kwargs: dict | None = None,
):
    minisheet_kwargs = dict(minisheet_kwargs or {})
    return {
        "PipeDelimited": lambda ws, tc, wp=None: _all_printed_evidence_with_inferred_types(
            ws, tc, _visible_full_sheet(ws)
        ),
        "SheetCompressor": lambda ws, tc, wp=None: _evidence_sheetcompressor(ws, tc),
        "Windowed": lambda ws, tc, wp=None: _all_printed_evidence_with_inferred_types(
            ws, tc, _visible_windowed(ws, tc)
        ),
        "SpreadsheetCoder": lambda ws, tc, wp=None: _evidence_spreadsheetcoder(ws, tc),
        "MiniSheet": lambda ws, tc, wp=None: _evidence_minisheet(
            ws, tc, wb_path=wp, width=minisheet_width, **minisheet_kwargs
        ),
    }

