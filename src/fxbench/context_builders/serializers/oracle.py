"""Oracle context builders — compressed ground-truth-aware representations.

Oracle builders reveal upper-bound signals for a formula-completion problem:
the ground-truth formula's references and intent cells, compressed by grouping
referenced cells with the same value and formulas with the same structural
pattern. Variants additionally expose a dependency chain, a neighbour grid,
same-column formulas, sheetwide formulas, computed values, a target-cell format
hint, or a natural-language instruction. Each builder carries its own
``system_prompt`` class attribute, which ``ContextPromptBuilder`` picks up.
"""

import json
import re
from collections import defaultdict
from datetime import date, datetime, time
from itertools import groupby
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter

from fxbench.context_builders.serializers.base import IContextBuilder
from fxbench.context_builders.serializers._utils import normalize_value
from fxbench.data_models import Problem

REPO_ROOT = Path(__file__).resolve().parents[4]
INTENT_CELLS_FILE = REPO_ROOT / "Assets" / "Annotations" / "FxBench" / "intent_cells.jsonl"
NL_PROMPTS_FILE = REPO_ROOT / "Assets" / "Annotations" / "FxBench" / "nl_prompts.jsonl"


# --- helpers (from oracle.py) ---


_CELL_REF_PATTERN = re.compile(
    r"(?:(?:'[^']*'|[\w.]+)!)?"
    r"(\$?[A-Z]{1,3}\$?\d+"
    r"(?::\$?[A-Z]{1,3}\$?\d+)?)"
)


_CROSS_SHEET_REF_PATTERN = re.compile(
    r"(?:'([^']*)'|([\w.]+))!"
    r"(\$?[A-Z]{1,3}\$?\d+"
    r"(?::\$?[A-Z]{1,3}\$?\d+)?)"
)


def _normalize_ref(ref: str) -> str:
    return ref.replace("$", "")


MAX_EXPANDED_CELLS = 500


def _expand_range(start: str, end: str) -> set[str]:
    r1, c1 = coordinate_to_tuple(start)
    r2, c2 = coordinate_to_tuple(end)
    total = (abs(r2 - r1) + 1) * (abs(c2 - c1) + 1)
    if total > MAX_EXPANDED_CELLS:
        # Too large to expand — just include corners
        cells = set()
        for r in [min(r1, r2), max(r1, r2)]:
            for c in [min(c1, c2), max(c1, c2)]:
                cells.add(f"{get_column_letter(c)}{r}")
        return cells
    cells = set()
    for r in range(min(r1, r2), max(r1, r2) + 1):
        for c in range(min(c1, c2), max(c1, c2) + 1):
            cells.add(f"{get_column_letter(c)}{r}")
    return cells


def _extract_refs_and_ranges(formula: str) -> tuple[set[str], list[str], bool]:
    """Extract same-sheet cell refs, range notations, and cross-sheet flag."""
    cells: set[str] = set()
    ranges: list[str] = []
    has_cross = False

    for m in _CELL_REF_PATTERN.finditer(formula):
        full_match = m.group(0)
        ref = _normalize_ref(m.group(1))
        if "!" in full_match:
            has_cross = True
            continue
        if ":" in ref:
            ranges.append(ref)
            start, end = ref.split(":")
            cells.update(_expand_range(start, end))
        else:
            cells.add(ref)
    return cells, ranges, has_cross


def _extract_cross_sheet_refs(formula: str) -> dict[str, set[str]]:
    """Extract cross-sheet refs grouped by sheet name → set of cell addresses."""
    by_sheet: dict[str, set[str]] = {}
    for m in _CROSS_SHEET_REF_PATTERN.finditer(formula):
        sheet_name = m.group(1) or m.group(2)
        ref = _normalize_ref(m.group(3))
        cells = by_sheet.setdefault(sheet_name, set())
        if ":" in ref:
            start, end = ref.split(":")
            cells.update(_expand_range(start, end))
        else:
            cells.add(ref)
    return by_sheet


def _extract_named_ranges(wb) -> dict[str, str]:
    """Resolve workbook defined names to 'Name → Sheet!Range' mapping."""
    resolved: dict[str, str] = {}
    try:
        for name in wb.defined_names.definedName:
            try:
                dests = list(name.destinations)
                if dests:
                    sheet, coord = dests[0]
                    resolved[name.name] = f"{sheet}!{coord}"
            except Exception:
                continue
    except Exception:
        pass
    return resolved


_FUNC_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_.]*(?=\()", re.IGNORECASE)


_ERROR_WRAPPERS = {"IFERROR", "IFNA", "_XLFN.IFERROR", "_XLFN.IFNA"}


def _extract_formula_hint(formula: str, function: str) -> str:
    """Extract formula hint: primary function, or IFERROR + inner function."""
    if not formula or not function:
        return ""

    upper_func = function.upper()
    if upper_func not in ("IFERROR", "IFNA"):
        return function

    # Find the second outermost function name
    funcs = _FUNC_NAME_PATTERN.findall(formula)
    # Skip _xlfn. prefixed versions and the wrapper itself
    clean_funcs = []
    for f in funcs:
        name = f.upper().replace("_XLFN.", "").replace("_XLWS.", "").replace("_XLPM.", "")
        if name and name not in _ERROR_WRAPPERS and not name.startswith("_"):
            clean_funcs.append(name)

    if clean_funcs:
        return f"{upper_func} + {clean_funcs[0]}"
    return upper_func


# --- helpers (from oracle.py) ---


def _compress_addresses(cells: list[str]) -> list[str]:
    """Compress a list of cell addresses into range notation."""
    if not cells:
        return []
    cells = sorted(cells, key=lambda x: (coordinate_to_tuple(x)[1], coordinate_to_tuple(x)[0]))
    compressed = []
    for _, group in groupby(cells, key=lambda x: coordinate_to_tuple(x)[1]):
        group = list(group)
        sequences = []
        current = [group[0]]
        for i in range(1, len(group)):
            prev_row = coordinate_to_tuple(group[i - 1])[0]
            curr_row = coordinate_to_tuple(group[i])[0]
            if curr_row == prev_row + 1:
                current.append(group[i])
            else:
                sequences.append(current)
                current = [group[i]]
        sequences.append(current)
        for seq in sequences:
            if len(seq) == 1:
                compressed.append(seq[0])
            else:
                compressed.append(f"{seq[0]}:{seq[-1]}")
    return compressed


def _format_value(value, width: int = 32) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        value = value.replace("\n", " ").replace("\r", " ")
        if len(value) > width:
            value = value[:width - 3] + "..."
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return repr(value)


def _group_formula(formula: str) -> str:
    """Replace row numbers in cell refs with @ to create a pattern."""
    pattern = re.compile(r"(\$?[A-Z]{1,3})\$?(\d+)")
    def _replace(m):
        return f"{m.group(1)}@"
    return pattern.sub(_replace, formula)


ORACLE_SYSTEM_PROMPT = """\
# Role

You are an expert Excel formula writer. Given a worksheet context and a target cell, \
predict the correct formula the user is typing.

# Context format

The context below contains **all the cells relevant to the formula** — they have been \
pre-identified from the full worksheet. You do not need to search for additional data; \
everything you need is provided.

The context has these sections:

## Sheet info
Sheet name, target cell, and what the formula should compute. A function suggestion \
may be provided — use it if appropriate.

## Intent cells
Header or label cells that describe the output (what the result represents) and the \
input (what data the formula consumes). Format: `address | value | role`

## Referenced cells (grouped by value)
The actual data cells the formula operates on. Cells with the same value are grouped \
together on one line. Format: `address_or_range | value`
- Addresses are compressed: `A2,A5,A8` means cells A2, A5, A8 share that value
- Ranges use colon notation: `B2:B100` means all cells from B2 to B100
- This shows you which cells contain which values and where they are in the sheet

## Formulas (grouped by pattern)
Other formula cells in the referenced area. Formulas with the same structure are grouped, \
with row numbers replaced by @. Use these as templates.

## Data ranges
Range notations (e.g., A2:A100) that the formula operates over.

# Task

Complete the formula prefix for the target cell.

# Response format

Return a JSON object: {"formula": "=YOUR_FORMULA_HERE"}

# Rules

- If the intent is unclear, return: {"formula": ""}
- Never modify the provided formula prefix."""


def _is_valid(addr: str) -> bool:
    try:
        coordinate_to_tuple(addr)
        return True
    except Exception:
        return False


def _sort_key(addrs: list[str]) -> tuple:
    if not addrs:
        return (9999, 9999)
    coords = []
    for a in addrs:
        try:
            coords.append(coordinate_to_tuple(a))
        except Exception:
            pass
    if not coords:
        return (9999, 9999)
    return min(coords)


ORACLE_NL_SYSTEM_PROMPT = """\
# Role

You are an expert Excel formula writer. Given a worksheet context and a target cell, \
predict the correct formula the user is typing.

# Context format

The context below contains **all the cells relevant to the formula** — they have been \
pre-identified from the full worksheet. You do not need to search for additional data; \
everything you need is provided.

The context has these sections:

## Sheet info
Sheet name, target cell, and what the formula should compute. A function suggestion \
may be provided — use it if appropriate.

## Formula instruction
A natural language description of exactly what the formula should do, including which \
function to use and how to wire the arguments. Follow this instruction closely.

## Intent cells
Header or label cells that describe the output (what the result represents) and the \
input (what data the formula consumes). Format: `address | value | role`

## Referenced cells (grouped by value)
The actual data cells the formula operates on. Cells with the same value are grouped \
together on one line. Format: `address_or_range | value`
- Addresses are compressed: `A2,A5,A8` means cells A2, A5, A8 share that value
- Ranges use colon notation: `B2:B100` means all cells from B2 to B100
- This shows you which cells contain which values and where they are in the sheet

## Formulas (grouped by pattern)
Other formula cells in the referenced area. Formulas with the same structure are grouped, \
with row numbers replaced by @. Use these as templates.

## Data ranges
Range notations (e.g., A2:A100) that the formula operates over.

# Task

Complete the formula prefix for the target cell. Follow the formula instruction — it \
describes exactly what to compute and which cells to use.

# Response format

Return a JSON object: {"formula": "=YOUR_FORMULA_HERE"}

# Rules

- If the intent is unclear, return: {"formula": ""}
- Never modify the provided formula prefix."""


# --- builder classes ---


class OracleContextBuilder(IContextBuilder):
    """Oracle context with MiniSheet-style grouping of referenced cells."""

    _read_only = False
    _intent_cache: dict | None = None
    system_prompt = ORACLE_SYSTEM_PROMPT

    def _load_intents(self) -> dict:
        if self._intent_cache is not None:
            return self._intent_cache
        cache: dict = {}
        if INTENT_CELLS_FILE.exists():
            with open(INTENT_CELLS_FILE, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    cache[rec["id"]] = rec.get("intent_cells") or {}
        self._intent_cache = cache
        return cache

    def build(self, problem: Problem) -> str:
        wb, ws = self._open_worksheet(problem)
        wb_data = load_workbook(
            str(problem.workbook_path), read_only=True, data_only=True
        )
        ws_data = wb_data[problem.worksheet_name]
        try:
            return self._build_context(wb, ws, ws_data, problem)
        finally:
            wb.close()
            wb_data.close()

    def _read_cell(self, ws, ws_data, address: str, target_cell: str):
        """Read cell value and formula. Returns (display_value, formula_or_None, raw_value)."""
        if address == target_cell:
            return "[TARGET]", None, None
        try:
            r, c = coordinate_to_tuple(address)
        except Exception:
            return "(error)", None, None

        cell = ws.cell(row=r, column=c)
        raw = normalize_value(cell)
        formula = None
        value = None

        if raw is not None and str(raw).startswith("="):
            formula = str(raw)
            data_cell = ws_data.cell(row=r, column=c)
            value = data_cell.value
        else:
            value = raw

        return value, formula, raw

    def _build_context(self, wb, ws, ws_data, problem: Problem) -> str:
        formula = problem.formula
        target_cell = problem.active_cell
        sheet_name = problem.worksheet_name

        # Get intent annotation
        intents = self._load_intents()
        intent_data = intents.get(problem.id, {})
        intent_desc = intent_data.get("intent", "")
        direct_intent = set(intent_data.get("direct_intent", []))
        data_desc = set(intent_data.get("data_description", []))
        annotated_ranges = intent_data.get("data_range", [])

        # Extract formula refs
        referenced_cells, formula_ranges, has_cross = _extract_refs_and_ranges(formula)
        cross_refs = _extract_cross_sheet_refs(formula) if has_cross else {}
        all_ranges = annotated_ranges or formula_ranges
        named_ranges = _extract_named_ranges(wb)

        # Collect all cells to process
        all_addrs = referenced_cells | direct_intent | data_desc
        all_addrs.discard(target_cell)

        # Read all cell values
        string_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)  # display_val -> [(addr, role)]
        number_groups: dict[str, list[str]] = defaultdict(list)  # format_name -> [addrs]
        formula_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)  # pattern -> [(addr, formula)]
        intent_cells: list[tuple[str, str, str]] = []  # (addr, value, role)

        for addr in sorted(all_addrs, key=lambda a: coordinate_to_tuple(a) if _is_valid(a) else (9999, 9999)):
            value, cell_formula, raw = self._read_cell(ws, ws_data, addr, target_cell)

            # Determine role
            role = None
            if addr in direct_intent:
                role = "output_header"
            elif addr in data_desc:
                role = "input_header"

            # Intent cells get their own section
            if role:
                display = _format_value(value) if value is not None else "(empty)"
                intent_cells.append((addr, display, role))
                continue

            # Skip empty
            if value is None or (isinstance(value, str) and value.strip() == ""):
                if cell_formula:
                    # Formula with no cached value
                    pattern = _group_formula(cell_formula)
                    formula_groups[pattern].append((addr, cell_formula))
                continue

            # Formula cells
            if cell_formula:
                pattern = _group_formula(cell_formula)
                formula_groups[pattern].append((addr, cell_formula))
                continue

            # String values
            if isinstance(value, str):
                string_groups[value].append(addr)
                continue

            # Numeric/date/other values
            display = _format_value(value)
            number_groups[display].append(addr)

        # Build output
        sections = []

        # Sheet info
        sections.append(f"Sheet: {sheet_name}")
        sections.append(f"Target cell: {target_cell}")
        if intent_desc:
            sections.append(f"Intent: {intent_desc}")

        # Hint
        annotation = problem.annotation or {}
        function = annotation.get("function") or ""
        if not function:
            funcs = _FUNC_NAME_PATTERN.findall(problem.formula or "")
            clean = [f.upper().replace("_XLFN.", "").replace("_XLWS.", "")
                     for f in funcs if not f.startswith("_")]
            function = clean[0] if clean else ""
        if function and function != "ABSTAIN":
            hint = _extract_formula_hint(problem.formula, function)
            if hint:
                sections.append(f"Consider using: {hint}")

        sections.append("")

        # Intent cells
        if intent_cells:
            sections.append("Intent cells:")
            for addr, val, role in intent_cells:
                sections.append(f"  {addr} | {val} | {role}")
            sections.append("")

        # Context cells grouped by value
        sections.append("Referenced cells (grouped by value):")

        # Strings
        for val, addrs in sorted(string_groups.items(), key=lambda x: _sort_key(x[1])):
            compressed = ",".join(_compress_addresses(addrs))
            sections.append(f"  {compressed} | {val}")

        # Numbers
        for val, addrs in sorted(number_groups.items(), key=lambda x: _sort_key(x[1])):
            compressed = ",".join(_compress_addresses(addrs))
            sections.append(f"  {compressed} | {val}")

        # Formulas
        if formula_groups:
            sections.append("")
            sections.append("Formulas (grouped by pattern):")
            for pattern, entries in sorted(formula_groups.items(), key=lambda x: _sort_key([a for a, _ in x[1]])):
                addrs = [a for a, _ in entries]
                compressed = ",".join(_compress_addresses(addrs))
                if len(entries) == 1:
                    sections.append(f"  {compressed} | {entries[0][1]}")
                else:
                    sections.append(f"  {compressed} | {pattern}")

        sections.append("")

        # Target
        sections.append(f"{target_cell} | [TARGET]")

        # Cross-sheet
        if cross_refs:
            sections.append("")
            sections.append("Cross-sheet references:")
            for sheet, addrs in sorted(cross_refs.items()):
                if sheet not in wb.sheetnames:
                    continue
                cross_ws = wb[sheet]
                for addr in sorted(addrs):
                    try:
                        r, c = coordinate_to_tuple(addr)
                        cell = cross_ws.cell(row=r, column=c)
                        raw = normalize_value(cell)
                        val = _format_value(raw) if raw else "(empty)"
                        sections.append(f"  '{sheet}'!{addr} | {val}")
                    except Exception:
                        pass

        # Data ranges
        if all_ranges:
            sections.append("")
            sections.append(f"Data ranges: {', '.join(all_ranges)}")

        if named_ranges:
            sections.append(f"Named ranges: {', '.join(f'{n} → {r}' for n, r in sorted(named_ranges.items()))}")

        return "\n".join(sections)


class OracleWithCellFormatContextBuilder(OracleContextBuilder):
    """Compressed oracle context plus the target cell's number-format hint."""

    @staticmethod
    def _classify_format(fmt: str) -> str | None:
        if not fmt or fmt == "General":
            return None
        if "%" in fmt:
            return "Percentage"
        if any(c in fmt for c in ("$", "\u20ac", "\u00a3", "\u00a5")):
            return "Currency"
        fmt_lower = fmt.lower()
        if any(p in fmt_lower for p in ("yy", "mm/dd", "dd/mm", "mmm")):
            if any(p in fmt_lower for p in ("h:", "hh:")):
                return "DateTime"
            return "Date"
        if any(p in fmt_lower for p in ("h:", "hh:", "mm:ss")):
            return "Time"
        if "E+" in fmt or "E-" in fmt:
            return "Scientific"
        return None

    def _build_context(self, wb, ws, ws_data, problem: Problem) -> str:
        context = super()._build_context(wb, ws, ws_data, problem)

        target_cell = problem.active_cell
        try:
            r, c = coordinate_to_tuple(target_cell)
            cell = ws.cell(row=r, column=c)
            fmt = cell.number_format
            label = self._classify_format(fmt)
        except Exception:
            label = None

        if label:
            lines = context.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Target cell:"):
                    lines.insert(i + 1, f"Expected format: {label}")
                    break
            context = "\n".join(lines)

        return context


class OracleWithDepChainContextBuilder(OracleContextBuilder):
    """Compressed oracle context plus an explicit dependency chain showing computation flow."""

    system_prompt = ORACLE_SYSTEM_PROMPT + """

## Computation chain
Shows the dependency flow for the target formula. Referenced cells that are themselves \
formulas are shown with both their formula text and computed value (formula \u2192 value). \
This reveals intermediate computation steps leading to the target cell."""

    def build(self, problem: Problem) -> str:
        wb, ws = self._open_worksheet(problem)
        wb_data = load_workbook(
            str(problem.workbook_path), read_only=True, data_only=True
        )
        ws_data = wb_data[problem.worksheet_name]
        try:
            context = self._build_context(wb, ws, ws_data, problem)

            target_cell = problem.active_cell
            target_formula = problem.formula

            ref_pat = re.compile(r"\$?([A-Z]{1,3})\$?(\d+)")
            direct_refs = set()
            for m in ref_pat.finditer(target_formula):
                addr = f"{m.group(1)}{m.group(2)}"
                if addr != target_cell:
                    direct_refs.add(addr)

            chain: list[tuple[str, str | None, str | None, int]] = []
            visited = set()

            def _trace(addr: str, depth: int):
                if addr in visited or depth > 3:
                    return
                visited.add(addr)
                try:
                    r, c = coordinate_to_tuple(addr)
                except Exception:
                    return
                cell = ws.cell(row=r, column=c)
                raw = normalize_value(cell)
                if raw is not None and str(raw).startswith("="):
                    formula_text = str(raw)
                    data_cell = ws_data.cell(row=r, column=c)
                    val = _format_value(data_cell.value) if data_cell.value is not None else "(no value)"
                    chain.append((addr, formula_text, val, depth))
                    for m in ref_pat.finditer(formula_text):
                        sub_addr = f"{m.group(1)}{m.group(2)}"
                        if sub_addr != target_cell:
                            _trace(sub_addr, depth + 1)
                else:
                    val = _format_value(raw) if raw else "(empty)"
                    chain.append((addr, None, val, depth))

            for ref in sorted(direct_refs):
                _trace(ref, 0)

            if chain:
                chain.sort(key=lambda x: (-x[3], coordinate_to_tuple(x[0]) if _is_valid(x[0]) else (9999, 9999)))

                ctx_lines = context.split("\n")
                ins = len(ctx_lines)
                for i, line in enumerate(ctx_lines):
                    if line.startswith("Data ranges:") or line.startswith("Named ranges:"):
                        ins = i
                        break

                block = ["Computation chain:"]
                for addr, formula_text, val, depth in chain:
                    indent = "  " + "  " * depth
                    if formula_text:
                        block.append(f"{indent}{addr} = {formula_text} \u2192 {val}")
                    else:
                        block.append(f"{indent}{addr} = {val}")
                block.append(f"  \u2192 {target_cell} = [TARGET]")
                block.append("")

                for j, line in enumerate(block):
                    ctx_lines.insert(ins + j, line)
                context = "\n".join(ctx_lines)

            return context
        finally:
            wb.close()
            wb_data.close()


class OracleSameColumnContextBuilder(OracleContextBuilder):
    """Oracle context plus all formulas in the same column as the target, with computed values."""

    system_prompt = ORACLE_SYSTEM_PROMPT + """

## Same-column formulas
All formulas in the same column as the target cell, with their computed values. \
These are likely computing the same thing for different rows — use them as templates."""

    def build(self, problem: Problem) -> str:
        wb, ws = self._open_worksheet(problem)
        wb_data = load_workbook(
            str(problem.workbook_path), read_only=True, data_only=True
        )
        ws_data = wb_data[problem.worksheet_name]
        try:
            context = self._build_context(wb, ws, ws_data, problem)
            target_cell = problem.active_cell
            target_row, target_col = coordinate_to_tuple(target_cell)

            col_formulas = []
            for row in ws.iter_rows(
                min_row=ws.min_row, max_row=ws.max_row,
                min_col=target_col, max_col=target_col,
            ):
                for cell in row:
                    if cell.row == target_row or cell.data_type != "f":
                        continue
                    raw = normalize_value(cell)
                    if raw is None or not str(raw).startswith("="):
                        continue
                    formula_text = str(raw)
                    addr = f"{get_column_letter(cell.column)}{cell.row}"
                    dv = ws_data.cell(row=cell.row, column=cell.column).value
                    val = _format_value(dv) if dv is not None else "(no value)"
                    col_formulas.append((addr, formula_text, val))

            if col_formulas:
                ctx_lines = context.split("\n")
                ins = len(ctx_lines)
                for i, line in enumerate(ctx_lines):
                    if line.startswith("Data ranges:") or line.startswith("Named ranges:"):
                        ins = i
                        break
                block = ["Same-column formulas:"]
                for addr, ft, val in col_formulas[:20]:
                    block.append(f"  {addr} | {ft} \u2192 {val}")
                if len(col_formulas) > 20:
                    block.append(f"  ... ({len(col_formulas) - 20} more)")
                block.append("")
                for j, line in enumerate(block):
                    ctx_lines.insert(ins + j, line)
                context = "\n".join(ctx_lines)

            return context
        finally:
            wb.close()
            wb_data.close()


class OracleGridNeighborContextBuilder(OracleContextBuilder):
    """Oracle context plus a mini spreadsheet grid view centered on the target cell."""

    system_prompt = ORACLE_SYSTEM_PROMPT + """

## Grid neighborhood
A mini spreadsheet view showing all cells within \u00b12 rows and \u00b13 columns of the target. \
Each cell shows its address, displayed value, and formula (if any). \
Use the spatial layout to understand patterns in the surrounding area."""

    def build(self, problem: Problem) -> str:
        wb, ws = self._open_worksheet(problem)
        wb_data = load_workbook(
            str(problem.workbook_path), read_only=True, data_only=True
        )
        ws_data = wb_data[problem.worksheet_name]
        try:
            context = self._build_context(wb, ws, ws_data, problem)
            target_cell = problem.active_cell
            target_row, target_col = coordinate_to_tuple(target_cell)

            min_r = max(1, target_row - 2)
            max_r = target_row + 2
            min_c = max(1, target_col - 3)
            max_c = target_col + 3

            grid_lines = []
            for r in range(min_r, max_r + 1):
                row_cells = []
                for c in range(min_c, max_c + 1):
                    addr = f"{get_column_letter(c)}{r}"
                    if addr == target_cell:
                        row_cells.append(f"{addr}=[TARGET]")
                        continue
                    cell = ws.cell(row=r, column=c)
                    raw = normalize_value(cell)
                    data_val = ws_data.cell(row=r, column=c).value

                    if raw is not None and str(raw).startswith("="):
                        formula_text = str(raw)
                        val = _format_value(data_val) if data_val is not None else ""
                        row_cells.append(f"{addr}={formula_text}\u2192{val}")
                    elif raw is not None and raw != "":
                        val = _format_value(raw)
                        row_cells.append(f"{addr}={val}")
                    else:
                        row_cells.append(f"{addr}=(empty)")
                grid_lines.append(" | ".join(row_cells))

            if grid_lines:
                ctx_lines = context.split("\n")
                ins = len(ctx_lines)
                for i, line in enumerate(ctx_lines):
                    if line.startswith("Data ranges:") or line.startswith("Named ranges:"):
                        ins = i
                        break
                block = ["Grid neighborhood:"]
                for gl in grid_lines:
                    block.append(f"  {gl}")
                block.append("")
                for j, line in enumerate(block):
                    ctx_lines.insert(ins + j, line)
                context = "\n".join(ctx_lines)

            return context
        finally:
            wb.close()
            wb_data.close()


class OracleWithAllFormulasContextBuilder(OracleContextBuilder):
    """Compressed oracle context plus all formulas in the sheet, grouped by structural pattern."""

    system_prompt = ORACLE_SYSTEM_PROMPT + """

## All sheet formulas (grouped by pattern)
Every formula in the worksheet (not just those referenced by the target), grouped \
by structural pattern. Row numbers in cell references are replaced with @ so that \
structurally identical formulas are merged. Use these to understand the sheet's \
formula landscape and common patterns."""

    def build(self, problem: Problem) -> str:
        from openpyxl.worksheet.formula import ArrayFormula
        from openpyxl.utils.cell import get_column_letter as _gl

        wb, ws = self._open_worksheet(problem)
        wb_data = load_workbook(
            str(problem.workbook_path), read_only=True, data_only=True
        )
        ws_data = wb_data[problem.worksheet_name]
        try:
            context = self._build_context(wb, ws, ws_data, problem)

            target_cell = problem.active_cell

            # Scan the full worksheet for all formula cells
            all_formula_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for row in ws.iter_rows(
                min_row=ws.min_row, max_row=ws.max_row,
                min_col=ws.min_column, max_col=ws.max_column,
            ):
                for cell in row:
                    addr = f"{_gl(cell.column)}{cell.row}"
                    if addr == target_cell:
                        continue
                    if cell.data_type != "f":
                        continue
                    raw = cell.value
                    if isinstance(raw, ArrayFormula):
                        formula_text = raw.text
                    elif raw is not None:
                        formula_text = str(raw)
                    else:
                        continue
                    if formula_text and not formula_text.startswith("="):
                        formula_text = f"={formula_text}"
                    if not formula_text:
                        continue
                    pattern = _group_formula(formula_text)
                    all_formula_groups[pattern].append((addr, formula_text))

            if all_formula_groups:
                ctx_lines = context.split("\n")
                ins = len(ctx_lines)
                for i, line in enumerate(ctx_lines):
                    if line.startswith("Data ranges:") or line.startswith("Named ranges:"):
                        ins = i
                        break
                block = ["All sheet formulas (grouped by pattern):"]
                for pattern, entries in sorted(
                    all_formula_groups.items(),
                    key=lambda x: _sort_key([a for a, _ in x[1]]),
                ):
                    addrs = [a for a, _ in entries]
                    compressed = ",".join(_compress_addresses(addrs))
                    if len(entries) == 1:
                        block.append(f"  {compressed} | {entries[0][1]}")
                    else:
                        block.append(f"  {compressed} | {pattern}")
                block.append("")
                for j, line in enumerate(block):
                    ctx_lines.insert(ins + j, line)
                context = "\n".join(ctx_lines)

            return context
        finally:
            wb.close()
            wb_data.close()


class OracleWithFormulaValueContextBuilder(OracleContextBuilder):
    """Compressed oracle context where formula cells show both formula text and computed value."""

    system_prompt = ORACLE_SYSTEM_PROMPT.replace(
        "## Formulas (grouped by pattern)\n"
        "Other formula cells in the referenced area. Formulas with the same structure are grouped, \n"
        "with row numbers replaced by @. Use these as templates.",
        "## Formulas (with computed values)\n"
        "Other formula cells in the referenced area. Each line shows the formula and its \n"
        "computed value (formula \u2192 value). Use these to understand intermediate computations.",
    )

    def _build_context(self, wb, ws, ws_data, problem: Problem) -> str:
        formula = problem.formula
        target_cell = problem.active_cell
        sheet_name = problem.worksheet_name

        intents = self._load_intents()
        intent_data = intents.get(problem.id, {})
        intent_desc = intent_data.get("intent", "")
        direct_intent = set(intent_data.get("direct_intent", []))
        data_desc = set(intent_data.get("data_description", []))
        annotated_ranges = intent_data.get("data_range", [])

        referenced_cells, formula_ranges, has_cross = _extract_refs_and_ranges(formula)
        cross_refs = _extract_cross_sheet_refs(formula) if has_cross else {}
        all_ranges = annotated_ranges or formula_ranges
        named_ranges = _extract_named_ranges(wb)

        all_addrs = referenced_cells | direct_intent | data_desc
        all_addrs.discard(target_cell)

        string_groups: dict[str, list[str]] = defaultdict(list)
        number_groups: dict[str, list[str]] = defaultdict(list)
        formula_entries: list[tuple[str, str, str]] = []  # (addr, formula, display_value)
        intent_cells: list[tuple[str, str, str]] = []

        for addr in sorted(all_addrs, key=lambda a: coordinate_to_tuple(a) if _is_valid(a) else (9999, 9999)):
            value, cell_formula, raw = self._read_cell(ws, ws_data, addr, target_cell)

            role = None
            if addr in direct_intent:
                role = "output_header"
            elif addr in data_desc:
                role = "input_header"

            if role:
                display = _format_value(value) if value is not None else "(empty)"
                intent_cells.append((addr, display, role))
                continue

            if cell_formula:
                display = _format_value(value) if value is not None else "(no value)"
                formula_entries.append((addr, cell_formula, display))
                continue

            if value is None or (isinstance(value, str) and value.strip() == ""):
                continue

            if isinstance(value, str):
                string_groups[value].append(addr)
                continue

            display = _format_value(value)
            number_groups[display].append(addr)

        # Build output
        sections = []
        sections.append(f"Sheet: {sheet_name}")
        sections.append(f"Target cell: {target_cell}")
        if intent_desc:
            sections.append(f"Intent: {intent_desc}")

        annotation = problem.annotation or {}
        function = annotation.get("function") or ""
        if not function:
            funcs = _FUNC_NAME_PATTERN.findall(problem.formula or "")
            clean = [f.upper().replace("_XLFN.", "").replace("_XLWS.", "")
                     for f in funcs if not f.startswith("_")]
            function = clean[0] if clean else ""
        if function and function != "ABSTAIN":
            hint = _extract_formula_hint(problem.formula, function)
            if hint:
                sections.append(f"Consider using: {hint}")

        sections.append("")

        if intent_cells:
            sections.append("Intent cells:")
            for addr, val, role in intent_cells:
                sections.append(f"  {addr} | {val} | {role}")
            sections.append("")

        sections.append("Referenced cells (grouped by value):")
        for val, addrs in sorted(string_groups.items(), key=lambda x: _sort_key(x[1])):
            compressed = ",".join(_compress_addresses(addrs))
            sections.append(f"  {compressed} | {val}")
        for val, addrs in sorted(number_groups.items(), key=lambda x: _sort_key(x[1])):
            compressed = ",".join(_compress_addresses(addrs))
            sections.append(f"  {compressed} | {val}")

        if formula_entries:
            sections.append("")
            sections.append("Formulas (with computed values):")
            for addr, f_text, f_val in formula_entries:
                sections.append(f"  {addr} | {f_text} \u2192 {f_val}")

        sections.append("")
        sections.append(f"{target_cell} | [TARGET]")

        if cross_refs:
            sections.append("")
            sections.append("Cross-sheet references:")
            for sheet, addrs in sorted(cross_refs.items()):
                if sheet not in wb.sheetnames:
                    continue
                cross_ws = wb[sheet]
                cross_ws_data = None
                wb_data_cross = None
                try:
                    wb_data_cross = load_workbook(
                        str(problem.workbook_path), read_only=True, data_only=True
                    )
                    cross_ws_data = wb_data_cross[sheet]
                except Exception:
                    pass
                for addr in sorted(addrs):
                    try:
                        r, c = coordinate_to_tuple(addr)
                        cell = cross_ws.cell(row=r, column=c)
                        raw = normalize_value(cell)
                        if raw is not None and str(raw).startswith("="):
                            f_text = str(raw)
                            val = "(no value)"
                            if cross_ws_data:
                                dv = cross_ws_data.cell(row=r, column=c).value
                                val = _format_value(dv) if dv is not None else "(no value)"
                            sections.append(f"  '{sheet}'!{addr} | {f_text} \u2192 {val}")
                        else:
                            val = _format_value(raw) if raw else "(empty)"
                            sections.append(f"  '{sheet}'!{addr} | {val}")
                    except Exception:
                        pass
                if wb_data_cross:
                    wb_data_cross.close()

        if all_ranges:
            sections.append("")
            sections.append(f"Data ranges: {', '.join(all_ranges)}")

        if named_ranges:
            sections.append(f"Named ranges: {', '.join(f'{n} \u2192 {r}' for n, r in sorted(named_ranges.items()))}")

        return "\n".join(sections)


class OracleKitchenSinkContextBuilder(OracleWithFormulaValueContextBuilder):
    """Stack ALL sheet-available signals: formula-value pairs, dep chain,
    all sheet formulas, cell format, and all-functions hint."""

    system_prompt = OracleWithFormulaValueContextBuilder.system_prompt + """

## Computation chain
Shows the dependency flow for the target formula with computed values.

## All sheet formulas (grouped by pattern)
Every formula in the worksheet grouped by structural pattern.

## Expected format
The number format of the target cell (if applicable)."""

    def build(self, problem: Problem) -> str:
        from openpyxl.worksheet.formula import ArrayFormula
        from openpyxl.utils.cell import get_column_letter as _gl

        wb, ws = self._open_worksheet(problem)
        wb_data = load_workbook(
            str(problem.workbook_path), read_only=True, data_only=True
        )
        ws_data = wb_data[problem.worksheet_name]
        try:
            context = self._build_context(wb, ws, ws_data, problem)
            target_cell = problem.active_cell

            # --- All-functions hint ---
            all_funcs = _FUNC_NAME_PATTERN.findall(problem.formula or "")
            clean_fns = []
            seen_fns = set()
            for f in all_funcs:
                name = f.upper().replace("_XLFN.", "").replace("_XLWS.", "")
                if name.startswith("_") or name in seen_fns or name == "ABSTAIN":
                    continue
                seen_fns.add(name)
                clean_fns.append(name)
            if len(clean_fns) > 1:
                all_funcs_str = ", ".join(clean_fns)
                lines = context.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("Consider using:"):
                        lines[i] = f"Consider using: {all_funcs_str}"
                        context = "\n".join(lines)
                        break

            # --- Cell format ---
            try:
                r, c = coordinate_to_tuple(target_cell)
                cell = ws.cell(row=r, column=c)
                label = OracleWithCellFormatContextBuilder._classify_format(cell.number_format)
            except Exception:
                label = None
            if label:
                lines = context.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("Target cell:"):
                        lines.insert(i + 1, f"Expected format: {label}")
                        context = "\n".join(lines)
                        break

            # --- Dependency chain ---
            ref_pat = re.compile(r"\$?([A-Z]{1,3})\$?(\d+)")
            direct_refs = set()
            for m in ref_pat.finditer(problem.formula):
                addr = f"{m.group(1)}{m.group(2)}"
                if addr != target_cell:
                    direct_refs.add(addr)

            chain = []
            visited = set()

            def _trace(addr, depth):
                if addr in visited or depth > 3:
                    return
                visited.add(addr)
                try:
                    r2, c2 = coordinate_to_tuple(addr)
                except Exception:
                    return
                cell2 = ws.cell(row=r2, column=c2)
                raw2 = normalize_value(cell2)
                if raw2 is not None and str(raw2).startswith("="):
                    ftext = str(raw2)
                    dv = ws_data.cell(row=r2, column=c2).value
                    val = _format_value(dv) if dv is not None else "(no value)"
                    chain.append((addr, ftext, val, depth))
                    for m2 in ref_pat.finditer(ftext):
                        sub = f"{m2.group(1)}{m2.group(2)}"
                        if sub != target_cell:
                            _trace(sub, depth + 1)
                else:
                    val = _format_value(raw2) if raw2 else "(empty)"
                    chain.append((addr, None, val, depth))

            for ref in sorted(direct_refs):
                _trace(ref, 0)

            if chain:
                chain.sort(key=lambda x: (-x[3], coordinate_to_tuple(x[0]) if _is_valid(x[0]) else (9999, 9999)))
                ctx_lines = context.split("\n")
                ins = len(ctx_lines)
                for i, line in enumerate(ctx_lines):
                    if line.startswith("Data ranges:") or line.startswith("Named ranges:"):
                        ins = i
                        break
                block = ["Computation chain:"]
                for addr, ftext, val, depth in chain:
                    indent = "  " + "  " * depth
                    if ftext:
                        block.append(f"{indent}{addr} = {ftext} \u2192 {val}")
                    else:
                        block.append(f"{indent}{addr} = {val}")
                block.append(f"  \u2192 {target_cell} = [TARGET]")
                block.append("")
                for j, line in enumerate(block):
                    ctx_lines.insert(ins + j, line)
                context = "\n".join(ctx_lines)

            # --- All sheet formulas ---
            all_formula_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for row in ws.iter_rows(
                min_row=ws.min_row, max_row=ws.max_row,
                min_col=ws.min_column, max_col=ws.max_column,
            ):
                for cell3 in row:
                    addr3 = f"{_gl(cell3.column)}{cell3.row}"
                    if addr3 == target_cell or cell3.data_type != "f":
                        continue
                    raw3 = cell3.value
                    if isinstance(raw3, ArrayFormula):
                        ft = raw3.text
                    elif raw3 is not None:
                        ft = str(raw3)
                    else:
                        continue
                    if ft and not ft.startswith("="):
                        ft = f"={ft}"
                    if not ft:
                        continue
                    pattern = _group_formula(ft)
                    all_formula_groups[pattern].append((addr3, ft))

            if all_formula_groups:
                ctx_lines = context.split("\n")
                ins = len(ctx_lines)
                for i, line in enumerate(ctx_lines):
                    if line.startswith("Data ranges:") or line.startswith("Named ranges:"):
                        ins = i
                        break
                block2 = ["All sheet formulas (grouped by pattern):"]
                for pattern, entries in sorted(
                    all_formula_groups.items(),
                    key=lambda x: _sort_key([a for a, _ in x[1]]),
                ):
                    addrs = [a for a, _ in entries]
                    compressed = ",".join(_compress_addresses(addrs))
                    if len(entries) == 1:
                        block2.append(f"  {compressed} | {entries[0][1]}")
                    else:
                        block2.append(f"  {compressed} | {pattern}")
                block2.append("")
                for j, line in enumerate(block2):
                    ctx_lines.insert(ins + j, line)
                context = "\n".join(ctx_lines)

            return context
        finally:
            wb.close()
            wb_data.close()


class OracleWithNLPromptContextBuilder(OracleContextBuilder):
    """Oracle context plus a natural-language formula instruction.

    Adds a detailed NL description of the formula, e.g.:
    'Formula instruction: Write a SUMIF that sums C2:C10 where B2:B10 equals East.'
    """

    system_prompt = ORACLE_NL_SYSTEM_PROMPT

    def __init__(self):
        super().__init__()
        self._nl_cache: dict[str, str] | None = None

    def _load_nl_prompts(self) -> dict[str, str]:
        if self._nl_cache is not None:
            return self._nl_cache
        cache: dict[str, str] = {}
        if NL_PROMPTS_FILE.exists():
            with open(NL_PROMPTS_FILE, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    nl = rec.get("nl_prompt") or ""
                    if nl:
                        cache[rec["id"]] = nl
        self._nl_cache = cache
        return cache

    def build(self, problem: Problem) -> str:
        base_context = super().build(problem)

        nl_prompts = self._load_nl_prompts()
        nl = nl_prompts.get(problem.id, "")
        if not nl:
            return base_context

        lines = base_context.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("Intent:") or line.startswith("Target cell:"):
                insert_idx = i + 1

        lines.insert(insert_idx, f"Formula instruction: {nl}")
        return "\n".join(lines)
