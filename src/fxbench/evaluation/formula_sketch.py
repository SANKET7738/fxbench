"""Evaluation-time formula sketch normalization."""

from __future__ import annotations

import re


_STRING_RE = re.compile(r'"(?:[^"]|"")*"')


def formula_to_metric_sketch(formula: str | None) -> str:
    """Return a normalized formula sketch for evaluation metrics."""

    if formula is None:
        return ""
    s = _uppercase_outside_strings(_strip_formula_prefixes(str(formula).strip()))
    if s.startswith("{=") and s.endswith("}"):
        s = "=" + s[2:-1].strip()

    s = _STRING_RE.sub("STR", s)
    s = re.sub(r"'[^']+'!", "SHEET!", s)
    s = re.sub(r"[A-Z_]\w*!", "SHEET!", s)

    s = re.sub(r"[A-Z_]\w*\[@\[[^\]]*\]\]", "TABLE[@COL]", s)
    s = re.sub(r"[A-Z_]\w*\[\[[^\]]*\]\]", "TABLE[COL]", s)
    s = re.sub(r"[A-Z_]\w*\[@[^\]]*\]", "TABLE[@COL]", s)
    s = re.sub(r"[A-Z_]\w*\[[^\]]*\]", "TABLE[COL]", s)

    s = re.sub(r"\$?[A-Z]{1,3}\$?\d{1,7}\s*:\s*\$?[A-Z]{1,3}\$?\d{1,7}", "CELL:CELL", s)
    s = re.sub(r"\$?[A-Z]{1,3}\s*:\s*\$?[A-Z]{1,3}", "CELL:CELL", s)
    s = re.sub(r"\$?\d{1,7}\s*:\s*\$?\d{1,7}", "CELL:CELL", s)
    s = re.sub(r"\$?[A-Z]{1,3}\$?\d{1,7}", "CELL", s)
    s = re.sub(r"\b\d+\.?\d*\b", "NUM", s)
    return s


def _strip_formula_prefixes(formula: str) -> str:
    return (
        formula.replace("_xlfn.", "")
        .replace("_XLFN.", "")
        .replace("_xlws.", "")
        .replace("_XLWS.", "")
    )


def _uppercase_outside_strings(formula: str) -> str:
    parts: list[str] = []
    pos = 0
    for match in _STRING_RE.finditer(formula):
        parts.append(formula[pos:match.start()].upper())
        parts.append(match.group(0))
        pos = match.end()
    parts.append(formula[pos:].upper())
    return "".join(parts)
