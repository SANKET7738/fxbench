from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable

_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_path_component(value: str, field_name: str) -> str:
    component = str(value)
    if component in {"", ".", ".."} or _SAFE_PATH_COMPONENT_RE.fullmatch(component) is None:
        raise ValueError(
            f"{field_name} must be a single path component containing only letters, "
            "numbers, '.', '_' or '-'"
        )
    return component


def is_latin_string(text: str, threshold: float = 0.7) -> bool:
    latin = non_latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            if "LATIN" in unicodedata.name(ch, ""):
                latin += 1
            else:
                non_latin += 1
        except ValueError:
            non_latin += 1
    total = latin + non_latin
    if total == 0:
        return True
    return latin / total > threshold


def is_predominantly_english(strings: Iterable[str], threshold: float = 0.6) -> bool:
    items = list(strings)
    if not items:
        return True
    latin = sum(1 for s in items if is_latin_string(s))
    return latin / len(items) >= threshold


def get_formula_text(cell) -> str | None:
    if cell.data_type != "f":
        return None
    raw = cell.value
    if hasattr(raw, "text"):
        text = raw.text
    elif isinstance(raw, str):
        text = raw
    else:
        return None
    if text and text.startswith("="):
        text = text[1:]
    if not text or text == "None":
        return None
    return text

def read_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data
