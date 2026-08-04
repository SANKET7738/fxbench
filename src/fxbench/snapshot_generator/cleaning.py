from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fxbench.llm_utils import ChatHistory, LLMClient
from fxbench.snapshot_generator.dense_representation import create_dense_representation
from fxbench.snapshot_generator.workbook_ops import create_sheet_with_ranges_deleted


SNAPSHOT_CLEANING_PROMPT = """
You are an expert Excel analyst specializing in spreadsheet workflow reconstruction. Your task is to identify cell ranges to DELETE to create a historical snapshot of a spreadsheet-representing the state BEFORE a specific formula was written.

## Context
You are given:
1. A markdown representation of a spreadsheet
2. An active cell containing a formula

## Goal
Return cell ranges to DELETE so the remaining spreadsheet represents the plausible state when the user was ABOUT to write the active cell formula. This generates training data for formula completion models.

## Core Assumptions
1. Users populate spreadsheets top-to-bottom, left-to-right
2. Users enter raw data BEFORE writing formulas that reference them
3. Users complete earlier time periods before later ones (left columns before right columns)

## Deletion Rules
Delete cells which would logically should be written after the active cell formula.

Use headers and labels to identify the active cell's role (e.g., "Total", "Sum", "Percent", "Growth"). Delete cells sharing that same role that would be authored after the active cell.

Authoring order:
- Reading order: same row to the right, then rows below
- Temporal order: if columns represent time periods, later periods come later

Example: If active cell is in "Total" column at row 10, delete other "Total" column cells in rows 11+, and any "Total" cells in later time period columns.

## Preservation Rules (Never Delete)
- Cells directly or indirectly referenced by the active cell formula
- Row/column headers that provide context for understanding the formula
- Data regions that the formula operates on
- Raw input data regions
- Cells in different logical roles (different column/row purpose)

## Output Format
Return ONLY valid JSON with no additional text:

{
    "regions_to_delete": ["<range1>", "<range2>"]
}

## Range Specification Rules
- Use maximally merged ranges where possible
- Use Excel-style notation (e.g., "A10:D20")
- Return empty array [] if nothing should be deleted

## Critical Reminders
- NEVER delete cells referenced by the active cell formula
- Temporal columns (future years/periods) should be deleted ENTIRELY within data regions
- When uncertain, prefer preservation over deletion
- Consider both row AND column dimensions for deletion
"""


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    if not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def extract_json_text(text: str) -> str:
    stripped = strip_markdown_fences(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]


def parse_json_object(text: str, context: str) -> dict[str, Any]:
    json_text = extract_json_text(text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse {context} as JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"Expected {context} to be a JSON object")
    return payload


def parse_string_list_field(payload: dict[str, Any], field_name: str, context: str) -> list[str]:
    if field_name not in payload:
        raise ValueError(f"Missing '{field_name}' in {context}")

    value = payload[field_name]
    if not isinstance(value, list):
        raise ValueError(f"Expected '{field_name}' in {context} to be a list")

    values = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Expected all '{field_name}' entries in {context} to be strings")
        values.append(item)
    return values


def build_cleaning_messages(dense_representation: str, active_cell: str) -> list[dict]:
    chat_history = ChatHistory()
    chat_history.add_message("system", SNAPSHOT_CLEANING_PROMPT)
    chat_history.add_message("user", f"Markdown Representation:\n{dense_representation}\n\nActive Cell: {active_cell}")
    return chat_history.get_message_history()


def parse_cleaning_response(response_text: str) -> list[str]:
    payload = parse_json_object(response_text, "snapshot cleaning response")
    return parse_string_list_field(payload, "regions_to_delete", "snapshot cleaning response")


def get_ranges_to_delete(
    filepath: str | Path,
    sheet_name: str,
    active_cell: str,
    *,
    model: str | None = None,
    client: LLMClient | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[str]:
    llm_client = client or LLMClient()
    dense_representation = create_dense_representation(filepath, sheet_name)
    result = llm_client.prompt_llm(
        model=model,
        message_history=build_cleaning_messages(dense_representation, active_cell),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return parse_cleaning_response(result["response"])


def create_cleaned_snapshot(
    filepath: str | Path,
    sheet_name: str,
    active_cell: str,
    output_dir: str | Path,
    *,
    model: str | None = None,
    client: LLMClient | None = None,
    include_summary_sheet: bool = True,
) -> Path:
    ranges_to_delete = get_ranges_to_delete(
        filepath,
        sheet_name,
        active_cell,
        model=model,
        client=client,
    )
    return create_sheet_with_ranges_deleted(
        filepath,
        sheet_name,
        ranges_to_delete,
        output_dir,
        include_summary_sheet=include_summary_sheet,
    )
