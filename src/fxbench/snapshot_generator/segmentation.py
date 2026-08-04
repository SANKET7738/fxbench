from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fxbench.llm_utils import ChatHistory, LLMClient
from fxbench.snapshot_generator.dense_representation import create_dense_representation


SEGMENTATION_PROMPT = """
You are an expert Excel analyst. Your task is to analyse Excel sheet text representation and identify logical segments within the sheet.
You are provided with a markdown representation of the spreadsheet:

Your goal is to identify large, complete logical segments based on FUNCTIONAL PURPOSE.

Region Types:
- "data_table": "A structured table with headers containing organized rows and columns of data. The primary data structure."
- "calculation_section": "Area containing formulas, calculations, or computed results separate from main tables. Often labeled with descriptive headers."
- "sheet_title_header": "Main title or primary heading of the sheet."
- "text_description": "Explanatory text, notes, instructions, or descriptive content."
- "miscellaneous": "Any other content that doesn't fit the above categories."

Key Analysis Principles:
1. FUNCTIONAL SEPARATION: If headers indicate different business purposes, separate them
2. COMPLETE BOUNDARIES: Follow logical content to its natural end, don't stop arbitrarily
3. HEADER CLUES: Use section headers to understand logical groupings
4. SPATIAL AWARENESS: Consider layout but prioritize logical function
5. BUSINESS LOGIC: Think about how someone would build/use this sheet

Pattern Recognition:
- Look for distinct header groups that indicate separate purposes
- Identify calculation areas that serve different functions
- Recognize when data tables extend beyond visible breaks
- Separate standalone sections from main data structures

Requirements:
- Each region must have a unique integer 'id' starting from 0
- 'range' must be in standard Excel notation (e.g., "A1:G32")
- Cover ALL non-empty cells without overlap
- Prioritize logical separation over minimizing region count
- Include 'dependencies' field listing region IDs that this region depends on

CRITICAL: Return ONLY valid JSON with no additional text, explanations, or markdown formatting.

Output Format:
{
    "regions": [
        {
            "id": 0,
            "range": "A1:G32",
            "type": "data_table",
            "dependencies": []
        },
        {
            "id": 1,
            "range": "J7:K9",
            "type": "calculation_section",
            "dependencies": []
        }
    ]
}"""


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


def validate_regions(payload: dict[str, Any]) -> list[dict]:
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Segmentation response must contain a 'regions' list")

    validated_regions = []
    seen_ids = set()
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Every segmentation region must be an object")

        if "id" not in region:
            raise ValueError("Every segmentation region must include 'id'")
        if "range" not in region:
            raise ValueError("Every segmentation region must include 'range'")
        if "type" not in region:
            raise ValueError("Every segmentation region must include 'type'")

        region_id = region["id"]
        if not isinstance(region_id, int):
            raise ValueError("Segmentation region 'id' must be an integer")
        if region_id in seen_ids:
            raise ValueError(f"Duplicate segmentation region id: {region_id}")
        seen_ids.add(region_id)

        region_range = region["range"]
        if not isinstance(region_range, str) or not region_range:
            raise ValueError("Segmentation region 'range' must be a non-empty string")

        region_type = region["type"]
        if not isinstance(region_type, str) or not region_type:
            raise ValueError("Segmentation region 'type' must be a non-empty string")

        dependencies = region.get("dependencies", [])
        if dependencies is None:
            dependencies = []
        if not isinstance(dependencies, list):
            raise ValueError("Segmentation region 'dependencies' must be a list")

        normalized_dependencies = []
        for dependency in dependencies:
            if not isinstance(dependency, int):
                raise ValueError("Segmentation dependencies must be integers")
            normalized_dependencies.append(dependency)

        validated_regions.append({**region, "dependencies": normalized_dependencies})

    return validated_regions


def build_segmentation_messages(dense_representation: str) -> list[dict]:
    chat_history = ChatHistory()
    chat_history.add_message("system", SEGMENTATION_PROMPT)
    chat_history.add_message("user", f"Markdown Representation:\n{dense_representation}")
    return chat_history.get_message_history()


def parse_segmentation_response(response_text: str) -> dict[str, Any]:
    payload = parse_json_object(response_text, "segmentation response")
    regions = validate_regions(payload)
    return {**payload, "regions": regions}


def segment_sheet(
    filepath: str | Path,
    sheet_name: str | None = None,
    *,
    model: str | None = None,
    client: LLMClient | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict[str, Any]:
    llm_client = client or LLMClient()
    dense_representation = create_dense_representation(filepath, sheet_name)
    result = llm_client.prompt_llm(
        model=model,
        message_history=build_segmentation_messages(dense_representation),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    response_text = result["response"]
    segmentation = parse_segmentation_response(response_text)

    return {
        "success": True,
        "dense_representation": dense_representation,
        "regions": segmentation["regions"],
        "raw_response": response_text,
    }
