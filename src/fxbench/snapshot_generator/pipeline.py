from __future__ import annotations

from pathlib import Path

from fxbench.llm_utils import LLMClient
from fxbench.snapshot_generator.cleaning import create_cleaned_snapshot
from fxbench.snapshot_generator.focus import focused_regions_for_active_cell
from fxbench.snapshot_generator.segmentation import segment_sheet
from fxbench.snapshot_generator.workbook_ops import reconstruct_sheet_with_focused_regions


DEFAULT_SNAPSHOT_DIR = Path("Assets") / "Snapshots"


def generate_snapshot(
    filepath: str | Path,
    sheet_name: str,
    active_cell: str,
    *,
    output_dir: str | Path | None = None,
    model: str | None = None,
    client: LLMClient | None = None,
    include_summary_sheets: bool = True,
) -> Path:
    snapshot_dir = Path(output_dir) if output_dir is not None else DEFAULT_SNAPSHOT_DIR
    focused_output_dir = snapshot_dir / "focused_regions"
    cleaned_output_dir = snapshot_dir / "cleaned"
    llm_client = client or LLMClient()

    segmentation_result = segment_sheet(
        filepath,
        sheet_name,
        model=model,
        client=llm_client,
    )
    focused_regions = focused_regions_for_active_cell(
        filepath,
        sheet_name,
        active_cell,
        segmentation_result["regions"],
    )
    focused_path = reconstruct_sheet_with_focused_regions(
        filepath,
        sheet_name,
        focused_regions,
        focused_output_dir,
        include_summary_sheet=include_summary_sheets,
    )
    return create_cleaned_snapshot(
        focused_path,
        sheet_name,
        active_cell,
        cleaned_output_dir,
        model=model,
        client=llm_client,
        include_summary_sheet=include_summary_sheets,
    )


process_sheet = generate_snapshot
