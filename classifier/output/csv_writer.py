import json
import logging
from pathlib import Path

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from classifier.models import PairPipelineResult, PipelineResult

logger = logging.getLogger(__name__)

# Maps column -> set of values that mean "no issue".
# A row is clean only when ALL listed columns hold a clean value.
# Add a new entry here to extend issue detection to future checks.
_CLEAN_VALUES: dict[str, set[object]] = {
    "success": {True},
    "overall": {"MATCH"},
    "errors": {"", None},
}

# Fixed patient metadata columns in output order
_META_COLUMNS = [
    "patient_name",
    "dob",
    "age",
    "sex",
    "account_number",
    "dos",
    "phone",
    "address",
]

_PAIR_COLUMNS_BASE = [
    "dr_file_path", "nurse_file_path", "success", "overall", "clinical_verdict",
    "clinical_reasoning", "errors",
]
_PAIR_COLUMNS_VERBOSE = [
    "dr_file_path", "nurse_file_path", "success", "overall", "clinical_verdict",
    "clinical_reasoning", "identity_match", "dr_fields", "nurse_fields", "errors",
]


def _row_has_issue(row: dict[str, object]) -> bool:
    """Return True if row fails any clean-value check."""
    return any(row.get(col) not in clean for col, clean in _CLEAN_VALUES.items())


def _write_pair_sheet(ws: Worksheet, rows: list[dict[str, object]], columns: list[str]) -> None:
    ws.append(columns)
    for row in rows:
        ws.append([row.get(col, "") for col in columns])


def write_pair_xlsx(results: list[PairPipelineResult], output_path: Path, *, verbose: bool = False) -> None:
    """Write pair pipeline results to a 3-sheet Excel workbook.

    Sheets: All (every row), No Issues (clean rows), Issues (rows with problems).
    Pass verbose=True to include identity_match, dr_fields, nurse_fields columns.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _PAIR_COLUMNS_VERBOSE if verbose else _PAIR_COLUMNS_BASE
    rows = [_pair_result_to_row(r, verbose=verbose) for r in results]

    clean_rows = [r for r in rows if not _row_has_issue(r)]
    issue_rows = [r for r in rows if _row_has_issue(r)]

    wb = openpyxl.Workbook()
    wb.active.title = "All"  # type: ignore[union-attr]
    _write_pair_sheet(wb.active, rows, columns)  # type: ignore[arg-type]

    _write_pair_sheet(wb.create_sheet("No Issues"), clean_rows, columns)
    _write_pair_sheet(wb.create_sheet("Issues"), issue_rows, columns)

    wb.save(output_path)
    logger.info(
        "Pair XLSX written to %s (%d total, %d clean, %d issues)",
        output_path,
        len(rows),
        len(clean_rows),
        len(issue_rows),
    )


def _pair_result_to_row(result: PairPipelineResult, *, verbose: bool = False) -> dict[str, object]:
    base: dict[str, object] = {
        "dr_file_path": str(result.dr_file_path),
        "nurse_file_path": str(result.nurse_file_path),
        "success": result.success,
        "errors": result.error or "",
    }
    if not result.success or result.result is None:
        base.update({"overall": "", "clinical_verdict": "", "clinical_reasoning": ""})
        if verbose:
            base.update({"identity_match": "", "dr_fields": "", "nurse_fields": ""})
        return base

    r = result.result
    base.update({
        "overall": r.overall,
        "clinical_verdict": r.clinical_verdict,
        "clinical_reasoning": r.clinical_reasoning,
    })
    if verbose:
        base.update({
            "identity_match": json.dumps(r.identity_match.model_dump(), default=str),
            "dr_fields": json.dumps(r.dr_metadata.model_dump(exclude={"file_path", "raw_text"}), default=str),
            "nurse_fields": json.dumps(r.nurse_fields.model_dump(), default=str),
        })
    return base


def write_xlsx(results: list[PipelineResult], output_path: Path, task_names: list[str]) -> None:
    """Write pipeline results to an Excel workbook (single sheet).

    Columns: file_path, success, category, <one per task>, <patient meta fields>, errors
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["file_path", "success", "category"] + task_names + _META_COLUMNS + ["errors"]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"  # type: ignore[union-attr]
    ws.append(columns)  # type: ignore[union-attr]
    for result in results:
        row = _result_to_row(result, task_names)
        ws.append([row.get(col, "") for col in columns])  # type: ignore[union-attr]

    wb.save(output_path)
    logger.info("XLSX written to %s (%d rows)", output_path, len(results))


def _result_to_row(result: PipelineResult, task_names: list[str]) -> dict[str, object]:
    base: dict[str, object] = {
        "file_path": str(result.file_path),
        "success": result.success,
        "errors": result.error or "",
    }

    if not result.success or result.result is None:
        base["category"] = ""
        for task in task_names:
            base[task] = ""
        for col in _META_COLUMNS:
            base[col] = ""
        return base

    meta = result.result.metadata.meta
    base["category"] = result.result.category
    for task in task_names:
        base[task] = result.result.task_results.get(task, "")
    for col in _META_COLUMNS:
        base[col] = getattr(meta, col) or ""
    return base
