import json
import logging
from pathlib import Path
from urllib.parse import quote

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from classifier.models import (
    Diagnosis,
    DiagnosisCheck,
    DiagnosisExtractionResult,
    PairPipelineResult,
    PipelineResult,
    TreatmentRequestResult,
)

logger = logging.getLogger(__name__)

# Maps column -> set of values that mean "no issue".
# A row is clean only when ALL listed columns hold a clean value.
# Add a new entry here to extend issue detection to future checks.
_CLEAN_VALUES: dict[str, set[object]] = {
    "success": {True},
    "overall": {"MATCH"},
    "errors": {"", None},
    # "" = check did not apply to this pair's category.
    "diagnosis_check": {"", DiagnosisCheck.EXISTS},
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
    "folder", "patient_name", "dr_file_path", "nurse_file_path", "success", "overall",
    "clinical_verdict", "clinical_reasoning", "diagnosis_check", "diagnoses", "errors",
]
_PAIR_COLUMNS_VERBOSE = [
    "folder", "patient_name", "dr_file_path", "nurse_file_path", "success", "overall",
    "clinical_verdict", "clinical_reasoning", "diagnosis_check", "diagnoses",
    "identity_match", "dr_fields", "nurse_fields",
    "errors",
]


def format_diagnoses(diagnoses: list[Diagnosis]) -> str:
    """Render diagnoses as proof text: 'dr: name (ICD); nurse: name'."""
    parts = []
    for d in diagnoses:
        name = f"{d.name} ({d.icd_code})" if d.icd_code else d.name
        parts.append(f"{d.source}: {name}" if d.source else name)
    return "; ".join(parts)


def _row_has_issue(row: dict[str, object]) -> bool:
    """Return True if row fails any clean-value check."""
    return any(row.get(col) not in clean for col, clean in _CLEAN_VALUES.items())


def _write_pair_sheet(ws: Worksheet, rows: list[dict[str, object]], columns: list[str]) -> None:
    ws.append(columns)
    for row in rows:
        ws.append([row.get(col, "") for col in columns])


def write_pair_xlsx(
    results: list[PairPipelineResult],
    output_path: Path,
    *,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
) -> None:
    """Write pair pipeline results to a 3-sheet Excel workbook.

    Sheets: All (every row), No Issues (clean rows), Issues (rows with problems).
    Pass verbose=True to include identity_match, dr_fields, nurse_fields columns.
    Pass local_root + sp_web_url to show SharePoint paths instead of local tmp paths.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _PAIR_COLUMNS_VERBOSE if verbose else _PAIR_COLUMNS_BASE
    rows = [_pair_result_to_row(r, verbose=verbose, local_root=local_root, sp_web_url=sp_web_url) for r in results]

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


def _join_names(paths: list[Path], fallback: Path) -> str:
    """Join file names with '; '. Falls back to the representative path if empty."""
    return "; ".join(p.name for p in (paths or [fallback]))


def resolve_folder(file_path: Path, local_root: Path | None, sp_web_url: str | None) -> str:
    if local_root is not None and sp_web_url is not None:
        rel = file_path.parent.relative_to(local_root)
        if str(rel) == ".":
            return sp_web_url
        encoded = "/".join(quote(part, safe="") for part in rel.parts)
        return f"{sp_web_url.rstrip('/')}/{encoded}"
    return str(file_path.parent)


def _pair_result_to_row(
    result: PairPipelineResult,
    *,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
) -> dict[str, object]:
    base: dict[str, object] = {
        "folder": resolve_folder(result.dr_file_path, local_root, sp_web_url),
        "patient_name": result.dr_file_path.parent.name,
        "dr_file_path": _join_names(result.dr_paths, result.dr_file_path),
        "nurse_file_path": _join_names(result.nurse_paths, result.nurse_file_path),
        "success": result.success,
        "errors": result.error or "",
    }
    if not result.success or result.result is None:
        base.update({
            "overall": "", "clinical_verdict": "", "clinical_reasoning": "",
            "diagnosis_check": "", "diagnoses": "",
        })
        if verbose:
            base.update({"identity_match": "", "dr_fields": "", "nurse_fields": ""})
        return base

    r = result.result
    base.update({
        "overall": r.overall,
        "clinical_verdict": r.clinical_verdict,
        "clinical_reasoning": r.clinical_reasoning,
        "diagnosis_check": r.diagnosis_check or "",
        "diagnoses": format_diagnoses(r.diagnoses),
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


_DIAGNOSIS_COLUMNS = ["file_path", "diagnosis", "icd_code"]


def write_diagnosis_xlsx(
    results: list[DiagnosisExtractionResult], output_path: Path
) -> None:
    """Write diagnosis-extraction results to a single-sheet Excel workbook.

    One row per diagnosis (file_path, diagnosis, icd_code). A file with no
    diagnoses still gets one row with empty diagnosis/icd_code columns.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Diagnoses"
    ws.append(_DIAGNOSIS_COLUMNS)

    row_count = 0
    for result in results:
        file_path = str(result.file_path)
        if not result.diagnoses:
            ws.append([file_path, "", ""])
            row_count += 1
            continue
        for diagnosis in result.diagnoses:
            ws.append([file_path, diagnosis.name, diagnosis.icd_code or ""])
            row_count += 1

    wb.save(output_path)
    logger.info(
        "Diagnosis XLSX written to %s (%d files, %d rows)",
        output_path,
        len(results),
        row_count,
    )


_TREATMENT_COLUMNS = ["file_path", "treatment_requested", "reasoning"]


def write_treatment_request_xlsx(
    results: list[TreatmentRequestResult], output_path: Path
) -> None:
    """Write treatment-request results to a single-sheet Excel workbook.

    One row per file (file_path, patient_requested_treatment, reasoning).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "TreatmentRequest"
    ws.append(_TREATMENT_COLUMNS)
    for result in results:
        ws.append(
            [
                str(result.file_path),
                result.treatment_requested,
                result.reasoning or "",
            ]
        )

    wb.save(output_path)
    logger.info(
        "Treatment-request XLSX written to %s (%d rows)", output_path, len(results)
    )


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
