import json
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from classifier.classifiers.intake_verdict import dr_nurse_compare
from classifier.models import (
    CareCheck,
    Diagnosis,
    DiagnosisCategory,
    DiagnosisCheck,
    DiagnosisExtractionResult,
    DrValidityPipelineResult,
    IntakeReviewPipelineResult,
    IntakeVerdict,
    MainDiagnosisResult,
    PairPipelineResult,
    PatientGroup,
    PipelineResult,
    TreatmentRequestResult,
    ValidityVerdict,
    unique_categories,
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
    "medical_necessity": {"", CareCheck.NEEDED},
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
    "clinical_verdict", "clinical_reasoning", "diagnosis_check", "diagnoses",
    "diagnosis_categories", "medical_necessity", "medical_necessity_reasoning", "errors",
]
_PAIR_COLUMNS_VERBOSE = [
    "folder", "patient_name", "dr_file_path", "nurse_file_path", "success", "overall",
    "clinical_verdict", "clinical_reasoning", "diagnosis_check", "diagnoses",
    "diagnosis_categories", "medical_necessity", "medical_necessity_reasoning",
    "identity_match", "dr_fields", "nurse_fields",
    "errors",
]


def format_diagnoses(diagnoses: list[Diagnosis]) -> str:
    """Render diagnoses as proof text: 'name (ICD) [Category]; name [Category]'."""
    return "; ".join(_format_diagnosis(d) for d in diagnoses)


def _format_diagnosis(d: Diagnosis) -> str:
    text = f"{d.name} ({d.icd_code})" if d.icd_code else d.name
    return f"{text} [{d.category}]" if d.category else text


def format_categories(categories: list[DiagnosisCategory]) -> str:
    """Render a category list as one cell: 'Respiratory; Cardiovascular'."""
    return "; ".join(categories)


def format_pair_diagnoses(dr: list[Diagnosis], nurse: list[Diagnosis]) -> str:
    """Render a pair's diagnoses labelled by note: 'dr: name (ICD); nurse: name'."""
    return "; ".join(
        f"{label}: {format_diagnoses([d])}"
        for label, diagnoses in (("dr", dr), ("nurse", nurse))
        for d in diagnoses
    )


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
    """Write pair pipeline results to a 4-sheet Excel workbook.

    Sheets: All (every row), No Issues (clean rows), Issues (rows with problems) and
    All Categories (category of any dr-note diagnosis -> pair count).
    Each row lists ALL files of the pair in dr_file_path / nurse_file_path.
    Pass verbose=True to include identity_match, dr_fields, nurse_fields columns.
    Pass local_root to list those files relative to the scanned root (bare names
    otherwise); add sp_web_url to show SharePoint folder links instead of tmp paths.
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
    _write_counts_sheet(
        wb.create_sheet("All Categories"),
        "diagnosis_category",
        count_all_categories(
            [unique_categories(r.result.dr_diagnoses) if r.result else None for r in results]
        ),
    )

    wb.save(output_path)
    logger.info(
        "Pair XLSX written to %s (%d total, %d clean, %d issues)",
        output_path,
        len(rows),
        len(clean_rows),
        len(issue_rows),
    )


def relative_file_names(
    paths: list[Path], fallback: Path, local_root: Path | None = None
) -> list[str]:
    """Every file of one pair side, as paths relative to local_root.

    Falls back to the bare file name when the path lies outside local_root or no
    root is given, and to the representative path when the list is empty.
    """
    names = []
    for p in paths or [fallback]:
        if local_root is not None:
            try:
                names.append(p.relative_to(local_root).as_posix())
                continue
            except ValueError:
                pass
        names.append(p.name)
    return names


def _join_names(paths: list[Path], fallback: Path, local_root: Path | None = None) -> str:
    """Join a pair side's files with '; ' for a single spreadsheet cell."""
    return "; ".join(relative_file_names(paths, fallback, local_root))


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
        "dr_file_path": _join_names(result.dr_paths, result.dr_file_path, local_root),
        "nurse_file_path": _join_names(result.nurse_paths, result.nurse_file_path, local_root),
        "success": result.success,
        "errors": result.error or "",
    }
    if not result.success or result.result is None:
        base.update({
            "overall": "",
            "clinical_verdict": "", "clinical_reasoning": "",
            "diagnosis_check": "", "diagnoses": "", "diagnosis_categories": "",
            "medical_necessity": "", "medical_necessity_reasoning": "",
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
        "diagnoses": format_pair_diagnoses(r.dr_diagnoses, r.nurse_diagnoses),
        "diagnosis_categories": format_categories(r.diagnosis_categories),
        "medical_necessity": r.care_check or "",
        "medical_necessity_reasoning": r.care_reasoning,
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


_DIAGNOSIS_COLUMNS = ["file_path", "diagnosis", "icd_code", "category"]


def write_diagnosis_xlsx(
    results: list[DiagnosisExtractionResult], output_path: Path
) -> None:
    """Write diagnosis-extraction results to a single-sheet Excel workbook.

    One row per diagnosis (file_path, diagnosis, icd_code, category). A file with no
    diagnoses still gets one row with empty diagnosis/icd_code/category columns.
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
            ws.append([file_path, "", "", ""])
            row_count += 1
            continue
        for diagnosis in result.diagnoses:
            ws.append(
                [file_path, diagnosis.name, diagnosis.icd_code or "", diagnosis.category or ""]
            )
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


_MAIN_DIAGNOSIS_COLUMNS = [
    "file_path", "main_diagnosis", "main_diagnosis_category", "icd_code", "reasoning",
]
# Group label for notes with no main diagnosis (none found, or the call failed).
_UNKNOWN_DIAGNOSIS = "Unknown"


def write_main_diagnosis_xlsx(
    results: list[MainDiagnosisResult], output_path: Path
) -> None:
    """Write main-diagnosis results to a three-sheet Excel workbook.

    Sheets: MainDiagnosis (one row per file: file_path, main_diagnosis, diagnosis_category,
    icd_code, reasoning), Groups (main_diagnosis, count) and Categories
    (diagnosis_category, count) — counts most common first; no diagnosis counts as "Unknown".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MainDiagnosis"
    ws.append(_MAIN_DIAGNOSIS_COLUMNS)
    for result in results:
        ws.append(
            [
                str(result.file_path),
                result.main_diagnosis or "",
                result.diagnosis_category or "",
                result.icd_code or "",
                result.reasoning or "",
            ]
        )

    group_count = _write_groups_sheet(
        wb.create_sheet("Groups"), "main_diagnosis", [r.main_diagnosis for r in results]
    )
    _write_groups_sheet(
        wb.create_sheet("Categories"),
        "main_diagnosis_category",
        [r.diagnosis_category for r in results],
    )

    wb.save(output_path)
    logger.info(
        "Main-diagnosis XLSX written to %s (%d files, %d groups)",
        output_path,
        len(results),
        group_count,
    )


def count_main_diagnoses(labels: Sequence[str | None]) -> list[tuple[str, int]]:
    """Count patients per main diagnosis (label or category), most common first.

    A missing label counts as "Unknown"; ties are ordered by label for stable output.
    """
    counts = Counter(str(label) if label else _UNKNOWN_DIAGNOSIS for label in labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def count_all_categories(
    category_lists: Sequence[list[DiagnosisCategory] | None],
) -> list[tuple[str, int]]:
    """Count patients per category of ANY of their diagnoses, most common first.

    Each entry is one patient's categories (None for a failed run). A patient counts once
    per distinct category, so counts can sum to more than the number of patients; a patient
    with no categories (or a failed run) counts as "Unknown". Ties ordered by label.
    """
    counts: Counter[str] = Counter()
    for categories in category_lists:
        counts.update({str(c) for c in categories or []} or {_UNKNOWN_DIAGNOSIS})
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _write_groups_sheet(ws: Worksheet, header: str, labels: Sequence[str | None]) -> int:
    """Write <header>/count rows, most common first; return the number of groups."""
    return _write_counts_sheet(ws, header, count_main_diagnoses(labels))


def _write_counts_sheet(ws: Worksheet, header: str, groups: list[tuple[str, int]]) -> int:
    """Write <header>/count rows as given; return the number of groups."""
    ws.append([header, "count"])
    for label, count in groups:
        ws.append([label, count])
    return len(groups)


_DR_VALIDITY_COLUMNS = [
    "folder", "patient_name", "dr_file_path", "verdict", "reasons", "patient_group", "age",
    "diagnosis_check", "diagnoses", "diagnosis_categories", "medical_necessity", "medical_necessity_reasoning", "errors",
]


def write_dr_validity_xlsx(
    results: list[DrValidityPipelineResult],
    output_path: Path,
    *,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
    root_name: str | None = None,
) -> None:
    """Write per-patient dr-validity results to a four-sheet Excel workbook.

    Sheets: All, Valid, Needs Review (INVALID, UNKNOWN_AGE and failed rows) and All
    Categories (category of any diagnosis -> patient count).
    Pass local_root / sp_web_url as for write_pair_xlsx.
    Pass root_name to name a patient whose notes sit directly in local_root (see
    ``patient_display_name``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_dr_validity_row(r, local_root, sp_web_url, root_name) for r in results]
    valid_rows = [r for r in rows if r["verdict"] == ValidityVerdict.VALID]
    review_rows = [r for r in rows if r["verdict"] != ValidityVerdict.VALID]

    wb = openpyxl.Workbook()
    wb.active.title = "All"  # type: ignore[union-attr]
    _write_pair_sheet(wb.active, rows, _DR_VALIDITY_COLUMNS)  # type: ignore[arg-type]
    _write_pair_sheet(wb.create_sheet("Valid"), valid_rows, _DR_VALIDITY_COLUMNS)
    _write_pair_sheet(wb.create_sheet("Needs Review"), review_rows, _DR_VALIDITY_COLUMNS)
    _write_counts_sheet(
        wb.create_sheet("All Categories"),
        "diagnosis_category",
        count_all_categories(
            [r.result.diagnosis_categories if r.result else None for r in results]
        ),
    )

    wb.save(output_path)
    logger.info(
        "Dr-validity XLSX written to %s (%d patients, %d valid, %d need review)",
        output_path,
        len(rows),
        len(valid_rows),
        len(review_rows),
    )


_INTAKE_REVIEW_COLUMNS = [
    "folder", "patient_name", "dr_file_path", "nurse_file_path", "age", "patient_group",
    "verdict", "verdict_phase_1", "verdict_phase_2", "explanation", "diagnosis_categories",
    "diagnoses", "medical_necessity", "medical_necessity_reasoning", "dr_nurse_compare", "warnings",
    "errors",
]

_INTAKE_VERDICT_SHEETS = {
    IntakeVerdict.ACCEPT: "Accept",
    IntakeVerdict.REJECT: "Reject",
    IntakeVerdict.NEEDS_REVIEW: "Needs Review",
    IntakeVerdict.NO_NURSE_NOTE: "No Nurse Note",
}


def medical_necessity_flag(care: CareCheck | None) -> bool | None:
    """Medical necessity (the care check) as true/false; None when not evaluated (adults)."""
    return None if care is None else care is CareCheck.NEEDED


def write_intake_review_xlsx(
    results: list[IntakeReviewPipelineResult],
    output_path: Path,
    *,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
    root_name: str | None = None,
) -> None:
    """Write per-patient intake-review results to an Excel workbook.

    Sheets: All, then one per final verdict (failed patients and folders with no dr note
    appear only under All), and All Categories (category of any Phase I diagnosis ->
    patient count). Pass local_root / sp_web_url / root_name as for write_dr_validity_xlsx.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_intake_review_row(r, local_root, sp_web_url, root_name) for r in results]
    wb = openpyxl.Workbook()
    wb.active.title = "All"
    _write_pair_sheet(wb.active, rows, _INTAKE_REVIEW_COLUMNS)
    for verdict, title in _INTAKE_VERDICT_SHEETS.items():
        _write_pair_sheet(
            wb.create_sheet(title),
            [r for r in rows if r["verdict"] == verdict],
            _INTAKE_REVIEW_COLUMNS,
        )
    _write_counts_sheet(
        wb.create_sheet("All Categories"),
        "diagnosis_category",
        count_all_categories(
            [
                r.result.phase_1.diagnosis_categories if r.result else None
                for r in results
                if r.result or not r.success  # folders with no dr note are not counted
            ]
        ),
    )
    wb.save(output_path)
    logger.info("Intake-review XLSX written to %s (%d patients)", output_path, len(rows))


def _intake_review_row(
    result: IntakeReviewPipelineResult,
    local_root: Path | None,
    sp_web_url: str | None,
    root_name: str | None = None,
) -> dict[str, object]:
    # Any file in the patient folder locates it; a folder with no notes gets a stand-in.
    first = (result.dr_paths or result.nurse_paths or [result.patient_folder / "_"])[0]
    row: dict[str, object] = {
        "folder": resolve_folder(first, local_root, sp_web_url),
        "patient_name": patient_display_name(result.patient_folder, local_root, root_name),
        "dr_file_path": _join_names(result.dr_paths, first, local_root) if result.dr_paths else "",
        "nurse_file_path": (
            _join_names(result.nurse_paths, first, local_root) if result.nurse_paths else ""
        ),
        "errors": result.error or "",
    }
    r = result.result
    if not result.success or r is None:
        row.update({col: "" for col in _INTAKE_REVIEW_COLUMNS if col not in row})
        return row
    p1 = r.phase_1
    care = medical_necessity_flag(p1.care_check)
    row.update({
        "age": format_age(p1.age_years),
        "patient_group": p1.patient_group,
        "verdict": r.verdict,
        "verdict_phase_1": r.verdict_phase_1,
        "verdict_phase_2": r.verdict_phase_2 or "",
        "explanation": r.explanation or "",
        "diagnosis_categories": format_categories(p1.diagnosis_categories),
        "diagnoses": format_diagnoses(p1.diagnoses),
        "medical_necessity": "" if care is None else care,
        "medical_necessity_reasoning": (
            p1.care_reasoning if p1.patient_group is not PatientGroup.ADULT else ""
        ),
        "dr_nurse_compare": dr_nurse_compare(r.pairs) or "",
        "warnings": "; ".join(r.warnings),
    })
    return row


_DAYS_PER_YEAR = 365.25
# Absorbs float noise so e.g. 0.5 years floors to 6 months, not 5.
_AGE_EPSILON = 1e-6


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def format_age(age_years: float | None) -> str:
    """Render an age in years the way a clinician would say it; "" when unknown.

    Under 2 weeks -> days, under 2 months -> weeks, under 2 years -> months, otherwise
    whole years. Units are floored, so 17.9 reads "17 years" (still pediatric).
    """
    if age_years is None or age_years < 0:
        return ""
    days = int(age_years * _DAYS_PER_YEAR + _AGE_EPSILON)
    if days < 14:
        return _plural(days, "day")
    if days < 60:
        return _plural(days // 7, "week")
    if age_years < 2:
        return _plural(int(age_years * 12 + _AGE_EPSILON), "month")
    return _plural(int(age_years + _AGE_EPSILON), "year")


def patient_display_name(
    patient_folder: Path, local_root: Path | None = None, root_name: str | None = None
) -> str:
    """Patient name shown in reports: the name of the folder holding the patient's dr notes.

    When the notes sit directly in local_root (the input was a single patient folder),
    that folder may be a temp/download dir, so root_name — the real (e.g. SharePoint)
    folder name — is used instead when given.
    """
    if root_name and local_root is not None and patient_folder == local_root:
        return root_name
    return patient_folder.name or patient_folder.resolve().name


def _dr_validity_row(
    result: DrValidityPipelineResult,
    local_root: Path | None,
    sp_web_url: str | None,
    root_name: str | None = None,
) -> dict[str, object]:
    first = result.dr_paths[0] if result.dr_paths else result.patient_folder
    row: dict[str, object] = {
        "folder": resolve_folder(first, local_root, sp_web_url),
        "patient_name": patient_display_name(result.patient_folder, local_root, root_name),
        "dr_file_path": _join_names(result.dr_paths, first, local_root),
        "errors": result.error or "",
    }
    r = result.result
    if not result.success or r is None:
        row.update({col: "" for col in _DR_VALIDITY_COLUMNS if col not in row})
        return row
    row.update({
        "verdict": r.verdict,
        "reasons": "; ".join(r.reasons),
        "patient_group": r.patient_group,
        "age": format_age(r.age_years),
        "diagnosis_check": r.diagnosis_check,
        "diagnoses": format_diagnoses(r.diagnoses),
        "diagnosis_categories": format_categories(r.diagnosis_categories),
        "medical_necessity": r.care_check or "",
        "medical_necessity_reasoning": r.care_reasoning,
    })
    return row


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
