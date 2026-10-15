"""JSON-ready dr-validity report, shared by the CLI (``--dr-validity``) and the API."""

from pathlib import Path

from classifier.models import DrValidityPipelineResult, RunError, ValidityVerdict
from classifier.output.csv_writer import (
    count_main_diagnoses,
    format_age,
    patient_display_name,
    relative_file_names,
)


def dr_validity_patient_dict(
    r: DrValidityPipelineResult, local_root: Path | None = None, root_name: str | None = None
) -> dict[str, object]:
    """One patient's result; failed patients carry only identity fields plus ``error``.

    Every patient carries ``errors`` (empty when none); a successful patient may still
    have non-fatal ones, e.g. one of several notes had no text.
    """
    out: dict[str, object] = {
        "patient": patient_display_name(r.patient_folder, local_root, root_name),
        "dr_files": relative_file_names(r.dr_paths, r.patient_folder, local_root),
        "success": r.success,
        "errors": [run_error_dict(e, local_root, root_name) for e in r.errors],
    }
    if not r.success or r.result is None:
        out["error"] = r.error
        return out
    res = r.result
    out.update({
        "verdict": res.verdict,
        "reasons": res.reasons,
        "patient_group": res.patient_group,
        "age": format_age(res.age_years),
        "age_years": res.age_years,
        "main_diagnosis": res.main_diagnosis,
        "diagnosis_check": res.diagnosis_check,
        "diagnoses": [d.model_dump() for d in res.diagnoses],
        "care_check": res.care_check,
        "care_reasoning": res.care_reasoning,
    })
    return out


def run_error_dict(
    e: RunError, local_root: Path | None = None, root_name: str | None = None
) -> dict[str, object]:
    """JSON-ready error; folder/file shown as patient name and path relative to local_root."""
    return {
        "patient": patient_display_name(e.folder, local_root, root_name) if e.folder else None,
        "file": relative_file_names([e.file], e.file, local_root)[0] if e.file else None,
        "stage": e.stage,
        "code": e.code,
        "message": e.message,
    }


def dr_validity_errors(
    results: list[DrValidityPipelineResult],
    folder_errors: list[RunError],
    local_root: Path | None = None,
    root_name: str | None = None,
) -> list[dict[str, object]]:
    """Every error of the run: folder-level ones first, then each patient's."""
    errors = [*folder_errors, *(e for r in results for e in r.errors)]
    return [run_error_dict(e, local_root, root_name) for e in errors]


def dr_validity_summary(
    results: list[DrValidityPipelineResult], folder_errors: list[RunError] | None = None
) -> dict[str, int]:
    """Patient counts per outcome; failed runs count as ``failed``, not by verdict.

    ``errors`` counts every error (folder-level and per patient); ``skipped_folders``
    counts folders that could not be processed at all.
    """
    folder_errors = folder_errors or []
    verdicts = [r.result.verdict for r in results if r.success and r.result is not None]
    return {
        "total": len(results),
        "valid": verdicts.count(ValidityVerdict.VALID),
        "invalid": verdicts.count(ValidityVerdict.INVALID),
        "unknown_age": verdicts.count(ValidityVerdict.UNKNOWN_AGE),
        "failed": len(results) - len(verdicts),
        "skipped_folders": len(folder_errors),
        "errors": len(folder_errors) + sum(len(r.errors) for r in results),
    }


def dr_validity_groups(results: list[DrValidityPipelineResult]) -> list[dict[str, object]]:
    """Patients per main diagnosis, most common first ("Unknown" when none/failed)."""
    labels = [r.result.main_diagnosis if r.result else None for r in results]
    return [
        {"main_diagnosis": label, "count": count}
        for label, count in count_main_diagnoses(labels)
    ]
