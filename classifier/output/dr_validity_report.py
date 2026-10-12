"""JSON-ready dr-validity report, shared by the CLI (``--dr-validity``) and the API."""

from pathlib import Path

from classifier.models import DrValidityPipelineResult, ValidityVerdict
from classifier.output.csv_writer import (
    count_main_diagnoses,
    format_age,
    patient_display_name,
    relative_file_names,
)


def dr_validity_patient_dict(
    r: DrValidityPipelineResult, local_root: Path | None = None, root_name: str | None = None
) -> dict[str, object]:
    """One patient's result; failed patients carry only identity fields plus ``error``."""
    out: dict[str, object] = {
        "patient": patient_display_name(r.patient_folder, local_root, root_name),
        "dr_files": relative_file_names(r.dr_paths, r.patient_folder, local_root),
        "success": r.success,
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


def dr_validity_summary(results: list[DrValidityPipelineResult]) -> dict[str, int]:
    """Patient counts per outcome; failed runs count as ``failed``, not by verdict."""
    verdicts = [r.result.verdict for r in results if r.success and r.result is not None]
    return {
        "total": len(results),
        "valid": verdicts.count(ValidityVerdict.VALID),
        "invalid": verdicts.count(ValidityVerdict.INVALID),
        "unknown_age": verdicts.count(ValidityVerdict.UNKNOWN_AGE),
        "failed": len(results) - len(verdicts),
    }


def dr_validity_groups(results: list[DrValidityPipelineResult]) -> list[dict[str, object]]:
    """Patients per main diagnosis, most common first ("Unknown" when none/failed)."""
    labels = [r.result.main_diagnosis if r.result else None for r in results]
    return [
        {"main_diagnosis": label, "count": count}
        for label, count in count_main_diagnoses(labels)
    ]
