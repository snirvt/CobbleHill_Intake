"""JSON-ready intake-review report, shared by the CLI (``--intake-review``) and the API."""

from pathlib import Path

from classifier.models import (
    IntakeReviewPipelineResult,
    IntakeVerdict,
    PairMatch,
    PhaseVerdict,
    RunError,
)
from classifier.output.csv_writer import (
    count_all_categories,
    format_age,
    patient_display_name,
    relative_file_names,
)
from classifier.output.dr_validity_report import run_error_dict


def _pair_match_dict(p: PairMatch, local_root: Path | None) -> dict[str, object]:
    """One Phase II pair: files, clinical verdict and the nurse note's diagnoses."""
    return {
        "category": p.category,
        "dr_files": relative_file_names(p.dr_paths, p.dr_paths[0], local_root),
        "nurse_files": relative_file_names(p.nurse_paths, p.nurse_paths[0], local_root),
        "clinical_verdict": p.clinical_verdict,
        "clinical_reasoning": p.clinical_reasoning,
        "nurse_diagnoses": [d.model_dump() for d in p.nurse_diagnoses],
    }


def intake_review_patient_dict(
    r: IntakeReviewPipelineResult, local_root: Path | None = None, root_name: str | None = None
) -> dict[str, object]:
    """One patient's review; failed patients carry only identity fields plus ``error``."""
    out: dict[str, object] = {
        "patient": patient_display_name(r.patient_folder, local_root, root_name),
        "dr_files": relative_file_names(r.dr_paths, r.patient_folder, local_root),
        "nurse_files": (
            relative_file_names(r.nurse_paths, r.patient_folder, local_root)
            if r.nurse_paths
            else []
        ),
        "success": r.success,
        "errors": [run_error_dict(e, local_root, root_name) for e in r.errors],
    }
    if not r.success or r.result is None:
        out["error"] = r.error
        return out
    res = r.result
    p1 = res.phase_1
    out.update({
        "verdict_phase_1": res.verdict_phase_1,
        "verdict_phase_2": res.verdict_phase_2,
        "verdict": res.verdict,
        "explanation": res.explanation,
        "warnings": res.warnings,
        "phase_1": {
            "reasons": res.phase_1_reasons,
            "patient_group": p1.patient_group,
            "age": format_age(p1.age_years),
            "age_years": p1.age_years,
            "diagnosis_check": p1.diagnosis_check,
            "diagnoses": [d.model_dump() for d in p1.diagnoses],
            "care_check": p1.care_check,
            "care_reasoning": p1.care_reasoning,
            "main_diagnosis": p1.main_diagnosis,
            "main_diagnosis_category": p1.diagnosis_category,
            "diagnosis_categories": p1.diagnosis_categories,
            "main_diagnosis_icd": p1.main_diagnosis_icd,
        },
        "phase_2": {"pairs": [_pair_match_dict(p, local_root) for p in res.pairs]},
    })
    return out


def intake_review_summary(
    results: list[IntakeReviewPipelineResult], folder_errors: list[RunError] | None = None
) -> dict[str, int]:
    """Patient counts per final verdict and per failed phase; failed runs count as ``failed``."""
    folder_errors = folder_errors or []
    reviewed = [r.result for r in results if r.success and r.result is not None]
    verdicts = [r.verdict for r in reviewed]
    return {
        "total": len(results),
        "accept": verdicts.count(IntakeVerdict.ACCEPT),
        "reject": verdicts.count(IntakeVerdict.REJECT),
        "incomplete": verdicts.count(IntakeVerdict.INCOMPLETE),
        "failed": len(results) - len(reviewed),
        "phase_1_failed": sum(r.verdict_phase_1 is PhaseVerdict.FAIL for r in reviewed),
        "phase_2_failed": sum(r.verdict_phase_2 is PhaseVerdict.FAIL for r in reviewed),
        "skipped_folders": len(folder_errors),
        "errors": len(folder_errors) + sum(len(r.errors) for r in results),
    }


def intake_review_all_category_groups(
    results: list[IntakeReviewPipelineResult],
) -> list[dict[str, object]]:
    """Patients per category of any Phase I diagnosis (once per category), most common first."""
    categories = [r.result.phase_1.diagnosis_categories if r.result else None for r in results]
    return [
        {"diagnosis_category": category, "count": count}
        for category, count in count_all_categories(categories)
    ]


def intake_review_report(
    results: list[IntakeReviewPipelineResult],
    folder_errors: list[RunError],
    local_root: Path | None = None,
    root_name: str | None = None,
    web_url: str | None = None,
) -> dict[str, object]:
    """Full report body: {summary, all_category_groups, patients, errors}.

    ``web_url`` is accepted for the shared report-builder signature and not used.
    """
    errors = [*folder_errors, *(e for r in results for e in r.errors)]
    return {
        "summary": intake_review_summary(results, folder_errors),
        "all_category_groups": intake_review_all_category_groups(results),
        "patients": [intake_review_patient_dict(r, local_root, root_name) for r in results],
        "errors": [run_error_dict(e, local_root, root_name) for e in errors],
    }
