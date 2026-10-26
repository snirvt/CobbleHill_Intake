"""JSON-ready intake-review report, shared by the CLI (``--intake-review``) and the API."""

from pathlib import Path

from classifier.classifiers.intake_verdict import dr_nurse_compare
from classifier.models import (
    IntakeReviewPipelineResult,
    IntakeVerdict,
    PatientGroup,
    PhaseVerdict,
    RunError,
)
from classifier.output.csv_writer import (
    count_all_categories,
    format_age,
    medical_necessity_flag,
    patient_display_name,
    relative_file_names,
)
from classifier.output.dr_validity_report import run_error_dict

# Review fields of a patient that was not reviewed (no dr note, or the run failed).
_EMPTY_REVIEW: dict[str, object] = {
    "age": None,
    "age_years": None,
    "patient_group": None,
    "verdict": None,
    "verdict_phase_1": None,
    "verdict_phase_2": None,
    "explanation": None,
    "phase_1": None,
    "phase_2": None,
    "warnings": [],
}


def intake_review_patient_dict(
    r: IntakeReviewPipelineResult, local_root: Path | None = None, root_name: str | None = None
) -> dict[str, object]:
    """One patient's review: identity, age, verdicts, explanation, Phase I and II details.

    Folders with no dr note and failed patients carry identity fields with every review
    field None; failed patients add ``error``.
    """
    out: dict[str, object] = {
        "patient": patient_display_name(r.patient_folder, local_root, root_name),
        "dr_files": (
            relative_file_names(r.dr_paths, r.patient_folder, local_root) if r.dr_paths else []
        ),
        "nurse_files": (
            relative_file_names(r.nurse_paths, r.patient_folder, local_root)
            if r.nurse_paths
            else []
        ),
        "success": r.success,
        "errors": [run_error_dict(e, local_root, root_name) for e in r.errors],
        **_EMPTY_REVIEW,
    }
    if not r.success:
        out["error"] = r.error
    if r.result is None:
        return out
    res = r.result
    p1 = res.phase_1
    pediatric = p1.patient_group is not PatientGroup.ADULT
    out.update({
        "age": format_age(p1.age_years) or None,
        "age_years": p1.age_years,
        "patient_group": p1.patient_group,
        "verdict": res.verdict,
        "verdict_phase_1": res.verdict_phase_1,
        "verdict_phase_2": res.verdict_phase_2,
        "explanation": res.explanation,
        "phase_1": {
            "diagnosis_categories": p1.diagnosis_categories,
            "diagnoses": [d.model_dump() for d in p1.diagnoses],
            "medical_necessity": medical_necessity_flag(p1.care_check),
            "medical_necessity_reasoning": (p1.care_reasoning or None) if pediatric else None,
        },
        "phase_2": {"dr_nurse_compare": dr_nurse_compare(res.pairs)} if res.pairs else None,
        "warnings": res.warnings,
    })
    return out


def intake_review_summary(
    results: list[IntakeReviewPipelineResult], folder_errors: list[RunError] | None = None
) -> dict[str, int]:
    """Patient counts per final verdict and per failed phase; failed runs count as ``failed``,
    folders with no dr note as ``no_dr_note``."""
    folder_errors = folder_errors or []
    reviewed = [r.result for r in results if r.success and r.result is not None]
    verdicts = [r.verdict for r in reviewed]
    return {
        "total": len(results),
        "accept": verdicts.count(IntakeVerdict.ACCEPT),
        "reject": verdicts.count(IntakeVerdict.REJECT),
        "needs_review": verdicts.count(IntakeVerdict.NEEDS_REVIEW),
        "no_nurse_note": verdicts.count(IntakeVerdict.NO_NURSE_NOTE),
        "no_dr_note": sum(r.success and r.result is None for r in results),
        "failed": sum(not r.success for r in results),
        "phase_1_failed": sum(r.verdict_phase_1 is PhaseVerdict.FAIL for r in reviewed),
        "phase_2_failed": sum(r.verdict_phase_2 is PhaseVerdict.FAIL for r in reviewed),
        "skipped_folders": len(folder_errors),
        "errors": len(folder_errors) + sum(len(r.errors) for r in results),
    }


def intake_review_all_category_groups(
    results: list[IntakeReviewPipelineResult],
) -> list[dict[str, object]]:
    """Patients per category of any Phase I diagnosis (once per category), most common first.

    Failed patients count as Unknown; folders with no dr note are not counted.
    """
    categories = [
        r.result.phase_1.diagnosis_categories if r.result else None
        for r in results
        if r.result or not r.success
    ]
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
