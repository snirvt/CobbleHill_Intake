"""JSON-ready dr-nurse match report, shared by the CLI (pair folders) and the API."""

from pathlib import Path

from classifier.models import ClinicalVerdict, PairPipelineResult, RunError, unique_categories
from classifier.naming import parse_note_name
from classifier.output.csv_writer import (
    count_all_categories,
    count_main_diagnoses,
    patient_display_name,
    relative_file_names,
    resolve_folder,
)
from classifier.output.dr_validity_report import run_error_dict


def pair_dict(
    r: PairPipelineResult,
    *,
    verbose: bool = False,
    local_root: Path | None = None,
    sp_web_url: str | None = None,
    root_name: str | None = None,
) -> dict[str, object]:
    """One pair's result; failed pairs carry only file fields plus ``error``.

    Every pair carries ``errors`` (empty when none); a successful pair may still have
    non-fatal ones, e.g. one of several notes had no text. ``verbose`` adds the identity
    match and the fields parsed from both notes. ``folder`` is a SharePoint link when
    sp_web_url is given, else the local folder path.
    """
    parsed = parse_note_name(r.dr_file_path.name)
    out: dict[str, object] = {
        "patient": patient_display_name(r.dr_file_path.parent, local_root, root_name),
        "category": parsed.category if parsed else None,
        "folder": resolve_folder(r.dr_file_path, local_root, sp_web_url),
        "dr_file": relative_file_names(r.dr_paths, r.dr_file_path, local_root),
        "nurse_file": relative_file_names(r.nurse_paths, r.nurse_file_path, local_root),
        "success": r.success,
        "errors": [run_error_dict(e, local_root, root_name) for e in r.errors],
    }
    if not r.success or r.result is None:
        out["error"] = r.error
        return out
    res = r.result
    out.update({
        "overall": res.overall,
        "main_diagnosis": res.main_diagnosis,
        "diagnosis_category": res.diagnosis_category,
        "main_diagnosis_icd": res.main_diagnosis_icd,
        "clinical_verdict": res.clinical_verdict,
        "clinical_reasoning": res.clinical_reasoning,
        "diagnosis_check": res.diagnosis_check or "",
        "dr_diagnoses": [d.model_dump() for d in res.dr_diagnoses],
        "nurse_diagnoses": [d.model_dump() for d in res.nurse_diagnoses],
        "diagnosis_categories": res.diagnosis_categories,
        "care_check": res.care_check or "",
        "care_reasoning": res.care_reasoning,
    })
    if verbose:
        dr = res.dr_metadata
        out["identity_match"] = res.identity_match.model_dump()
        out["dr_extracted"] = {
            "meta": dr.meta.model_dump(),
            "complaints": dr.complaints,
            "assessment": dr.assessment,
            "hpi": dr.hpi,
            "ros": dr.ros,
            "plan": dr.plan,
            "medications": dr.medications.model_dump() if dr.medications else None,
            "examination": dr.examination.model_dump() if dr.examination else None,
            "medical_history": dr.medical_history,
            "surgical_history": dr.surgical_history,
            "procedure_codes": dr.procedure_codes,
        }
        out["nurse_extracted"] = res.nurse_fields.model_dump()
    return out


def pair_errors(
    results: list[PairPipelineResult],
    folder_errors: list[RunError],
    local_root: Path | None = None,
    root_name: str | None = None,
) -> list[dict[str, object]]:
    """Every error of the run: folder-level ones first, then each pair's."""
    errors = [*folder_errors, *(e for r in results for e in r.errors)]
    return [run_error_dict(e, local_root, root_name) for e in errors]


def pair_summary(
    results: list[PairPipelineResult], folder_errors: list[RunError] | None = None
) -> dict[str, int]:
    """Pair counts per overall verdict; failed runs count as ``failed``, not by verdict.

    ``errors`` counts every error (folder-level and per pair); ``skipped_folders``
    counts folders that could not be processed at all.
    """
    folder_errors = folder_errors or []
    verdicts = [r.result.overall for r in results if r.success and r.result is not None]
    return {
        "total": len(results),
        "match": verdicts.count(ClinicalVerdict.MATCH),
        "mismatch": verdicts.count(ClinicalVerdict.MISMATCH),
        "undecided": verdicts.count(ClinicalVerdict.UNDECIDED),
        "failed": len(results) - len(verdicts),
        "skipped_folders": len(folder_errors),
        "errors": len(folder_errors) + sum(len(r.errors) for r in results),
    }


def pair_groups(results: list[PairPipelineResult]) -> list[dict[str, object]]:
    """Pairs per main diagnosis, most common first ("Unknown" when none/failed)."""
    labels = [r.result.main_diagnosis if r.result else None for r in results]
    return [
        {"main_diagnosis": label, "count": count}
        for label, count in count_main_diagnoses(labels)
    ]


def pair_category_groups(results: list[PairPipelineResult]) -> list[dict[str, object]]:
    """Pairs per diagnosis category, most common first ("Unknown" when none/failed)."""
    categories = [r.result.diagnosis_category if r.result else None for r in results]
    return [
        {"diagnosis_category": category, "count": count}
        for category, count in count_main_diagnoses(categories)
    ]


def pair_all_category_groups(results: list[PairPipelineResult]) -> list[dict[str, object]]:
    """Pairs per category of any dr-note diagnosis (once per category), most common first."""
    categories = [unique_categories(r.result.dr_diagnoses) if r.result else None for r in results]
    return [
        {"diagnosis_category": category, "count": count}
        for category, count in count_all_categories(categories)
    ]


def pair_report(
    results: list[PairPipelineResult],
    folder_errors: list[RunError],
    local_root: Path | None = None,
    root_name: str | None = None,
    web_url: str | None = None,
) -> dict[str, object]:
    """Full report body: {summary, groups, category_groups, all_category_groups, pairs, errors}.

    Pass ``web_url`` (the SharePoint folder link) so each pair's ``folder`` is a link
    rather than a path inside the temporary download dir.
    """
    return {
        "summary": pair_summary(results, folder_errors),
        "groups": pair_groups(results),
        "category_groups": pair_category_groups(results),
        "all_category_groups": pair_all_category_groups(results),
        "pairs": [
            pair_dict(r, local_root=local_root, sp_web_url=web_url, root_name=root_name)
            for r in results
        ],
        "errors": pair_errors(results, folder_errors, local_root, root_name),
    }
