"""Sample API responses built from obviously fake data, to show the output format.

The fake results go through the real report builders, so the sample always has exactly the
shape of a real response. Nothing here touches SharePoint, an LLM or patient data.
"""

from pathlib import Path

from classifier.classifiers.intake_verdict import (
    explain,
    final_verdict,
    phase_1_verdict,
    phase_2_verdict,
)
from classifier.models import (
    CareCheck,
    ClinicalVerdict,
    Diagnosis,
    DiagnosisCategory,
    DiagnosisCheck,
    DocumentMetadata,
    DrValidityPipelineResult,
    DrValidityResult,
    ErrorCode,
    ErrorStage,
    IdentityMatchResult,
    IntakeReviewPipelineResult,
    IntakeReviewResult,
    NursePatientMeta,
    NurseVisitFields,
    PairClassificationResult,
    PairMatch,
    PairPipelineResult,
    PatientGroup,
    PatientMetadata,
    PhaseVerdict,
    RunError,
    ValidityVerdict,
    unique_categories,
)
from classifier.output.dr_validity_report import dr_validity_report
from classifier.output.intake_review_report import intake_review_report
from classifier.output.pair_report import pair_report

_ROOT = Path("/example")
_SHAREPOINT_FOLDER = "EXAMPLE/Physician Notes/Pending"
_WEB_URL = "https://sharepoint.example/EXAMPLE/Physician%20Notes/Pending"
_ROOT_NAME = "Pending"
_REASONING = "EXAMPLE - not real data"
_DIAGNOSIS = Diagnosis(
    name="Example Diagnosis", icd_code="Z00.00", category=DiagnosisCategory.OTHER
)

_OK_FOLDER = _ROOT / "EXAMPLE, PATIENT ONE"
_FAILED_FOLDER = _ROOT / "EXAMPLE, PATIENT TWO"
_SKIPPED_FOLDER = _ROOT / "EXAMPLE, PATIENT THREE"


def _llm_error(folder: Path) -> RunError:
    """A failed LLM step, as reported after retries."""
    return RunError(
        stage=ErrorStage.MAIN_DIAGNOSIS,
        code=ErrorCode.LLM_FAILED,
        message="LLM error: EXAMPLE timeout",
        folder=folder,
    )


def _folder_error(message: str) -> RunError:
    """A folder that could not be processed at all."""
    return RunError(
        stage=ErrorStage.FOLDER,
        code=ErrorCode.MISSING_REQUIRED_NOTES,
        message=message,
        folder=_SKIPPED_FOLDER,
    )


def _wrap(report: dict[str, object]) -> dict[str, object]:
    """Add the request fields FolderReportService.run puts in front of every report."""
    return {"sharepoint_folder": _SHAREPOINT_FOLDER, "web_url": _WEB_URL, **report}


def dr_validity_example() -> dict[str, object]:
    """A /dr-validity response: one valid patient, one failed patient, one skipped folder."""
    ok_paths = [_OK_FOLDER / "peds_dr_example.pdf"]
    failed_paths = [_FAILED_FOLDER / "dr_example.pdf"]
    results = [
        DrValidityPipelineResult(
            patient_folder=_OK_FOLDER,
            dr_paths=ok_paths,
            success=True,
            result=DrValidityResult(
                patient_folder=_OK_FOLDER,
                dr_paths=ok_paths,
                age_years=0.5,
                patient_group=PatientGroup.PEDIATRIC,
                diagnosis_check=DiagnosisCheck.EXISTS,
                diagnoses=[_DIAGNOSIS],
                diagnosis_categories=[DiagnosisCategory.OTHER],
                care_check=CareCheck.NEEDED,
                care_reasoning=_REASONING,
                main_diagnosis="Example Diagnosis",
                diagnosis_category=DiagnosisCategory.OTHER,
                main_diagnosis_icd="Z00.00",
                verdict=ValidityVerdict.VALID,
                reasons=[],
            ),
        ),
        DrValidityPipelineResult(
            patient_folder=_FAILED_FOLDER,
            dr_paths=failed_paths,
            success=False,
            error="main_diagnosis: LLM error: EXAMPLE timeout",
            errors=[_llm_error(_FAILED_FOLDER)],
        ),
    ]
    folder_errors = [_folder_error("Folder has no dr notes")]
    return _wrap(dr_validity_report(results, folder_errors, _ROOT, _ROOT_NAME, _WEB_URL))


def dr_nurse_match_example() -> dict[str, object]:
    """A /dr-nurse-match response: one matching pair, one failed pair, one skipped folder."""
    ok_dr = _OK_FOLDER / "peds_dr_example.pdf"
    ok_nurse = _OK_FOLDER / "peds_nurse_example.pdf"
    failed_dr = _FAILED_FOLDER / "dr_example.pdf"
    failed_nurse = _FAILED_FOLDER / "nurse_example.pdf"
    results = [
        PairPipelineResult(
            dr_file_path=ok_dr,
            nurse_file_path=ok_nurse,
            dr_paths=[ok_dr],
            nurse_paths=[ok_nurse],
            success=True,
            result=PairClassificationResult(
                dr_file_path=ok_dr,
                nurse_file_path=ok_nurse,
                identity_match=IdentityMatchResult(
                    patient_name=True, dob=True, dos=True, sex=True
                ),
                clinical_verdict=ClinicalVerdict.MATCH,
                clinical_reasoning=_REASONING,
                overall=ClinicalVerdict.MATCH,
                dr_metadata=DocumentMetadata(
                    file_path=ok_dr, raw_text=_REASONING, meta=PatientMetadata()
                ),
                nurse_fields=NurseVisitFields(meta=NursePatientMeta()),
                dr_diagnoses=[_DIAGNOSIS],
                nurse_diagnoses=[_DIAGNOSIS],
                diagnosis_categories=[DiagnosisCategory.OTHER],
                diagnosis_check=DiagnosisCheck.EXISTS,
                care_check=CareCheck.NEEDED,
                care_reasoning=_REASONING,
                main_diagnosis="Example Diagnosis",
                diagnosis_category=DiagnosisCategory.OTHER,
                main_diagnosis_icd="Z00.00",
            ),
        ),
        PairPipelineResult(
            dr_file_path=failed_dr,
            nurse_file_path=failed_nurse,
            dr_paths=[failed_dr],
            nurse_paths=[failed_nurse],
            success=False,
            error="main_diagnosis: LLM error: EXAMPLE timeout",
            errors=[_llm_error(_FAILED_FOLDER)],
        ),
    ]
    folder_errors = [_folder_error("Folder has no nurse notes")]
    return _wrap(pair_report(results, folder_errors, _ROOT, _ROOT_NAME, _WEB_URL))


_NURSE_DIAGNOSIS = Diagnosis(
    name="Other Example Diagnosis", icd_code="Z99.99", category=DiagnosisCategory.OTHER
)
_SECOND_DIAGNOSIS = Diagnosis(
    name="Example Respiratory Diagnosis", icd_code="Z11.11", category=DiagnosisCategory.RESPIRATORY
)


def _review(
    name: str,
    age_years: float | None,
    group: PatientGroup,
    diagnosis: bool,
    care: CareCheck | None,
    nurse: tuple[ClinicalVerdict, Diagnosis] | None,
    extra: list[Diagnosis] | None = None,
) -> IntakeReviewPipelineResult:
    """One fake reviewed patient; the verdicts go through the real intake rules.

    ``extra`` adds secondary diagnoses after the main one.
    """
    folder = _ROOT / name
    diagnoses = [_DIAGNOSIS, *(extra or [])] if diagnosis else []
    dr_paths = [folder / "peds_dr_example.pdf"]
    nurse_paths = [folder / "peds_nurse_example.pdf"] if nurse else []
    phase_1 = DrValidityResult(
        patient_folder=folder,
        dr_paths=dr_paths,
        age_years=age_years,
        patient_group=group,
        diagnosis_check=DiagnosisCheck.EXISTS if diagnosis else DiagnosisCheck.MISSING,
        diagnoses=diagnoses,
        diagnosis_categories=unique_categories(diagnoses),
        care_check=care,
        care_reasoning=_REASONING if care else "",
        main_diagnosis="Example Diagnosis" if diagnosis else None,
        diagnosis_category=DiagnosisCategory.OTHER if diagnosis else None,
        main_diagnosis_icd="Z00.00" if diagnosis else None,
        verdict=ValidityVerdict.VALID,
    )
    verdict_1, reasons, warnings = phase_1_verdict(phase_1)
    pairs = (
        [
            PairMatch(
                category="peds",
                dr_paths=dr_paths,
                nurse_paths=nurse_paths,
                clinical_verdict=nurse[0],
                clinical_reasoning=_REASONING,
                nurse_diagnoses=[nurse[1]],
            )
        ]
        if nurse and verdict_1 is PhaseVerdict.PASS
        else []
    )
    verdict_2 = phase_2_verdict(pairs) if verdict_1 is PhaseVerdict.PASS else PhaseVerdict.NOT_RUN
    return IntakeReviewPipelineResult(
        patient_folder=folder,
        dr_paths=dr_paths,
        nurse_paths=nurse_paths,
        success=True,
        result=IntakeReviewResult(
            patient_folder=folder,
            phase_1=phase_1,
            verdict_phase_1=verdict_1,
            phase_1_reasons=reasons,
            pairs=pairs,
            verdict_phase_2=verdict_2,
            verdict=final_verdict(verdict_1, verdict_2),
            explanation=explain(verdict_1, reasons, phase_1.diagnoses, pairs),
            warnings=warnings,
        ),
    )


def intake_review_example() -> dict[str, object]:
    """An /intake-review response: one patient per outcome, plus one skipped folder.

    Accept; Phase I reject (adult, no diagnosis); Phase I reject (age not found, so
    pediatric rules, no medical necessity); Phase II reject (notes do not match);
    incomplete (no nurse note).
    """
    match = (ClinicalVerdict.MATCH, _DIAGNOSIS)
    results = [
        _review(
            "EXAMPLE, ACCEPTED",
            0.5,
            PatientGroup.PEDIATRIC,
            True,
            CareCheck.NEEDED,
            match,
            extra=[_SECOND_DIAGNOSIS],
        ),
        _review("EXAMPLE, NO DIAGNOSIS", 40.0, PatientGroup.ADULT, False, None, match),
        _review(
            "EXAMPLE, NO NECESSITY", None, PatientGroup.UNKNOWN, True, CareCheck.NOT_NEEDED, match
        ),
        _review(
            "EXAMPLE, MISMATCH",
            40.0,
            PatientGroup.ADULT,
            True,
            None,
            (ClinicalVerdict.MISMATCH, _NURSE_DIAGNOSIS),
        ),
        _review("EXAMPLE, NO NURSE NOTE", 40.0, PatientGroup.ADULT, True, None, None),
    ]
    folder_errors = [_folder_error("Folder has no dr notes")]
    return _wrap(intake_review_report(results, folder_errors, _ROOT, _ROOT_NAME, _WEB_URL))
