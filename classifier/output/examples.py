"""Sample API responses built from obviously fake data, to show the output format.

The fake results go through the real report builders, so the sample always has exactly the
shape of a real response. Nothing here touches SharePoint, an LLM or patient data.
"""

from pathlib import Path

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
    NursePatientMeta,
    NurseVisitFields,
    PairClassificationResult,
    PairPipelineResult,
    PatientGroup,
    PatientMetadata,
    RunError,
    ValidityVerdict,
)
from classifier.output.dr_validity_report import dr_validity_report
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
