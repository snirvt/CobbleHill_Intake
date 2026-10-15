from pathlib import Path

from classifier.models import (
    CareCheck,
    Diagnosis,
    DiagnosisCheck,
    DrValidityPipelineResult,
    DrValidityResult,
    ErrorCode,
    ErrorStage,
    PatientGroup,
    RunError,
    ValidityVerdict,
)
from classifier.output.dr_validity_report import (
    dr_validity_errors,
    dr_validity_groups,
    dr_validity_patient_dict,
    dr_validity_summary,
    run_error_dict,
)


def _ok(
    root: Path, name: str, verdict: ValidityVerdict, main: str | None = "GERD"
) -> DrValidityPipelineResult:
    folder = root / name
    paths = [folder / "dr_a.pdf", folder / "dr_b.pdf"]
    return DrValidityPipelineResult(
        patient_folder=folder,
        dr_paths=paths,
        success=True,
        result=DrValidityResult(
            patient_folder=folder,
            dr_paths=paths,
            age_years=0.5,
            patient_group=PatientGroup.PEDIATRIC,
            diagnosis_check=DiagnosisCheck.EXISTS,
            diagnoses=[Diagnosis(name="GERD", icd_code="K21.9")],
            care_check=CareCheck.NEEDED,
            care_reasoning="needs home services",
            main_diagnosis=main,
            verdict=verdict,
            reasons=[],
        ),
    )


def _llm_error(folder: Path) -> RunError:
    return RunError(
        stage=ErrorStage.CARE, code=ErrorCode.LLM_FAILED, message="LLM error: down", folder=folder
    )


def _failed(root: Path, name: str) -> DrValidityPipelineResult:
    folder = root / name
    return DrValidityPipelineResult(
        patient_folder=folder,
        dr_paths=[folder / "dr_a.pdf"],
        success=False,
        error="OCR failed",
        errors=[_llm_error(folder)],
    )


def test_patient_dict_success_has_all_fields(tmp_path: Path) -> None:
    out = dr_validity_patient_dict(_ok(tmp_path, "DOE, JANE", ValidityVerdict.VALID), tmp_path)

    assert out == {
        "patient": "DOE, JANE",
        "dr_files": ["DOE, JANE/dr_a.pdf", "DOE, JANE/dr_b.pdf"],
        "success": True,
        "errors": [],
        "verdict": "VALID",
        "reasons": [],
        "patient_group": "PEDIATRIC",
        "age": "6 months",
        "age_years": 0.5,
        "main_diagnosis": "GERD",
        "diagnosis_check": "EXISTS",
        "diagnoses": [{"name": "GERD", "icd_code": "K21.9", "source": None}],
        "care_check": "NEEDED",
        "care_reasoning": "needs home services",
    }


def test_patient_dict_failure_has_error_only(tmp_path: Path) -> None:
    out = dr_validity_patient_dict(_failed(tmp_path, "X"), tmp_path)
    assert out == {
        "patient": "X",
        "dr_files": ["X/dr_a.pdf"],
        "success": False,
        "errors": [
            {
                "patient": "X",
                "file": None,
                "stage": "care",
                "code": "llm_failed",
                "message": "LLM error: down",
            }
        ],
        "error": "OCR failed",
    }


def test_patient_dict_uses_root_name_for_notes_in_root(tmp_path: Path) -> None:
    root = tmp_path / "cobblehill_api_123"
    result = _ok(tmp_path, root.name, ValidityVerdict.VALID)
    assert dr_validity_patient_dict(result, root, "EISENBERG, SHEINDY")["patient"] == (
        "EISENBERG, SHEINDY"
    )


def test_summary_counts_each_outcome(tmp_path: Path) -> None:
    results = [
        _ok(tmp_path, "A", ValidityVerdict.VALID),
        _ok(tmp_path, "B", ValidityVerdict.VALID),
        _ok(tmp_path, "C", ValidityVerdict.INVALID),
        _ok(tmp_path, "D", ValidityVerdict.UNKNOWN_AGE),
        _failed(tmp_path, "E"),
    ]
    folder_errors = [
        RunError(stage=ErrorStage.FOLDER, code=ErrorCode.EMPTY_FOLDER, message="empty")
    ]
    assert dr_validity_summary(results, folder_errors) == {
        "total": 5, "valid": 2, "invalid": 1, "unknown_age": 1, "failed": 1,
        "skipped_folders": 1, "errors": 2,
    }


def test_summary_empty() -> None:
    assert dr_validity_summary([]) == {
        "total": 0, "valid": 0, "invalid": 0, "unknown_age": 0, "failed": 0,
        "skipped_folders": 0, "errors": 0,
    }


def test_run_error_dict_shows_patient_name_and_relative_file(tmp_path: Path) -> None:
    folder = tmp_path / "Pending" / "DOE, JANE"
    error = RunError(
        stage=ErrorStage.EXTRACT,
        code=ErrorCode.EMPTY_TEXT,
        message="No text extracted from note",
        folder=folder,
        file=folder / "dr_a.pdf",
    )

    assert run_error_dict(error, tmp_path) == {
        "patient": "DOE, JANE",
        "file": "Pending/DOE, JANE/dr_a.pdf",
        "stage": "extract",
        "code": "empty_text",
        "message": "No text extracted from note",
    }


def test_run_error_dict_uses_root_name_for_root_folder(tmp_path: Path) -> None:
    error = _llm_error(tmp_path)
    assert run_error_dict(error, tmp_path, "EISENBERG, SHEINDY")["patient"] == "EISENBERG, SHEINDY"


def test_errors_lists_folder_errors_then_patient_errors(tmp_path: Path) -> None:
    empty = RunError(
        stage=ErrorStage.FOLDER,
        code=ErrorCode.EMPTY_FOLDER,
        message="Folder has no files",
        folder=tmp_path / "EMPTY",
    )
    results = [_ok(tmp_path, "A", ValidityVerdict.VALID), _failed(tmp_path, "B")]

    errors = dr_validity_errors(results, [empty], tmp_path)

    assert [(e["patient"], e["code"]) for e in errors] == [
        ("EMPTY", "empty_folder"),
        ("B", "llm_failed"),
    ]


def test_groups_most_common_first_with_unknown(tmp_path: Path) -> None:
    results = [
        _ok(tmp_path, "A", ValidityVerdict.VALID, main="Dementia"),
        _ok(tmp_path, "B", ValidityVerdict.VALID, main="GERD"),
        _ok(tmp_path, "C", ValidityVerdict.VALID, main="Dementia"),
        _failed(tmp_path, "D"),
    ]
    assert dr_validity_groups(results) == [
        {"main_diagnosis": "Dementia", "count": 2},
        {"main_diagnosis": "GERD", "count": 1},
        {"main_diagnosis": "Unknown", "count": 1},
    ]
