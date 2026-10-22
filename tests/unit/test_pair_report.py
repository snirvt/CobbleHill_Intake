from pathlib import Path

from classifier.models import (
    CareCheck,
    ClinicalVerdict,
    Diagnosis,
    DiagnosisCategory,
    DocumentMetadata,
    ErrorCode,
    ErrorStage,
    IdentityMatchResult,
    NursePatientMeta,
    NurseVisitFields,
    PairClassificationResult,
    PairPipelineResult,
    PatientMetadata,
    RunError,
)
from classifier.output.pair_report import (
    pair_category_groups,
    pair_dict,
    pair_errors,
    pair_groups,
    pair_report,
    pair_summary,
)


def _ok(
    root: Path,
    name: str,
    overall: ClinicalVerdict = ClinicalVerdict.MATCH,
    main: str | None = "GERD",
    category: DiagnosisCategory | None = DiagnosisCategory.REFLUX,
    errors: list[RunError] | None = None,
) -> PairPipelineResult:
    folder = root / name
    dr, nurse = folder / "peds_dr_a.pdf", folder / "peds_nurse_b.pdf"
    return PairPipelineResult(
        dr_file_path=dr,
        nurse_file_path=nurse,
        dr_paths=[dr],
        nurse_paths=[nurse],
        success=True,
        errors=errors or [],
        result=PairClassificationResult(
            dr_file_path=dr,
            nurse_file_path=nurse,
            identity_match=IdentityMatchResult(patient_name=True, dob=True, dos=True, sex=True),
            clinical_verdict=overall,
            clinical_reasoning="consistent",
            overall=overall,
            dr_metadata=DocumentMetadata(file_path=dr, raw_text="t", meta=PatientMetadata()),
            nurse_fields=NurseVisitFields(meta=NursePatientMeta()),
            dr_diagnoses=[
                Diagnosis(name="GERD", icd_code="K21.9", category=DiagnosisCategory.REFLUX)
            ],
            diagnosis_categories=[DiagnosisCategory.REFLUX],
            care_check=CareCheck.NEEDED,
            care_reasoning="needs care",
            main_diagnosis=main,
            diagnosis_category=category,
            main_diagnosis_icd="K21.9",
        ),
    )


def _failed(root: Path, name: str) -> PairPipelineResult:
    folder = root / name
    dr, nurse = folder / "dr_a.pdf", folder / "nurse_b.pdf"
    error = RunError(
        stage=ErrorStage.CLINICAL_MATCH,
        code=ErrorCode.LLM_FAILED,
        message="LLM error: down",
        folder=folder,
    )
    return PairPipelineResult(
        dr_file_path=dr,
        nurse_file_path=nurse,
        dr_paths=[dr],
        nurse_paths=[nurse],
        success=False,
        error="clinical_match: LLM error: down",
        errors=[error],
    )


def _folder_error(root: Path, name: str) -> RunError:
    return RunError(
        stage=ErrorStage.FOLDER,
        code=ErrorCode.MISSING_REQUIRED_NOTES,
        message="Folder has no nurse notes",
        folder=root / name,
    )


def test_pair_dict_success_has_verdicts_and_relative_files(tmp_path: Path) -> None:
    out = pair_dict(_ok(tmp_path, "DOE, JANE"), local_root=tmp_path)

    assert out["patient"] == "DOE, JANE"
    assert out["category"] == "peds"
    assert out["dr_file"] == ["DOE, JANE/peds_dr_a.pdf"]
    assert out["nurse_file"] == ["DOE, JANE/peds_nurse_b.pdf"]
    assert (out["success"], out["overall"], out["main_diagnosis"]) == (True, "MATCH", "GERD")
    assert out["diagnosis_category"] == "Reflux / GERD"
    assert out["dr_diagnoses"] == [
        {"name": "GERD", "icd_code": "K21.9", "category": "Reflux / GERD"}
    ]
    assert out["diagnosis_categories"] == ["Reflux / GERD"]
    assert out["errors"] == []
    assert "identity_match" not in out


def test_pair_dict_folder_is_sharepoint_link_when_web_url_given(tmp_path: Path) -> None:
    out = pair_dict(_ok(tmp_path, "DOE, JANE"), local_root=tmp_path, sp_web_url="https://sp/P")
    assert out["folder"] == "https://sp/P/DOE%2C%20JANE"


def test_pair_dict_single_patient_root_uses_root_name(tmp_path: Path) -> None:
    result = _ok(tmp_path, "")
    out = pair_dict(result, local_root=tmp_path, root_name="DOE, JANE")
    assert out["patient"] == "DOE, JANE"


def test_pair_dict_failed_carries_error_only(tmp_path: Path) -> None:
    out = pair_dict(_failed(tmp_path, "X"), local_root=tmp_path)

    assert out["success"] is False
    assert out["error"] == "clinical_match: LLM error: down"
    assert "overall" not in out
    errors = out["errors"]
    assert isinstance(errors, list)
    assert errors[0]["stage"] == "clinical_match"


def test_pair_dict_verbose_adds_extracted_fields(tmp_path: Path) -> None:
    out = pair_dict(_ok(tmp_path, "A"), verbose=True, local_root=tmp_path)
    assert {"identity_match", "dr_extracted", "nurse_extracted"} <= set(out)


def test_pair_summary_counts_outcomes_and_errors(tmp_path: Path) -> None:
    blank = RunError(stage=ErrorStage.EXTRACT, code=ErrorCode.EMPTY_TEXT, message="blank")
    results = [
        _ok(tmp_path, "A"),
        _ok(tmp_path, "B", overall=ClinicalVerdict.MISMATCH, errors=[blank]),
        _ok(tmp_path, "C", overall=ClinicalVerdict.UNDECIDED),
        _failed(tmp_path, "D"),
    ]

    summary = pair_summary(results, [_folder_error(tmp_path, "E")])

    assert summary == {
        "total": 4, "match": 1, "mismatch": 1, "undecided": 1, "failed": 1,
        "skipped_folders": 1, "errors": 3,
    }


def test_pair_groups_count_labels_and_unknown(tmp_path: Path) -> None:
    results = [_ok(tmp_path, "A"), _ok(tmp_path, "B"), _ok(tmp_path, "C", main=None, category=None)]

    assert pair_groups(results) == [
        {"main_diagnosis": "GERD", "count": 2},
        {"main_diagnosis": "Unknown", "count": 1},
    ]
    assert pair_category_groups([*results, _failed(tmp_path, "D")]) == [
        {"diagnosis_category": "Reflux / GERD", "count": 2},
        {"diagnosis_category": "Unknown", "count": 2},
    ]


def test_pair_errors_lists_folder_errors_first(tmp_path: Path) -> None:
    errors = pair_errors([_failed(tmp_path, "D")], [_folder_error(tmp_path, "E")], tmp_path)
    assert [(e["patient"], e["stage"]) for e in errors] == [
        ("E", "folder"),
        ("D", "clinical_match"),
    ]


def test_pair_report_combines_all_sections(tmp_path: Path) -> None:
    results = [_ok(tmp_path, "A")]

    report = pair_report(results, [], tmp_path, "root", "https://sp/P")

    assert list(report) == ["summary", "groups", "category_groups", "pairs", "errors"]
    pairs = report["pairs"]
    assert isinstance(pairs, list)
    assert pairs[0]["folder"] == "https://sp/P/A"
