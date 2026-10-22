import json
from pathlib import Path

import openpyxl

from classifier.models import ErrorCode, ErrorStage, IntakeReviewPipelineResult, RunError
from classifier.output.csv_writer import write_intake_review_xlsx
from classifier.output.examples import intake_review_example
from classifier.output.intake_review_report import (
    intake_review_all_category_groups,
    intake_review_patient_dict,
    intake_review_report,
    intake_review_summary,
)

_ROOT = Path("/root")


def _failed() -> IntakeReviewPipelineResult:
    folder = _ROOT / "FAILED"
    return IntakeReviewPipelineResult(
        patient_folder=folder,
        dr_paths=[folder / "dr_note.pdf"],
        success=False,
        error="age: LLM error: x",
        errors=[
            RunError(stage=ErrorStage.AGE, code=ErrorCode.LLM_FAILED, message="x", folder=folder)
        ],
    )


def test_failed_patient_dict_has_only_identity_and_error() -> None:
    out = intake_review_patient_dict(_failed(), _ROOT)
    assert out["patient"] == "FAILED"
    assert out["dr_files"] == ["FAILED/dr_note.pdf"]
    assert out["nurse_files"] == []
    assert out["error"] == "age: LLM error: x"
    assert "verdict" not in out


def test_summary_counts_verdicts_phases_and_failures() -> None:
    folder_error = RunError(stage=ErrorStage.FOLDER, code=ErrorCode.EMPTY_FOLDER, message="e")
    assert intake_review_summary([_failed()], [folder_error]) == {
        "total": 1, "accept": 0, "reject": 0, "incomplete": 0, "failed": 1,
        "phase_1_failed": 0, "phase_2_failed": 0, "skipped_folders": 1, "errors": 2,
    }


def test_report_lists_folder_errors_first() -> None:
    folder_error = RunError(
        stage=ErrorStage.FOLDER, code=ErrorCode.EMPTY_FOLDER, message="e", folder=_ROOT / "E"
    )
    report = intake_review_report([_failed()], [folder_error], _ROOT)
    assert list(report) == ["summary", "all_category_groups", "patients", "errors"]
    assert [e["stage"] for e in report["errors"]] == ["folder", "age"]


def test_example_covers_every_outcome() -> None:
    example = intake_review_example()

    assert list(example) == [
        "sharepoint_folder", "web_url", "summary", "all_category_groups", "patients", "errors",
    ]
    patients = example["patients"]
    assert [p["verdict"] for p in patients] == [
        "ACCEPT", "REJECT", "REJECT", "REJECT", "INCOMPLETE",
    ]
    assert [p["verdict_phase_1"] for p in patients] == ["PASS", "FAIL", "FAIL", "PASS", "PASS"]
    assert [p["verdict_phase_2"] for p in patients] == [
        "PASS", "NOT_RUN", "NOT_RUN", "FAIL", "NOT_RUN",
    ]
    assert patients[2]["warnings"] == ["age not found — pediatric rules applied"]
    assert patients[0]["phase_2"]["pairs"][0]["clinical_verdict"] == "MATCH"
    assert example["summary"]["phase_1_failed"] == 2
    text = json.dumps(example)
    assert "EXAMPLE" in text and "/tmp" not in text


def test_xlsx_has_a_sheet_per_verdict(tmp_path: Path) -> None:
    from classifier.output import examples

    results = [
        examples._review("A", 40.0, examples.PatientGroup.ADULT, True, None, None),
        _failed(),
    ]
    out = tmp_path / "intake.xlsx"

    write_intake_review_xlsx(results, out, local_root=_ROOT)

    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["All", "Accept", "Reject", "Incomplete", "All Categories"]
    header = [c.value for c in wb["All"][1]]
    assert header[:8] == [
        "folder", "patient_name", "dr_file_path", "nurse_file_path", "verdict",
        "verdict_phase_1", "verdict_phase_2", "explanation",
    ]
    assert wb["All"].max_row == 3
    assert wb["Incomplete"].max_row == 2
    assert wb["Accept"].max_row == 1
    assert list(wb["All Categories"].iter_rows(values_only=True)) == [
        ("diagnosis_category", "count"), ("Other", 1), ("Unknown", 1),
    ]


def test_patient_has_main_and_all_diagnosis_categories(tmp_path: Path) -> None:
    accepted = intake_review_example()["patients"][0]
    assert accepted["phase_1"]["main_diagnosis_category"] == "Other"
    assert accepted["phase_1"]["diagnosis_categories"] == ["Other", "Respiratory"]

    from classifier.output import examples

    result = examples._review(
        "A", 40.0, examples.PatientGroup.ADULT, True, None, None,
        extra=[examples._SECOND_DIAGNOSIS],
    )
    out = tmp_path / "intake.xlsx"
    write_intake_review_xlsx([result], out)
    rows = list(openpyxl.load_workbook(out)["All"].iter_rows(values_only=True))
    row = dict(zip(rows[0], rows[1], strict=True))
    assert row["main_diagnosis_category"] == "Other"
    assert row["diagnosis_categories"] == "Other; Respiratory"


def test_all_category_groups_use_phase_1_diagnoses() -> None:
    example = intake_review_example()
    assert example["all_category_groups"] == [
        {"diagnosis_category": "Other", "count": 4},
        {"diagnosis_category": "Respiratory", "count": 1},
        {"diagnosis_category": "Unknown", "count": 1},
    ]
    assert intake_review_all_category_groups([_failed()]) == [
        {"diagnosis_category": "Unknown", "count": 1},
    ]
