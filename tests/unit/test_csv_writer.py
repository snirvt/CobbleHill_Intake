import csv
from pathlib import Path

import pytest

from classifier.models import (
    ClassificationResult,
    MedicationData,
    PipelineResult,
)
from classifier.output.csv_writer import write_csv


def _make_success(result: ClassificationResult) -> PipelineResult:
    return PipelineResult(file_path=result.file_path, success=True, result=result)


def _make_failure(file_path: Path, error: str) -> PipelineResult:
    return PipelineResult(file_path=file_path, success=False, error=error)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_csv_creates_file(
    tmp_path: Path,
    sample_document_metadata,  # from conftest
) -> None:
    result = ClassificationResult(
        file_path=sample_document_metadata.file_path,
        category="NEEDS_VISIT",
        task_results={"doctor_visit_needed": "NEEDS_VISIT"},
        metadata=sample_document_metadata,
    )
    out = tmp_path / "out" / "results.csv"
    write_csv([_make_success(result)], out, task_names=["doctor_visit_needed"])
    assert out.exists()


def test_csv_success_row_columns(
    tmp_path: Path,
    sample_document_metadata,
) -> None:
    result = ClassificationResult(
        file_path=sample_document_metadata.file_path,
        category="NEEDS_VISIT",
        task_results={"doctor_visit_needed": "NEEDS_VISIT"},
        metadata=sample_document_metadata,
    )
    out = tmp_path / "results.csv"
    write_csv([_make_success(result)], out, task_names=["doctor_visit_needed"])

    rows = _read_csv(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["success"] == "True"
    assert row["category"] == "NEEDS_VISIT"
    assert row["doctor_visit_needed"] == "NEEDS_VISIT"
    assert row["patient_name"] == "Test, Patient"
    assert row["dob"] == "05/30/2025"
    assert row["errors"] == ""


def test_csv_failure_row(tmp_path: Path, sample_pdf: Path) -> None:
    out = tmp_path / "results.csv"
    write_csv(
        [_make_failure(sample_pdf, "OCR failed")],
        out,
        task_names=["doctor_visit_needed"],
    )

    rows = _read_csv(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["success"] == "False"
    assert row["category"] == ""
    assert row["doctor_visit_needed"] == ""
    assert row["patient_name"] == ""
    assert row["errors"] == "OCR failed"


def test_csv_multiple_rows(
    tmp_path: Path,
    sample_document_metadata,
    sample_pdf: Path,
) -> None:
    success = ClassificationResult(
        file_path=sample_document_metadata.file_path,
        category="NO_VISIT_NEEDED",
        task_results={"doctor_visit_needed": "NO_VISIT_NEEDED"},
        metadata=sample_document_metadata,
    )
    out = tmp_path / "results.csv"
    write_csv(
        [_make_success(success), _make_failure(sample_pdf, "timeout")],
        out,
        task_names=["doctor_visit_needed"],
    )

    rows = _read_csv(out)
    assert len(rows) == 2
    assert rows[0]["category"] == "NO_VISIT_NEEDED"
    assert rows[1]["errors"] == "timeout"


def test_csv_creates_parent_dirs(
    tmp_path: Path,
    sample_document_metadata,
) -> None:
    result = ClassificationResult(
        file_path=sample_document_metadata.file_path,
        category="NEEDS_VISIT",
        task_results={},
        metadata=sample_document_metadata,
    )
    deep = tmp_path / "a" / "b" / "c" / "results.csv"
    write_csv([_make_success(result)], deep, task_names=[])
    assert deep.exists()


def test_csv_unknown_task_defaults_empty(
    tmp_path: Path,
    sample_document_metadata,
) -> None:
    result = ClassificationResult(
        file_path=sample_document_metadata.file_path,
        category="NEEDS_VISIT",
        task_results={},  # task_results empty but task_names non-empty
        metadata=sample_document_metadata,
    )
    out = tmp_path / "results.csv"
    write_csv([_make_success(result)], out, task_names=["doctor_visit_needed"])

    rows = _read_csv(out)
    assert rows[0]["doctor_visit_needed"] == ""
