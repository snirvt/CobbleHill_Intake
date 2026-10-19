from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from classifier.dr_validity_service import DrValidityService
from classifier.folder_checks import RequiredNotesCheck, leaf_folders
from classifier.models import (
    DiagnosisCategory,
    DiagnosisCheck,
    DrValidityPipelineResult,
    DrValidityResult,
    ErrorCode,
    ErrorStage,
    PatientGroup,
    RunError,
    ValidityVerdict,
)
from classifier.patient_pipeline import scan_patient_dr_notes


def _downloader(
    files: list[str], web_url: str = "https://sp/folder", dirs: list[str] | None = None
) -> MagicMock:
    """Fake downloader that writes the given relative files (and empty dirs) into local_dir."""

    async def download(folder_path: str, local_dir: Path | None = None) -> tuple[Path, str]:
        assert local_dir is not None
        for rel in files:
            path = local_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("note")
        for rel in dirs or []:
            (local_dir / rel).mkdir(parents=True, exist_ok=True)
        return local_dir, web_url

    downloader = MagicMock()
    downloader.download = AsyncMock(side_effect=download)
    return downloader


def _folder_errors(root: Path) -> list[RunError]:
    check = RequiredNotesCheck(roles=("dr",))
    return [e for f in leaf_folders(root) if (e := check.check(f)) is not None]


def _pipeline(patient_errors: list[RunError] | None = None) -> MagicMock:
    """Fake pipeline: every scanned patient is VALID with main diagnosis GERD.

    patient_errors are attached (as non-fatal) to every patient.
    """

    async def run_folder(root: Path) -> list[DrValidityPipelineResult]:
        out = []
        for paths in scan_patient_dr_notes(root):
            folder = paths[0].parent
            out.append(
                DrValidityPipelineResult(
                    patient_folder=folder,
                    dr_paths=paths,
                    success=True,
                    errors=patient_errors or [],
                    result=DrValidityResult(
                        patient_folder=folder,
                        dr_paths=paths,
                        age_years=80,
                        patient_group=PatientGroup.ADULT,
                        diagnosis_check=DiagnosisCheck.EXISTS,
                        main_diagnosis="GERD",
                        diagnosis_category=DiagnosisCategory.REFLUX,
                        verdict=ValidityVerdict.VALID,
                    ),
                )
            )
        return out

    pipeline = MagicMock()
    pipeline.run_folder = AsyncMock(side_effect=run_folder)
    pipeline.folder_errors = MagicMock(side_effect=_folder_errors)
    return pipeline


async def test_run_returns_report_for_folder_of_patients() -> None:
    downloader = _downloader(["A, B/dr_a.pdf", "A, B/nurse_x.pdf", "C, D/dr_c.pdf"])
    service = DrValidityService(downloader=downloader, pipeline=_pipeline())

    report = await service.run("/Patient Encounters/Physician Notes/Pending/")

    downloader.download.assert_awaited_once()
    assert downloader.download.await_args.args[0] == "Patient Encounters/Physician Notes/Pending"
    assert report["sharepoint_folder"] == "Patient Encounters/Physician Notes/Pending"
    assert report["web_url"] == "https://sp/folder"
    assert report["summary"] == {
        "total": 2, "valid": 2, "invalid": 0, "unknown_age": 0, "failed": 0,
        "skipped_folders": 0, "errors": 0,
    }
    assert report["errors"] == []
    assert report["groups"] == [{"main_diagnosis": "GERD", "count": 2}]
    assert report["category_groups"] == [{"diagnosis_category": "Reflux / GERD", "count": 2}]
    patients = report["patients"]
    assert isinstance(patients, list)
    assert [p["patient"] for p in patients] == ["A, B", "C, D"]
    assert patients[0]["dr_files"] == ["A, B/dr_a.pdf"]


async def test_single_patient_folder_named_from_sharepoint_path() -> None:
    service = DrValidityService(
        downloader=_downloader(["dr_EISENBERG, SHEINDY note.pdf"]), pipeline=_pipeline()
    )

    report = await service.run("Patient Encounters/Physician Notes/Pending/EISENBERG, SHEINDY")

    patients = report["patients"]
    assert isinstance(patients, list)
    assert patients[0]["patient"] == "EISENBERG, SHEINDY"


async def test_empty_folder_returns_no_patients() -> None:
    service = DrValidityService(downloader=_downloader([]), pipeline=_pipeline())

    report = await service.run("Pending")

    assert report["patients"] == []
    assert report["summary"] == {
        "total": 0, "valid": 0, "invalid": 0, "unknown_age": 0, "failed": 0,
        "skipped_folders": 1, "errors": 1,
    }
    assert report["errors"] == [
        {
            "patient": "Pending",
            "file": None,
            "stage": "folder",
            "code": "empty_folder",
            "message": "Folder has no files",
        }
    ]


async def test_download_dir_deleted_after_success() -> None:
    downloader = _downloader(["P/dr_a.pdf"])
    service = DrValidityService(downloader=downloader, pipeline=_pipeline())

    await service.run("Pending")

    local_dir = downloader.download.await_args.args[1]
    assert not local_dir.exists()


async def test_download_dir_deleted_when_pipeline_fails() -> None:
    downloader = _downloader(["P/dr_a.pdf"])
    pipeline = MagicMock()
    pipeline.run_folder = AsyncMock(side_effect=RuntimeError("boom"))
    service = DrValidityService(downloader=downloader, pipeline=pipeline)

    with pytest.raises(RuntimeError):
        await service.run("Pending")

    assert not downloader.download.await_args.args[1].exists()


async def test_download_dir_deleted_when_download_fails() -> None:
    seen: list[Path] = []

    async def download(folder_path: str, local_dir: Path | None = None) -> tuple[Path, str]:
        assert local_dir is not None
        seen.append(local_dir)
        (local_dir / "partial.pdf").write_text("half")
        request = httpx.Request("GET", "https://graph")
        raise httpx.HTTPStatusError("404", request=request, response=httpx.Response(404))

    downloader = MagicMock()
    downloader.download = AsyncMock(side_effect=download)
    service = DrValidityService(downloader=downloader, pipeline=_pipeline())

    with pytest.raises(httpx.HTTPStatusError):
        await service.run("Missing")

    assert seen and not seen[0].exists()


async def test_run_logs_request_timing(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    service = DrValidityService(downloader=_downloader(["A, B/dr_a.pdf"]), pipeline=_pipeline())

    with caplog.at_level(logging.INFO, logger="classifier.dr_validity_service"):
        await service.run("Pending")

    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("timing")]
    assert line.startswith("timing request folder=Pending download ")
    assert line.endswith("| patients=1")


async def test_run_reports_folder_and_patient_errors() -> None:
    downloader = _downloader(
        ["A, B/dr_a.pdf", "NURSE, ONLY/nurse_x.pdf"], dirs=["EMPTY, ONE"]
    )
    blank = RunError(
        stage=ErrorStage.EXTRACT,
        code=ErrorCode.EMPTY_TEXT,
        message="No text extracted from note",
    )
    service = DrValidityService(downloader=downloader, pipeline=_pipeline([blank]))

    report = await service.run("Pending")

    assert report["errors"] == [
        {
            "patient": "EMPTY, ONE",
            "file": None,
            "stage": "folder",
            "code": "empty_folder",
            "message": "Folder has no files",
        },
        {
            "patient": "NURSE, ONLY",
            "file": None,
            "stage": "folder",
            "code": "missing_required_notes",
            "message": "Folder has no dr notes",
        },
        {
            "patient": None,
            "file": None,
            "stage": "extract",
            "code": "empty_text",
            "message": "No text extracted from note",
        },
    ]
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert (summary["skipped_folders"], summary["errors"]) == (2, 3)
    patients = report["patients"]
    assert isinstance(patients, list)
    assert patients[0]["errors"] == [report["errors"][2]]
