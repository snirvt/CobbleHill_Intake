from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import cli
from classifier.models import MainDiagnosisResult


def test_upload_results_absent_means_no_upload() -> None:
    args = cli.build_parser().parse_args([])
    assert args.upload_results is None


def test_upload_results_bare_flag_means_default_folder() -> None:
    args = cli.build_parser().parse_args(["--upload-results"])
    assert args.upload_results == ""


def test_upload_results_takes_sharepoint_folder() -> None:
    args = cli.build_parser().parse_args(["--upload-results", "Some/SP/Folder"])
    assert args.upload_results == "Some/SP/Folder"


def test_results_folder_is_local_path() -> None:
    args = cli.build_parser().parse_args(["--results-folder", "./my_output"])
    assert args.results_folder == Path("my_output")


def test_results_folder_absent_is_none() -> None:
    args = cli.build_parser().parse_args([])
    assert args.results_folder is None


def test_results_folder_and_upload_results_parse_independently() -> None:
    args = cli.build_parser().parse_args(
        ["--results-folder", "./out", "--upload-results", "Some/SP/Folder", "--dr-validity"]
    )
    assert args.results_folder == Path("out")
    assert args.upload_results == "Some/SP/Folder"
    assert args.dr_validity is True


@pytest.mark.asyncio
async def test_run_main_diagnosis_writes_xlsx_into_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = tmp_path / "in" / "dr_note.txt"
    note.parent.mkdir()
    note.write_text("synthetic")
    pipeline = MagicMock()
    pipeline.run = AsyncMock(
        return_value=[MainDiagnosisResult(file_path=note, main_diagnosis="Asthma")]
    )
    monkeypatch.setattr(cli, "build_main_diagnosis_pipeline", lambda: pipeline)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exit_code, written = await cli._run_main_diagnosis(note, out_dir=out_dir)

    assert exit_code == 0
    assert written == out_dir / "main_diagnosis_results.xlsx"
    assert written.exists()
