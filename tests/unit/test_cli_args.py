from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import cli
from classifier.models import (
    ErrorCode,
    ErrorStage,
    MainDiagnosisResult,
    PairPipelineResult,
    RunError,
)


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


def test_dr_nurse_match_flag_parses() -> None:
    assert cli.build_parser().parse_args(["--dr-nurse-match"]).dr_nurse_match is True
    assert cli.build_parser().parse_args([]).dr_nurse_match is False


def _fake_pair_pipeline(
    results: list[PairPipelineResult], folder_errors: list[RunError]
) -> MagicMock:
    pipeline = MagicMock()
    pipeline.run_folder = AsyncMock(return_value=results)
    pipeline.folder_errors = MagicMock(return_value=folder_errors)
    return pipeline


@pytest.mark.asyncio
async def test_run_dr_nurse_match_writes_xlsx_and_counts_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    src = tmp_path / "in"
    src.mkdir()
    failed = PairPipelineResult(
        dr_file_path=src / "dr_a.pdf",
        nurse_file_path=src / "nurse_a.pdf",
        success=False,
        error="clinical_match: LLM error: down",
    )
    skipped = RunError(
        stage=ErrorStage.FOLDER,
        code=ErrorCode.MISSING_REQUIRED_NOTES,
        message="Folder has no nurse notes",
        folder=src / "X",
    )
    pipeline = _fake_pair_pipeline([failed], [skipped])
    monkeypatch.setattr(cli, "build_pair_pipeline", lambda: pipeline)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exit_code, written = await cli._run_dr_nurse_match(src, out_dir=out_dir)

    assert exit_code == 1
    assert written == out_dir / "pair_results.xlsx"
    assert written.exists()
    pipeline.run_folder.assert_awaited_once_with(src)
    assert "Folder has no nurse notes" in caplog.text


@pytest.mark.asyncio
async def test_run_dr_nurse_match_with_no_pairs_still_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "build_pair_pipeline", lambda: _fake_pair_pipeline([], []))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exit_code, written = await cli._run_dr_nurse_match(tmp_path, out_dir=out_dir)

    assert (exit_code, written) == (0, out_dir / "pair_results.xlsx")


@pytest.mark.asyncio
async def test_run_dr_nurse_match_rejects_non_folder(tmp_path: Path) -> None:
    note = tmp_path / "dr_a.txt"
    note.write_text("x")
    assert await cli._run_dr_nurse_match(note, out_dir=tmp_path) == (1, None)


def test_intake_review_flag_parses() -> None:
    args = cli.build_parser().parse_args(["--input", "./data", "--intake-review"])
    assert args.intake_review is True
    assert cli.build_parser().parse_args([]).intake_review is False
