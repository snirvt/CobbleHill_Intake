import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from classifier.folder_checks import RequiredNotesCheck
from classifier.models import (
    ClassificationStepError,
    DiagnosisCheck,
    DocumentMetadata,
    DrValidityResult,
    ErrorCode,
    ErrorStage,
    ExtractedText,
    PatientGroup,
    RunError,
    ValidityVerdict,
)
from classifier.patient_pipeline import DrValidityPipeline, scan_patient_dr_notes


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# scan_patient_dr_notes
# ---------------------------------------------------------------------------

def test_scan_groups_multiple_dr_notes_per_patient_and_ignores_nurse(tmp_path: Path) -> None:
    a1 = _touch(tmp_path / "EISENBERG, SHEINDY" / "dr_EISENBERG, SHEINDY note.pdf")
    a2 = _touch(tmp_path / "EISENBERG, SHEINDY" / "dr_necessity.pdf")
    _touch(tmp_path / "EISENBERG, SHEINDY" / "nurse_visit.pdf")
    b1 = _touch(tmp_path / "SMITH, JOHN" / "dr_note.txt")

    assert scan_patient_dr_notes(tmp_path) == [[a1, a2], [b1]]


def test_scan_skips_folders_without_dr_notes(tmp_path: Path) -> None:
    _touch(tmp_path / "NO DR" / "nurse_visit.pdf")
    _touch(tmp_path / "NO DR" / "readme.txt")
    assert scan_patient_dr_notes(tmp_path) == []


def test_scan_accepts_category_prefix_and_unsupported_extensions_are_ignored(
    tmp_path: Path,
) -> None:
    peds = _touch(tmp_path / "KID" / "peds_dr_note.pdf")
    _touch(tmp_path / "KID" / "dr_note.docx")
    assert scan_patient_dr_notes(tmp_path) == [[peds]]


def test_scan_finds_nested_patient_folders(tmp_path: Path) -> None:
    nested = _touch(tmp_path / "Pending" / "DOE, JANE" / "dr_note.pdf")
    assert scan_patient_dr_notes(tmp_path) == [[nested]]


def test_scan_ignores_results_folders(tmp_path: Path) -> None:
    note = _touch(tmp_path / "dr_note.pdf")
    _touch(tmp_path / "results" / "dr_copy.pdf")
    assert scan_patient_dr_notes(tmp_path) == [[note]]


def test_scan_treats_root_with_dr_notes_as_a_patient(tmp_path: Path) -> None:
    note = _touch(tmp_path / "dr_note.pdf")
    assert scan_patient_dr_notes(tmp_path) == [[note]]


# ---------------------------------------------------------------------------
# DrValidityPipeline
# ---------------------------------------------------------------------------

def _router(texts: dict[str, str], failing: set[str] | None = None) -> MagicMock:
    failing = failing or set()

    async def extract(path: Path) -> ExtractedText:
        if path.name in failing:
            raise OSError(f"cannot read {path.name}")
        return ExtractedText(file_path=path, text=texts[path.name], num_pages=1)

    extractor = MagicMock()
    extractor.extract = AsyncMock(side_effect=extract)
    router = MagicMock()
    router.route = MagicMock(return_value=extractor)
    return router


def _classifier() -> MagicMock:
    async def classify(metadata: DocumentMetadata, dr_paths: list[Path]) -> DrValidityResult:
        return DrValidityResult(
            patient_folder=metadata.file_path.parent,
            dr_paths=dr_paths,
            patient_group=PatientGroup.ADULT,
            diagnosis_check=DiagnosisCheck.EXISTS,
            verdict=ValidityVerdict.VALID,
        )

    clf = MagicMock()
    clf.classify = AsyncMock(side_effect=classify)
    return clf


def _make(router: MagicMock, classifier: MagicMock) -> DrValidityPipeline:
    return DrValidityPipeline(
        router=router, classifier=classifier, folder_check=RequiredNotesCheck(roles=("dr",))
    )


async def test_pipeline_combines_all_dr_notes_of_a_patient(tmp_path: Path) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    b = tmp_path / "P" / "dr_b.pdf"
    clf = _classifier()
    pipeline = _make(
        router=_router({"dr_a.pdf": "Diagnosis: GERD", "dr_b.pdf": "Needs home services"}),
        classifier=clf,
    )

    results = await pipeline.run([[a, b]])

    assert len(results) == 1 and results[0].success
    assert results[0].patient_folder == tmp_path / "P"
    metadata, dr_paths = clf.classify.await_args.args
    assert dr_paths == [a, b]
    assert metadata.file_path == a
    assert "--- dr_a.pdf ---\nDiagnosis: GERD" in metadata.raw_text
    assert "--- dr_b.pdf ---\nNeeds home services" in metadata.raw_text


async def test_pipeline_failure_does_not_abort_batch(tmp_path: Path) -> None:
    good = tmp_path / "GOOD" / "dr_a.pdf"
    bad = tmp_path / "BAD" / "dr_b.pdf"
    pipeline = _make(
        router=_router({"dr_a.pdf": "ok", "dr_b.pdf": "ok"}, failing={"dr_b.pdf"}),
        classifier=_classifier(),
    )

    results = await pipeline.run([[good], [bad]])

    assert [r.success for r in results] == [True, False]
    assert results[1].patient_folder == tmp_path / "BAD"
    assert results[1].error is not None and "cannot read" in results[1].error
    assert [(e.stage, e.code) for e in results[1].errors] == [
        (ErrorStage.PIPELINE, ErrorCode.UNEXPECTED)
    ]
    assert results[0].errors == []


async def test_run_folder_scans_patients(tmp_path: Path) -> None:
    _touch(tmp_path / "A" / "dr_a.txt")
    _touch(tmp_path / "B" / "dr_b.txt")
    clf = _classifier()
    router = _router({"dr_a.txt": "a", "dr_b.txt": "b"})
    pipeline = _make(router=router, classifier=clf)

    results = await pipeline.run_folder(tmp_path)

    assert [r.patient_folder.name for r in results] == ["A", "B"]
    assert clf.classify.await_count == 2


async def test_pipeline_logs_timing_per_patient(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    pipeline = _make(router=_router({"dr_a.pdf": "ok"}), classifier=_classifier())

    with caplog.at_level(logging.INFO, logger="classifier.timing"):
        await pipeline.run([[a]])

    messages = [r.getMessage() for r in caplog.records if r.name == "classifier.timing"]
    assert any(
        m.startswith("timing patient=P extract file=dr_a.pdf extractor=MagicMock pages=1 ")
        for m in messages
    )
    assert any(m.startswith("timing patient=P TOTAL ") for m in messages)


async def test_pipeline_logs_timing_even_when_patient_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = tmp_path / "BAD" / "dr_b.pdf"
    pipeline = _make(
        router=_router({"dr_b.pdf": "ok"}, failing={"dr_b.pdf"}), classifier=_classifier()
    )

    with caplog.at_level(logging.INFO, logger="classifier.timing"):
        await pipeline.run([[bad]])

    assert any(
        r.getMessage().startswith("timing patient=BAD TOTAL ")
        for r in caplog.records
        if r.name == "classifier.timing"
    )


async def test_pipeline_step_error_fails_patient_with_structured_errors(tmp_path: Path) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    step_errors = [
        RunError(stage=ErrorStage.CARE, code=ErrorCode.LLM_FAILED, message="LLM error: down"),
        RunError(stage=ErrorStage.AGE, code=ErrorCode.PARSE_FAILED, message="Parse error: x"),
    ]
    clf = MagicMock()
    clf.classify = AsyncMock(side_effect=ClassificationStepError(step_errors))
    pipeline = _make(router=_router({"dr_a.pdf": "ok"}), classifier=clf)

    [result] = await pipeline.run([[a]])

    assert not result.success and result.result is None
    assert result.errors == step_errors
    assert result.error == "care: LLM error: down; age: Parse error: x"


async def test_pipeline_all_blank_notes_fail_without_calling_classifier(tmp_path: Path) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    b = tmp_path / "P" / "dr_b.pdf"
    clf = _classifier()
    pipeline = _make(router=_router({"dr_a.pdf": "  ", "dr_b.pdf": ""}), classifier=clf)

    [result] = await pipeline.run([[a, b]])

    assert not result.success
    assert [(e.code, e.file) for e in result.errors] == [
        (ErrorCode.EMPTY_TEXT, a),
        (ErrorCode.EMPTY_TEXT, b),
    ]
    clf.classify.assert_not_awaited()


async def test_pipeline_some_blank_notes_are_non_fatal(tmp_path: Path) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    b = tmp_path / "P" / "dr_b.pdf"
    clf = _classifier()
    router = _router({"dr_a.pdf": "Diagnosis: GERD", "dr_b.pdf": ""})
    pipeline = _make(router=router, classifier=clf)

    [result] = await pipeline.run([[a, b]])

    assert result.success and result.result is not None
    assert [(e.stage, e.code, e.file) for e in result.errors] == [
        (ErrorStage.EXTRACT, ErrorCode.EMPTY_TEXT, b)
    ]
    clf.classify.assert_awaited_once()


async def test_pipeline_step_error_keeps_earlier_blank_note_errors(tmp_path: Path) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    b = tmp_path / "P" / "dr_b.pdf"
    step_error = RunError(stage=ErrorStage.DIAGNOSIS, code=ErrorCode.LLM_FAILED, message="down")
    clf = MagicMock()
    clf.classify = AsyncMock(side_effect=ClassificationStepError([step_error]))
    pipeline = _make(router=_router({"dr_a.pdf": "ok", "dr_b.pdf": ""}), classifier=clf)

    [result] = await pipeline.run([[a, b]])

    assert [e.code for e in result.errors] == [ErrorCode.EMPTY_TEXT, ErrorCode.LLM_FAILED]


def test_folder_errors_reports_unprocessable_leaf_folders(tmp_path: Path) -> None:
    _touch(tmp_path / "OK" / "dr_a.txt")
    _touch(tmp_path / "NURSE ONLY" / "nurse_visit.txt")
    (tmp_path / "EMPTY").mkdir()
    pipeline = _make(router=_router({}), classifier=_classifier())

    errors = pipeline.folder_errors(tmp_path)

    assert [(e.folder, e.code) for e in errors] == [
        (tmp_path / "EMPTY", ErrorCode.EMPTY_FOLDER),
        (tmp_path / "NURSE ONLY", ErrorCode.MISSING_REQUIRED_NOTES),
    ]


def test_folder_errors_uses_injected_check(tmp_path: Path) -> None:
    _touch(tmp_path / "A" / "dr_a.txt")
    check = MagicMock()
    check.check = MagicMock(return_value=None)
    pipeline = DrValidityPipeline(router=_router({}), classifier=_classifier(), folder_check=check)

    assert pipeline.folder_errors(tmp_path) == []
    check.check.assert_called_once_with(tmp_path / "A")


async def test_injected_semaphore_caps_patients_across_concurrent_runs(tmp_path: Path) -> None:
    import asyncio

    names = [f"dr_{i}.pdf" for i in range(4)]
    paths = [tmp_path / f"P{i}" / name for i, name in enumerate(names)]
    active = peak = 0

    async def classify(metadata: DocumentMetadata, dr_paths: list[Path]) -> DrValidityResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return DrValidityResult(
            patient_folder=metadata.file_path.parent,
            dr_paths=dr_paths,
            patient_group=PatientGroup.ADULT,
            diagnosis_check=DiagnosisCheck.EXISTS,
            verdict=ValidityVerdict.VALID,
        )

    clf = MagicMock()
    clf.classify = AsyncMock(side_effect=classify)
    pipeline = DrValidityPipeline(
        router=_router({n: "text" for n in names}),
        classifier=clf,
        folder_check=RequiredNotesCheck(roles=("dr",)),
        semaphore=asyncio.Semaphore(1),
    )

    await asyncio.gather(
        pipeline.run([[p] for p in paths[:2]]), pipeline.run([[p] for p in paths[2:]])
    )

    assert peak == 1
