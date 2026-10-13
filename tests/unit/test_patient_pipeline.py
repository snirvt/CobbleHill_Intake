import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from classifier.models import (
    DiagnosisCheck,
    DocumentMetadata,
    DrValidityResult,
    ExtractedText,
    PatientGroup,
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


async def test_pipeline_combines_all_dr_notes_of_a_patient(tmp_path: Path) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    b = tmp_path / "P" / "dr_b.pdf"
    clf = _classifier()
    pipeline = DrValidityPipeline(
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
    pipeline = DrValidityPipeline(
        router=_router({"dr_a.pdf": "ok", "dr_b.pdf": "ok"}, failing={"dr_b.pdf"}),
        classifier=_classifier(),
    )

    results = await pipeline.run([[good], [bad]])

    assert [r.success for r in results] == [True, False]
    assert results[1].patient_folder == tmp_path / "BAD"
    assert results[1].error is not None and "cannot read" in results[1].error


async def test_run_folder_scans_patients(tmp_path: Path) -> None:
    _touch(tmp_path / "A" / "dr_a.txt")
    _touch(tmp_path / "B" / "dr_b.txt")
    clf = _classifier()
    router = _router({"dr_a.txt": "a", "dr_b.txt": "b"})
    pipeline = DrValidityPipeline(router=router, classifier=clf)

    results = await pipeline.run_folder(tmp_path)

    assert [r.patient_folder.name for r in results] == ["A", "B"]
    assert clf.classify.await_count == 2


async def test_pipeline_logs_timing_per_patient(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    a = tmp_path / "P" / "dr_a.pdf"
    pipeline = DrValidityPipeline(router=_router({"dr_a.pdf": "ok"}), classifier=_classifier())

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
    pipeline = DrValidityPipeline(
        router=_router({"dr_b.pdf": "ok"}, failing={"dr_b.pdf"}), classifier=_classifier()
    )

    with caplog.at_level(logging.INFO, logger="classifier.timing"):
        await pipeline.run([[bad]])

    assert any(
        r.getMessage().startswith("timing patient=BAD TOTAL ")
        for r in caplog.records
        if r.name == "classifier.timing"
    )
