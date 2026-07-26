from pathlib import Path

import pytest

from classifier.diagnosis_pipeline import DiagnosisPipeline
from classifier.models import (
    Diagnosis,
    DiagnosisExtractionResult,
    DocumentMetadata,
    ExtractedText,
)


class _FakeExtractor:
    """ContentExtractor stub returning fixed text for a file."""

    async def extract(self, file_path: Path) -> ExtractedText:
        return ExtractedText(file_path=file_path, text=f"text of {file_path.name}", num_pages=1)


class _FakeRouter:
    def route(self, file_path: Path) -> _FakeExtractor:
        return _FakeExtractor()


class _FakeDiagnosisExtractor:
    """DiagnosisExtractor stub echoing the file's raw_text as a diagnosis name."""

    async def extract_diagnoses(self, metadata: DocumentMetadata) -> DiagnosisExtractionResult:
        return DiagnosisExtractionResult(
            file_path=metadata.file_path,
            diagnoses=[Diagnosis(name=metadata.raw_text)],
        )


class _FailingRouter:
    def route(self, file_path: Path) -> _FakeExtractor:
        raise ValueError("no extractor")


@pytest.mark.asyncio
async def test_run_processes_each_file() -> None:
    pipeline = DiagnosisPipeline(_FakeRouter(), _FakeDiagnosisExtractor())

    results = await pipeline.run([Path("a.pdf"), Path("b.txt")])

    assert [r.file_path.name for r in results] == ["a.pdf", "b.txt"]
    assert results[0].diagnoses[0].name == "text of a.pdf"


@pytest.mark.asyncio
async def test_run_returns_empty_result_on_failure() -> None:
    pipeline = DiagnosisPipeline(_FailingRouter(), _FakeDiagnosisExtractor())

    results = await pipeline.run([Path("a.pdf")])

    assert results[0].file_path == Path("a.pdf")
    assert results[0].diagnoses == []


@pytest.mark.asyncio
async def test_run_folder_processes_only_dr_notes_recursively(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "dr_progress_note.pdf").write_text("x")
    (tmp_path / "sub" / "dr_visit.txt").write_text("x")
    (tmp_path / "nurse_visit.txt").write_text("y")
    (tmp_path / "sub" / "nurse_visit_2.txt").write_text("y")
    (tmp_path / "ignore.md").write_text("z")

    pipeline = DiagnosisPipeline(_FakeRouter(), _FakeDiagnosisExtractor())
    results = await pipeline.run_folder(tmp_path)

    names = sorted(r.file_path.name for r in results)
    assert names == ["dr_progress_note.pdf", "dr_visit.txt"]


@pytest.mark.asyncio
async def test_run_folder_empty_when_no_dr_notes(tmp_path: Path) -> None:
    (tmp_path / "nurse_visit.txt").write_text("y")
    (tmp_path / "note.md").write_text("z")

    pipeline = DiagnosisPipeline(_FakeRouter(), _FakeDiagnosisExtractor())
    results = await pipeline.run_folder(tmp_path)

    assert results == []
