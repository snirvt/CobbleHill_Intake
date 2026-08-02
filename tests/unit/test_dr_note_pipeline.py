from pathlib import Path

import pytest

from classifier.dr_note_pipeline import DrNotePipeline, scan_dr_notes
from classifier.models import DocumentMetadata, ExtractedText


class _FakeExtractor:
    async def extract(self, file_path: Path) -> ExtractedText:
        return ExtractedText(file_path=file_path, text=f"text of {file_path.name}", num_pages=1)


class _FakeRouter:
    def route(self, file_path: Path) -> _FakeExtractor:
        return _FakeExtractor()


class _FailingRouter:
    def route(self, file_path: Path) -> _FakeExtractor:
        raise ValueError("no extractor")


async def _echo(metadata: DocumentMetadata) -> str:
    """extract_fn stub: return the file's raw_text."""
    return metadata.raw_text


def _on_error(path: Path, exc: Exception) -> str:
    return f"ERROR {path.name}: {exc}"


def _make_pipeline(router: object) -> DrNotePipeline[str]:
    return DrNotePipeline(router=router, extract_fn=_echo, error_fn=_on_error)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_applies_extract_fn_per_file() -> None:
    pipeline = _make_pipeline(_FakeRouter())

    results = await pipeline.run([Path("a.pdf"), Path("b.txt")])

    assert results == ["text of a.pdf", "text of b.txt"]


@pytest.mark.asyncio
async def test_run_uses_error_fn_on_failure() -> None:
    pipeline = _make_pipeline(_FailingRouter())

    results = await pipeline.run([Path("a.pdf")])

    assert results[0].startswith("ERROR a.pdf:")


@pytest.mark.asyncio
async def test_run_folder_processes_only_dr_notes_recursively(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "dr_progress_note.pdf").write_text("x")
    (tmp_path / "sub" / "dr_visit.txt").write_text("x")
    (tmp_path / "nurse_visit.txt").write_text("y")
    (tmp_path / "ignore.md").write_text("z")

    pipeline = _make_pipeline(_FakeRouter())
    results = await pipeline.run_folder(tmp_path)

    assert sorted(results) == ["text of dr_progress_note.pdf", "text of dr_visit.txt"]


@pytest.mark.asyncio
async def test_run_folder_empty_when_no_dr_notes(tmp_path: Path) -> None:
    (tmp_path / "nurse_visit.txt").write_text("y")

    pipeline = _make_pipeline(_FakeRouter())
    results = await pipeline.run_folder(tmp_path)

    assert results == []


def test_scan_dr_notes_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "dr_b.pdf").write_text("x")
    (tmp_path / "dr_a.txt").write_text("x")
    (tmp_path / "sub" / "nurse_visit.txt").write_text("y")
    (tmp_path / "notes.md").write_text("z")

    found = [p.name for p in scan_dr_notes(tmp_path)]

    assert found == ["dr_a.txt", "dr_b.pdf"]


def test_scan_dr_notes_includes_all_categories(tmp_path: Path) -> None:
    (tmp_path / "dr_a.pdf").write_text("x")
    (tmp_path / "hospital_dr_b.pdf").write_text("x")
    (tmp_path / "peds_dr_c.pdf").write_text("x")
    (tmp_path / "hospital_nurse_d.pdf").write_text("y")
    (tmp_path / "peds_summary.pdf").write_text("z")

    found = [p.name for p in scan_dr_notes(tmp_path)]

    assert found == ["dr_a.pdf", "hospital_dr_b.pdf", "peds_dr_c.pdf"]
