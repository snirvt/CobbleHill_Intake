from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from classifier.pipeline import Pipeline


async def test_pipeline_run_returns_success_result(
    mock_router: MagicMock,
    mock_meta_extractor: MagicMock,
    mock_classifier: MagicMock,
    sample_pdf: Path,
) -> None:
    pipeline = Pipeline(
        router=mock_router,
        meta_extractor=mock_meta_extractor,
        classifier=mock_classifier,
    )
    results = await pipeline.run([sample_pdf])

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].result is not None
    assert results[0].result.category == "TEST"


async def test_pipeline_run_captures_extractor_error(
    mock_router: MagicMock,
    mock_meta_extractor: MagicMock,
    mock_classifier: MagicMock,
    sample_pdf: Path,
) -> None:
    mock_router.route.side_effect = ValueError("No extractor registered for extension '.pdf'")

    pipeline = Pipeline(
        router=mock_router,
        meta_extractor=mock_meta_extractor,
        classifier=mock_classifier,
    )
    results = await pipeline.run([sample_pdf])

    assert len(results) == 1
    assert results[0].success is False
    assert "No extractor" in (results[0].error or "")


async def test_pipeline_run_folder_skips_unsupported_files(
    tmp_path: Path,
    mock_router: MagicMock,
    mock_meta_extractor: MagicMock,
    mock_classifier: MagicMock,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "doc.pdf").write_bytes(b"%PDF")
    (folder / "notes.txt").write_text("ignore me")

    pipeline = Pipeline(
        router=mock_router,
        meta_extractor=mock_meta_extractor,
        classifier=mock_classifier,
    )
    results = await pipeline.run_folder(folder)

    assert len(results) == 1
    assert results[0].file_path.name == "doc.pdf"


async def test_pipeline_processes_multiple_files(
    tmp_path: Path,
    mock_router: MagicMock,
    mock_meta_extractor: MagicMock,
    mock_classifier: MagicMock,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(3):
        (folder / f"doc{i}.pdf").write_bytes(b"%PDF")

    pipeline = Pipeline(
        router=mock_router,
        meta_extractor=mock_meta_extractor,
        classifier=mock_classifier,
    )
    results = await pipeline.run_folder(folder)

    assert len(results) == 3
    assert all(r.success for r in results)
