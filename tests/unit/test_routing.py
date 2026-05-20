from pathlib import Path
from unittest.mock import MagicMock

import pytest

from classifier.routing import DefaultFileRouter


@pytest.fixture
def pdf_extractor() -> MagicMock:
    return MagicMock()


@pytest.fixture
def txt_extractor() -> MagicMock:
    return MagicMock()


@pytest.fixture
def router(pdf_extractor: MagicMock, txt_extractor: MagicMock) -> DefaultFileRouter:
    return DefaultFileRouter({"pdf": pdf_extractor, "txt": txt_extractor})


def test_routes_pdf_to_pdf_extractor(
    router: DefaultFileRouter, pdf_extractor: MagicMock, tmp_path: Path
) -> None:
    pdf = tmp_path / "doc.pdf"
    result = router.route(pdf)
    assert result is pdf_extractor


def test_routes_uppercase_pdf_extension(
    router: DefaultFileRouter, pdf_extractor: MagicMock, tmp_path: Path
) -> None:
    pdf = tmp_path / "DOC.PDF"
    result = router.route(pdf)
    assert result is pdf_extractor


def test_routes_txt_to_txt_extractor(
    router: DefaultFileRouter, txt_extractor: MagicMock, tmp_path: Path
) -> None:
    txt = tmp_path / "nurse_visit.txt"
    result = router.route(txt)
    assert result is txt_extractor


def test_raises_for_unsupported_extension(
    router: DefaultFileRouter, tmp_path: Path
) -> None:
    docx = tmp_path / "doc.docx"
    with pytest.raises(ValueError, match="No extractor registered"):
        router.route(docx)


def test_raises_for_no_extension(router: DefaultFileRouter, tmp_path: Path) -> None:
    no_ext = tmp_path / "noextension"
    with pytest.raises(ValueError, match="No extractor registered"):
        router.route(no_ext)
