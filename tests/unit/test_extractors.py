from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classifier.extractors.pdf import PdfExtractor
from classifier.models import ExtractedText


@pytest.fixture
def mock_parse_result() -> MagicMock:
    result = MagicMock()
    result.text = "extracted text content"
    result.num_pages = 3
    return result


async def test_pdf_extractor_returns_extracted_text(
    tmp_path: Path, mock_parse_result: MagicMock
) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(return_value=mock_parse_result)

        extractor = PdfExtractor(node_bin_path="/fake/bin")
        result = await extractor.extract(pdf)

    assert isinstance(result, ExtractedText)
    assert result.text == "extracted text content"
    assert result.num_pages == 3
    assert result.file_path == pdf


async def test_pdf_extractor_raises_on_parse_error(tmp_path: Path) -> None:
    from liteparse.types import ParseError

    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a pdf")

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(side_effect=ParseError("corrupt"))

        extractor = PdfExtractor(node_bin_path="/fake/bin")
        with pytest.raises(RuntimeError, match="LiteParse failed"):
            await extractor.extract(pdf)


async def test_pdf_extractor_sets_path_env(tmp_path: Path) -> None:
    import os

    with patch("classifier.extractors.pdf.LiteParse"):
        extractor = PdfExtractor(node_bin_path="/custom/node/bin")

    assert "/custom/node/bin" in os.environ.get("PATH", "")
