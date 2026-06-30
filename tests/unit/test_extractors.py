from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from classifier.extractors.image import ImageExtractor
from classifier.extractors.pdf import PdfExtractor
from classifier.extractors.plaintext import PlaintextExtractor
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


async def test_image_extractor_converts_and_delegates_to_pdf(tmp_path: Path) -> None:
    img = tmp_path / "note.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    pdf_extractor = MagicMock()
    pdf_extractor.extract = AsyncMock(
        return_value=ExtractedText(
            file_path=Path("/tmp/whatever.pdf"), text="ocr text", num_pages=1
        )
    )

    with patch(
        "classifier.extractors.image.img2pdf.convert", return_value=b"%PDF-1.4"
    ) as mock_convert:
        extractor = ImageExtractor(pdf_extractor)
        result = await extractor.extract(img)

    mock_convert.assert_called_once_with(str(img))
    # delegated to the injected PdfExtractor with a generated temp .pdf path
    delegated_path = pdf_extractor.extract.await_args.args[0]
    assert delegated_path.suffix == ".pdf"
    assert isinstance(result, ExtractedText)
    assert result.text == "ocr text"
    assert result.num_pages == 1
    # original image path is preserved, not the temp pdf path
    assert result.file_path == img


async def test_image_extractor_raises_on_conversion_error(tmp_path: Path) -> None:
    import img2pdf

    img = tmp_path / "bad.jpg"
    img.write_bytes(b"not an image")

    pdf_extractor = MagicMock()
    pdf_extractor.extract = AsyncMock()

    with patch(
        "classifier.extractors.image.img2pdf.convert",
        side_effect=img2pdf.ImageOpenError("cannot read"),
    ):
        extractor = ImageExtractor(pdf_extractor)
        with pytest.raises(RuntimeError, match="Image-to-PDF conversion failed"):
            await extractor.extract(img)

    pdf_extractor.extract.assert_not_awaited()


async def test_plaintext_extractor_returns_extracted_text(tmp_path: Path) -> None:
    txt = tmp_path / "nurse_visit.txt"
    txt.write_text("Patient seen today. Blood pressure normal.", encoding="utf-8")

    extractor = PlaintextExtractor()
    result = await extractor.extract(txt)

    assert isinstance(result, ExtractedText)
    assert result.text == "Patient seen today. Blood pressure normal."
    assert result.num_pages == 1
    assert result.file_path == txt


async def test_plaintext_extractor_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.txt"

    extractor = PlaintextExtractor()
    with pytest.raises(RuntimeError, match="Failed to read"):
        await extractor.extract(missing)
