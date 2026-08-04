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


async def test_pdf_extractor_passes_dpi_to_liteparse(
    tmp_path: Path, mock_parse_result: MagicMock
) -> None:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(return_value=mock_parse_result)

        extractor = PdfExtractor(node_bin_path="/fake/bin", dpi=300)
        await extractor.extract(pdf)

    assert instance.parse_async.await_args.kwargs["dpi"] == 300


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


def _page(page_num: int, text: str) -> MagicMock:
    page = MagicMock()
    page.pageNum = page_num
    page.text = text
    return page


def _result(pages: list[MagicMock], num_pages: int = 50) -> MagicMock:
    result = MagicMock()
    result.pages = pages
    result.text = "\n".join(p.text for p in pages)
    result.num_pages = num_pages
    return result


HEADER = "Admission/ROC Summary"


async def test_pdf_extractor_returns_hinted_page_when_header_present(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "hospital_nurse_visit.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    hinted = _result([_page(48, f"{HEADER}\nvitals stable")])

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(return_value=hinted)

        extractor = PdfExtractor(node_bin_path="/fake/bin")
        result = await extractor.extract(pdf)

    assert result.text == f"{HEADER}\nvitals stable"
    # header found on the hinted page — no full-document reparse needed
    assert instance.parse_async.await_count == 1
    assert instance.parse_async.await_args.kwargs["target_pages"] == "48"


async def test_pdf_extractor_scans_all_pages_when_hint_misses(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "hospital_nurse_visit.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    hint_miss = _result([_page(48, "unrelated content on wrong page")])
    full = _result(
        [_page(48, "unrelated"), _page(49, f"case-INSENSITIVE {HEADER.upper()} here")]
    )

    def parse_side_effect(_path: Path, *, dpi: int, target_pages: str | None):
        return hint_miss if target_pages == "48" else full

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(side_effect=parse_side_effect)

        extractor = PdfExtractor(node_bin_path="/fake/bin")
        result = await extractor.extract(pdf)

    assert HEADER.upper() in result.text
    assert instance.parse_async.await_count == 2


async def test_pdf_extractor_falls_back_to_full_text_when_header_absent(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "hospital_nurse_visit.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    hint_miss = _result([_page(48, "no header here")])
    full = _result([_page(1, "page one"), _page(2, "page two")])

    def parse_side_effect(_path: Path, *, dpi: int, target_pages: str | None):
        return hint_miss if target_pages == "48" else full

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(side_effect=parse_side_effect)

        extractor = PdfExtractor(node_bin_path="/fake/bin")
        result = await extractor.extract(pdf)

    assert result.text == full.text
    assert result.num_pages == 50


async def test_pdf_extractor_ignores_targeting_for_non_matching_name(
    tmp_path: Path, mock_parse_result: MagicMock
) -> None:
    # dr note has no page-target rule — full text, no target_pages narrowing
    pdf = tmp_path / "dr_note.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch("classifier.extractors.pdf.LiteParse") as MockLiteParse:
        instance = MockLiteParse.return_value
        instance.parse_async = AsyncMock(return_value=mock_parse_result)

        extractor = PdfExtractor(node_bin_path="/fake/bin")
        result = await extractor.extract(pdf)

    assert result.text == "extracted text content"
    assert instance.parse_async.await_args.kwargs["target_pages"] is None


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
