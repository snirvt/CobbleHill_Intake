import asyncio
import logging
import tempfile
from pathlib import Path

import img2pdf  # type: ignore[import-untyped]

from classifier.extractors.pdf import PdfExtractor
from classifier.models import ExtractedText

logger = logging.getLogger(__name__)

# img2pdf failures that mean "this image cannot be converted"; all subclass Exception directly.
_IMG2PDF_ERRORS = (
    img2pdf.ImageOpenError,
    img2pdf.JpegColorspaceError,
    img2pdf.ExifOrientationError,
    img2pdf.UnsupportedColorspaceError,
    img2pdf.AlphaChannelError,
)


class ImageExtractor:
    """ContentExtractor for image files (jpg/jpeg).

    Converts the image to a single-page PDF with img2pdf, then delegates to a
    PdfExtractor so liteparse's bundled OCR turns it into text. No system
    dependencies (ImageMagick/tesseract) required.
    """

    def __init__(self, pdf_extractor: PdfExtractor) -> None:
        self._pdf_extractor = pdf_extractor

    async def extract(self, file_path: Path) -> ExtractedText:
        """Convert an image to PDF and return its OCR'd text content."""
        logger.debug("Converting image to PDF: %s", file_path)
        try:
            pdf_bytes = await asyncio.to_thread(img2pdf.convert, str(file_path))
        except (*_IMG2PDF_ERRORS, OSError) as exc:
            raise RuntimeError(f"Image-to-PDF conversion failed on {file_path}: {exc}") from exc

        tmp_pdf: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
                fh.write(pdf_bytes)
                tmp_pdf = Path(fh.name)
            parsed = await self._pdf_extractor.extract(tmp_pdf)
        finally:
            if tmp_pdf is not None:
                tmp_pdf.unlink(missing_ok=True)

        # Preserve the original image path, not the temp PDF path.
        return ExtractedText(
            file_path=file_path,
            text=parsed.text,
            num_pages=parsed.num_pages,
        )
