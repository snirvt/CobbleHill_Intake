import asyncio
import logging
import tempfile
from pathlib import Path

import img2pdf  # type: ignore[import-untyped]
from PIL import Image, UnidentifiedImageError

from classifier.extractors.pdf import PdfExtractor
from classifier.extractors.preprocess import preprocess_image
from classifier.models import ExtractedText
from config.settings import settings

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
    """ContentExtractor for image files (jpg/jpeg/png).

    Preprocesses the image with Pillow (flatten alpha, downscale to a size cap,
    grayscale + Otsu binarize), tags it at ``dpi`` so the downstream render is
    1:1 with the preprocessed pixels, converts it to a single-page PDF with
    img2pdf, then delegates to a PdfExtractor so liteparse's bundled OCR turns
    it into text.

    The dpi tag matters: img2pdf sizes the PDF page from the image's dpi, and
    liteparse re-renders that page at ``dpi``. Matching them renders at native
    resolution instead of blindly upscaling a raster (which blurs the text and,
    past the OCR engine's size limit, yields empty output). No system
    dependencies (ImageMagick/tesseract) required.
    """

    def __init__(
        self,
        pdf_extractor: PdfExtractor,
        dpi: int = settings.ocr_dpi,
        max_dimension: int = settings.ocr_max_dimension,
        binarize: bool = settings.ocr_binarize,
    ) -> None:
        self._pdf_extractor = pdf_extractor
        self._dpi = dpi
        self._max_dimension = max_dimension
        self._binarize = binarize

    def _to_pdf_bytes(self, file_path: Path) -> bytes:
        """Preprocess the image and encode it as a single-page PDF (sync)."""
        with Image.open(file_path) as image:
            processed = preprocess_image(
                image, self._max_dimension, binarize_image=self._binarize
            )
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as img_fh:
                processed.save(img_fh.name, format="PNG", dpi=(self._dpi, self._dpi))
                return img2pdf.convert(img_fh.name)  # type: ignore[no-any-return]

    async def extract(self, file_path: Path) -> ExtractedText:
        """Preprocess an image, convert to PDF, and return its OCR'd text."""
        logger.debug("Preprocessing and converting image to PDF: %s", file_path)
        try:
            pdf_bytes = await asyncio.to_thread(self._to_pdf_bytes, file_path)
        except (*_IMG2PDF_ERRORS, OSError, UnidentifiedImageError) as exc:
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
