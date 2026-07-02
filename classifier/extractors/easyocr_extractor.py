import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

from classifier.models import ExtractedText

logger = logging.getLogger(__name__)


class OcrReader(Protocol):
    """Minimal easyocr.Reader interface, so it can be injected and mocked."""

    def readtext(self, image: str, **kwargs: Any) -> list[Any]: ...


class EasyOcrExtractor:
    """ContentExtractor for image files using easyocr (image -> text directly).

    No PDF conversion and no node/liteparse dependency. The heavy easyocr Reader
    loads its model once at construction, so it is injected (built once at wiring
    time) rather than created per call, and is trivially mockable in tests.
    """

    def __init__(self, reader: OcrReader, paragraph: bool = True) -> None:
        self._reader = reader
        self._paragraph = paragraph

    def _read(self, file_path: Path) -> str:
        # detail=0 returns plain strings instead of (bbox, text, confidence) tuples.
        lines = self._reader.readtext(
            str(file_path), detail=0, paragraph=self._paragraph
        )
        return "\n".join(str(line) for line in lines)

    async def extract(self, file_path: Path) -> ExtractedText:
        """OCR an image file and return its text content (always one page)."""
        logger.debug("Running easyocr on image: %s", file_path)
        try:
            text = await asyncio.to_thread(self._read, file_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"easyocr failed on {file_path}: {exc}") from exc
        return ExtractedText(file_path=file_path, text=text, num_pages=1)


def create_easyocr_reader(
    languages: list[str],
    gpu: bool = False,
    model_storage_directory: str | None = None,
    download_enabled: bool = True,
) -> OcrReader:
    """Build an easyocr.Reader. Imported lazily so easyocr is only needed at runtime."""
    import easyocr  # type: ignore[import-untyped]

    logger.info(
        "Loading easyocr Reader (langs=%s, gpu=%s, offline=%s)",
        languages,
        gpu,
        not download_enabled,
    )
    reader: OcrReader = easyocr.Reader(
        languages,
        gpu=gpu,
        model_storage_directory=model_storage_directory,
        download_enabled=download_enabled,
    )
    return reader
