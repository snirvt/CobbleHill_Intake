import logging
import os
from pathlib import Path

from liteparse import LiteParse
from liteparse.types import ParseError

from classifier.models import ExtractedText
from config.settings import settings

logger = logging.getLogger(__name__)


class PdfExtractor:
    """ContentExtractor for PDF files using the LiteParse npm CLI."""

    def __init__(self, node_bin_path: str = settings.node_bin_path) -> None:
        if node_bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = node_bin_path + ":" + os.environ["PATH"]
        self._parser = LiteParse()

    async def extract(self, file_path: Path) -> ExtractedText:
        """Parse a PDF and return its full text content."""
        logger.debug("Parsing PDF: %s", file_path)
        try:
            result = await self._parser.parse_async(file_path)
        except ParseError as exc:
            raise RuntimeError(f"LiteParse failed on {file_path}: {exc}") from exc
        logger.debug("Parsed %s — %d page(s)", file_path.name, result.num_pages)
        return ExtractedText(
            file_path=file_path,
            text=result.text,
            num_pages=result.num_pages,
        )
