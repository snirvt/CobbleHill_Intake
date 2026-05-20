import logging
from pathlib import Path

import aiofiles

from classifier.models import ExtractedText

logger = logging.getLogger(__name__)


class PlaintextExtractor:
    """ContentExtractor for plain text files."""

    async def extract(self, file_path: Path) -> ExtractedText:
        """Read a .txt file and return its content as ExtractedText."""
        logger.debug("Reading plaintext: %s", file_path)
        try:
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                text = await f.read()
        except OSError as exc:
            raise RuntimeError(f"Failed to read {file_path}: {exc}") from exc
        logger.debug("Read %s — %d chars", file_path.name, len(text))
        return ExtractedText(file_path=file_path, text=text, num_pages=1)
