from pathlib import Path

from classifier.protocols import ContentExtractor
from config.settings import settings


class DefaultFileRouter:
    """Routes a file path to the registered ContentExtractor for its type."""

    def __init__(self, extractors: dict[str, ContentExtractor]) -> None:
        self._extractors = extractors

    def route(self, file_path: Path) -> ContentExtractor:
        """Return the extractor for this file's extension.

        Raises ValueError for unsupported extensions.
        """
        ext_key = settings.supported_extensions.get(file_path.suffix.lower())
        if not ext_key or ext_key not in self._extractors:
            raise ValueError(
                f"No extractor registered for extension '{file_path.suffix}'"
            )
        return self._extractors[ext_key]
