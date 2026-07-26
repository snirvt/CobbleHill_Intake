import asyncio
import logging
from pathlib import Path

from classifier.models import (
    DiagnosisExtractionResult,
    DocumentMetadata,
    PatientMetadata,
)
from classifier.protocols import DiagnosisExtractor, FileRouter
from config.settings import settings

logger = logging.getLogger(__name__)


class DiagnosisPipeline:
    """Orchestrates: route → extract text → extract diagnoses, for one or many files.

    The diagnosis extractor reads only raw text, so this pipeline works for both dr
    and nurse notes and skips note-type-specific metadata parsing.
    """

    def __init__(self, router: FileRouter, extractor: DiagnosisExtractor) -> None:
        self._router = router
        self._extractor = extractor

    async def run(self, files: list[Path]) -> list[DiagnosisExtractionResult]:
        """Extract diagnoses from files concurrently, up to max_concurrent_files."""
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(file_path: Path) -> DiagnosisExtractionResult:
            async with sem:
                try:
                    content_extractor = self._router.route(file_path)
                    text_result = await content_extractor.extract(file_path)
                    doc_meta = DocumentMetadata(
                        file_path=file_path,
                        raw_text=text_result.text,
                        meta=PatientMetadata(),
                    )
                    return await self._extractor.extract_diagnoses(doc_meta)
                except Exception as exc:
                    logger.error("Diagnosis pipeline failed for %s: %s", file_path, exc)
                    return DiagnosisExtractionResult(file_path=file_path, diagnoses=[])

        return list(await asyncio.gather(*[process(f) for f in files]))

    async def run_folder(self, folder: Path) -> list[DiagnosisExtractionResult]:
        """Scan folder recursively for supported files and extract diagnoses."""
        files = [
            f
            for f in sorted(folder.rglob("*"))
            if f.is_file() and f.suffix.lower() in settings.supported_extensions
        ]
        if not files:
            logger.warning("No supported files found in %s", folder)
        return await self.run(files)
