import asyncio
import logging
from pathlib import Path

from classifier.models import DocumentMetadata, PipelineResult
from classifier.protocols import Classifier, FileRouter, MetadataExtractor
from config.settings import settings

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates: route → extract text → extract metadata → classify."""

    def __init__(
        self,
        router: FileRouter,
        meta_extractor: MetadataExtractor,
        classifier: Classifier,
    ) -> None:
        self._router = router
        self._meta_extractor = meta_extractor
        self._classifier = classifier

    async def run(self, files: list[Path]) -> list[PipelineResult]:
        """Process files concurrently, up to max_concurrent_files at a time."""
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(file_path: Path) -> PipelineResult:
            async with sem:
                try:
                    extractor = self._router.route(file_path)
                    text_result = await extractor.extract(file_path)
                    fields = self._meta_extractor.extract(text_result.text)
                    doc_meta = DocumentMetadata(
                        file_path=file_path,
                        raw_text=text_result.text,
                        meta=fields.meta,
                        hpi=fields.hpi,
                        examination=fields.examination,
                        complaints=fields.complaints,
                        medical_history=fields.medical_history,
                        surgical_history=fields.surgical_history,
                        hospitalization=fields.hospitalization,
                        assessment=fields.assessment,
                        ros=fields.ros,
                        medications=fields.medications,
                        plan=fields.plan,
                        procedure_codes=fields.procedure_codes,
                        preventive_medicine=fields.preventive_medicine,
                    )
                    classification = await self._classifier.classify(doc_meta)
                    return PipelineResult(
                        file_path=file_path, success=True, result=classification
                    )
                except Exception as exc:
                    logger.error("Pipeline failed for %s: %s", file_path, exc)
                    return PipelineResult(
                        file_path=file_path, success=False, error=str(exc)
                    )

        return list(await asyncio.gather(*[process(f) for f in files]))

    async def run_folder(self, folder: Path) -> list[PipelineResult]:
        """Scan folder for supported files and run the pipeline."""
        files = [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in settings.supported_extensions
        ]
        if not files:
            logger.warning("No supported files found in %s", folder)
        return await self.run(files)
