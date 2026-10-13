import asyncio
import logging
import time
from pathlib import Path

from classifier.models import (
    DocumentMetadata,
    DrValidityPipelineResult,
    ExtractedText,
    PatientMetadata,
)
from classifier.naming import concat_notes, parse_note
from classifier.protocols import ContentExtractor, DrValidityClassifier, FileRouter
from classifier.timing import ExtractTiming, PatientTimings, current_timings
from config.settings import settings

logger = logging.getLogger(__name__)


def scan_patient_dr_notes(root: Path) -> list[list[Path]]:
    """Group dr notes by patient folder: one sorted list of dr files per folder.

    Walks root recursively. Every folder holding at least one dr note (any category
    prefix) is one patient; nurse notes and other files are ignored.
    """
    patients: list[list[Path]] = []
    for folder in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        dr_files = sorted(f for f in folder.iterdir() if parse_note(f, role="dr") is not None)
        if dr_files:
            patients.append(dr_files)
            if len(dr_files) > 1:
                logger.info(
                    "Folder %s has %d dr notes — combining %s",
                    folder,
                    len(dr_files),
                    [f.name for f in dr_files],
                )
    return patients


class DrValidityPipeline:
    """Orchestrates per patient: route → extract every dr note → combine → classify."""

    def __init__(self, router: FileRouter, classifier: DrValidityClassifier) -> None:
        self._router = router
        self._classifier = classifier

    async def run(self, patients: list[list[Path]]) -> list[DrValidityPipelineResult]:
        """Process patients concurrently, up to max_concurrent_files at a time.

        Each entry is one patient's dr notes; their text is combined under per-file
        headers. A routing/IO failure yields success=False instead of aborting the batch.
        """
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(dr_paths: list[Path]) -> DrValidityPipelineResult:
            folder = dr_paths[0].parent
            async with sem:
                timings = PatientTimings(patient=folder.name)
                current_timings.set(timings)
                started = time.perf_counter()
                try:
                    extracted = await asyncio.gather(
                        *(self._timed_extract(p, timings) for p in dr_paths)
                    )
                    timings.extract_wall_s = time.perf_counter() - started
                    metadata = DocumentMetadata(
                        file_path=dr_paths[0],
                        raw_text=concat_notes(dr_paths, [x.text for x in extracted]),
                        meta=PatientMetadata(),
                    )
                    classify_started = time.perf_counter()
                    result = await self._classifier.classify(metadata, dr_paths)
                    timings.classify_wall_s = time.perf_counter() - classify_started
                    return DrValidityPipelineResult(
                        patient_folder=folder, dr_paths=dr_paths, success=True, result=result
                    )
                except Exception as exc:
                    logger.error("Dr-validity pipeline failed for %s: %s", folder, exc)
                    return DrValidityPipelineResult(
                        patient_folder=folder, dr_paths=dr_paths, success=False, error=str(exc)
                    )
                finally:
                    timings.total_s = time.perf_counter() - started
                    timings.log()

        return list(await asyncio.gather(*[process(p) for p in patients]))

    async def _timed_extract(self, path: Path, timings: PatientTimings) -> ExtractedText:
        """Extract one file and record how long it took and with which extractor."""
        extractor: ContentExtractor = self._router.route(path)
        started = time.perf_counter()
        extracted = await extractor.extract(path)
        timings.extracts.append(
            ExtractTiming(
                file=path.name,
                extractor=type(extractor).__name__,
                pages=extracted.num_pages,
                seconds=time.perf_counter() - started,
            )
        )
        return extracted

    async def run_folder(self, root: Path) -> list[DrValidityPipelineResult]:
        """Scan root for patient folders with dr notes and process them."""
        patients = scan_patient_dr_notes(root)
        if not patients:
            logger.warning("No dr notes found in %s", root)
        return await self.run(patients)
