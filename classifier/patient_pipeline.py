import asyncio
import logging
import time
from pathlib import Path

from classifier.folder_checks import leaf_folders
from classifier.models import (
    ClassificationStepError,
    DocumentMetadata,
    DrValidityPipelineResult,
    ErrorCode,
    ErrorStage,
    ExtractedText,
    PatientMetadata,
    RunError,
)
from classifier.naming import concat_notes, in_results_folder, parse_note
from classifier.protocols import ContentExtractor, DrValidityClassifier, FileRouter, FolderCheck
from classifier.timing import ExtractTiming, PatientTimings, current_timings
from config.settings import settings

logger = logging.getLogger(__name__)


def scan_patient_dr_notes(root: Path) -> list[list[Path]]:
    """Group dr notes by patient folder: one sorted list of dr files per folder.

    Walks root recursively. Every folder holding at least one dr note (any category
    prefix) is one patient; nurse notes, other files and results folders are ignored.
    """
    patients: list[list[Path]] = []
    subfolders = sorted(p for p in root.rglob("*") if p.is_dir() and not in_results_folder(p, root))
    for folder in [root, *subfolders]:
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

    def __init__(
        self, router: FileRouter, classifier: DrValidityClassifier, folder_check: FolderCheck
    ) -> None:
        self._router = router
        self._classifier = classifier
        self._folder_check = folder_check

    async def run(self, patients: list[list[Path]]) -> list[DrValidityPipelineResult]:
        """Process patients concurrently, up to max_concurrent_files at a time.

        Each entry is one patient's dr notes; their text is combined under per-file
        headers. A failure yields success=False with structured ``errors`` instead of
        aborting the batch. Notes with blank text are reported; if every note is blank the
        patient fails without calling the LLM.
        """
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(dr_paths: list[Path]) -> DrValidityPipelineResult:
            folder = dr_paths[0].parent
            async with sem:
                timings = PatientTimings(patient=folder.name)
                current_timings.set(timings)
                started = time.perf_counter()
                errors: list[RunError] = []
                try:
                    extracted = await asyncio.gather(
                        *(self._timed_extract(p, timings) for p in dr_paths)
                    )
                    timings.extract_wall_s = time.perf_counter() - started
                    errors = self._empty_text_errors(dr_paths, extracted)
                    if len(errors) == len(dr_paths):
                        return self._failed(folder, dr_paths, errors)
                    metadata = DocumentMetadata(
                        file_path=dr_paths[0],
                        raw_text=concat_notes(dr_paths, [x.text for x in extracted]),
                        meta=PatientMetadata(),
                    )
                    classify_started = time.perf_counter()
                    result = await self._classifier.classify(metadata, dr_paths)
                    timings.classify_wall_s = time.perf_counter() - classify_started
                    return DrValidityPipelineResult(
                        patient_folder=folder,
                        dr_paths=dr_paths,
                        success=True,
                        result=result,
                        errors=errors,
                    )
                except ClassificationStepError as exc:
                    logger.error("Dr-validity classification failed for %s: %s", folder, exc)
                    return self._failed(folder, dr_paths, [*errors, *exc.errors])
                except Exception as exc:
                    logger.error("Dr-validity pipeline failed for %s: %s", folder, exc)
                    unexpected = RunError(
                        stage=ErrorStage.PIPELINE,
                        code=ErrorCode.UNEXPECTED,
                        message=str(exc) or type(exc).__name__,
                        folder=folder,
                    )
                    return self._failed(folder, dr_paths, [*errors, unexpected])
                finally:
                    timings.total_s = time.perf_counter() - started
                    timings.log()

        return list(await asyncio.gather(*[process(p) for p in patients]))

    @staticmethod
    def _empty_text_errors(paths: list[Path], extracted: list[ExtractedText]) -> list[RunError]:
        """One error per note whose extracted text is blank."""
        return [
            RunError(
                stage=ErrorStage.EXTRACT,
                code=ErrorCode.EMPTY_TEXT,
                message="No text extracted from note",
                folder=path.parent,
                file=path,
            )
            for path, text in zip(paths, extracted)
            if not text.text.strip()
        ]

    @staticmethod
    def _failed(
        folder: Path, dr_paths: list[Path], errors: list[RunError]
    ) -> DrValidityPipelineResult:
        """A failed patient result; ``error`` joins the messages for CLI/xlsx output."""
        return DrValidityPipelineResult(
            patient_folder=folder,
            dr_paths=dr_paths,
            success=False,
            error="; ".join(f"{e.stage}: {e.message}" for e in errors),
            errors=errors,
        )

    def folder_errors(self, root: Path) -> list[RunError]:
        """Leaf folders under root that cannot be processed (empty, or missing dr notes)."""
        return [
            error
            for folder in leaf_folders(root)
            if (error := self._folder_check.check(folder)) is not None
        ]

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
