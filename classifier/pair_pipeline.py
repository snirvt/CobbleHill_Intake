import asyncio
import logging
import time
from pathlib import Path

from classifier.folder_checks import leaf_folders
from classifier.models import (
    ClassificationStepError,
    DocumentMetadata,
    ErrorCode,
    ErrorStage,
    ExtractedText,
    PairDocumentMetadata,
    PairPipelineResult,
    RunError,
)
from classifier.naming import category_order, concat_notes, in_results_folder, label, parse_note
from classifier.protocols import (
    FileRouter,
    FolderCheck,
    NoteExtractor,
    NurseNoteExtractor,
    PairClassifier,
)
from classifier.timing import PatientTimings, current_timings
from config.settings import settings

logger = logging.getLogger(__name__)


def scan_pairs(root: Path) -> list[tuple[list[Path], list[Path]]]:
    """Recursively find dr + nurse pairs in root and its subdirectories.

    Pairing happens per category: hospital_dr_* pairs with hospital_nurse_*, peds_
    with peds_, and uncategorized dr_* with uncategorized nurse_*. One folder can
    therefore yield several pairs. When a folder holds multiple notes for the same
    category and role, ALL of them are included (sorted alphabetically); their
    extracted text is appended downstream. Results folders are ignored.
    """
    pairs: list[tuple[list[Path], list[Path]]] = []
    _collect_pairs(root, root, pairs)
    return pairs


def _collect_pairs(
    folder: Path, root: Path, pairs: list[tuple[list[Path], list[Path]]]
) -> None:
    pairs.extend(pairs_in_folder(folder))
    for child in sorted(folder.iterdir()):
        if child.is_dir() and not in_results_folder(child, root):
            _collect_pairs(child, root, pairs)


def pairs_in_folder(folder: Path) -> list[tuple[list[Path], list[Path]]]:
    """Same-category (dr_files, nurse_files) pairs directly in folder (not recursive).

    Categories come in ``category_order``; every note of a category and role is
    included, sorted alphabetically.
    """
    by_category: dict[str | None, dict[str, list[Path]]] = {}
    for f in sorted(folder.iterdir()):
        parsed = parse_note(f)
        if parsed is None:
            continue
        by_category.setdefault(parsed.category, {}).setdefault(parsed.role, []).append(f)

    pairs: list[tuple[list[Path], list[Path]]] = []
    for category in category_order():
        roles = by_category.get(category, {})
        dr_files, nurse_files = roles.get("dr", []), roles.get("nurse", [])
        if not (dr_files and nurse_files):
            continue
        pairs.append((dr_files, nurse_files))
        if len(dr_files) > 1 or len(nurse_files) > 1:
            logger.info(
                "Folder %s has multiple %s notes — appending dr=%s, nurse=%s",
                folder,
                label(category),
                [f.name for f in dr_files],
                [f.name for f in nurse_files],
            )
    return pairs


def build_pair_metadata(
    dr_meta_extractor: NoteExtractor,
    nurse_meta_extractor: NurseNoteExtractor,
    dr_paths: list[Path],
    nurse_paths: list[Path],
    dr_texts: list[str],
    nurse_texts: list[str],
) -> PairDocumentMetadata:
    """Combine each side's text and parse its fields into the pair classifier's input."""
    dr_path, nurse_path = dr_paths[0], nurse_paths[0]
    dr_full = concat_notes(dr_paths, dr_texts)
    nurse_full = concat_notes(nurse_paths, nurse_texts)
    dr_fields = dr_meta_extractor.extract(dr_full)
    nurse_fields = nurse_meta_extractor.extract(nurse_full)
    dr_meta = DocumentMetadata(
        file_path=dr_path,
        raw_text=dr_full,
        meta=dr_fields.meta,
        hpi=dr_fields.hpi,
        examination=dr_fields.examination,
        complaints=dr_fields.complaints,
        medical_history=dr_fields.medical_history,
        surgical_history=dr_fields.surgical_history,
        hospitalization=dr_fields.hospitalization,
        assessment=dr_fields.assessment,
        ros=dr_fields.ros,
        medications=dr_fields.medications,
        plan=dr_fields.plan,
        procedure_codes=dr_fields.procedure_codes,
        preventive_medicine=dr_fields.preventive_medicine,
    )
    parsed = parse_note(dr_path)
    return PairDocumentMetadata(
        dr_file_path=dr_path,
        nurse_file_path=nurse_path,
        dr=dr_meta,
        nurse=nurse_fields,
        nurse_raw_text=nurse_full,
        category=parsed.category if parsed else None,
    )


class PairPipeline:
    """Orchestrates: route → extract both docs → parse both → classify pair.

    A failure yields success=False with structured ``errors`` instead of aborting the
    batch; ``folder_errors`` reports leaf folders that hold no usable pair.
    """

    def __init__(
        self,
        router: FileRouter,
        dr_meta_extractor: NoteExtractor,
        nurse_meta_extractor: NurseNoteExtractor,
        pair_classifier: PairClassifier,
        folder_check: FolderCheck,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """``semaphore`` shares one pair limit across runs; else each run gets its own."""
        self._router = router
        self._dr_meta = dr_meta_extractor
        self._nurse_meta = nurse_meta_extractor
        self._pair_clf = pair_classifier
        self._folder_check = folder_check
        self._semaphore = semaphore

    async def run_pairs(
        self, pairs: list[tuple[list[Path], list[Path]]]
    ) -> list[PairPipelineResult]:
        """Process all pairs concurrently up to max_concurrent_files at a time (shared when
        a semaphore was injected).

        Each pair is (dr_paths, nurse_paths); multiple files per side are extracted
        and their text appended before classification. The first file of each side is
        the representative path used for folder/patient resolution. Notes with blank
        text are reported; if every note of one side is blank the pair fails without
        calling the LLM.
        """
        sem = self._semaphore or asyncio.Semaphore(settings.max_concurrent_files)

        async def process(dr_paths: list[Path], nurse_paths: list[Path]) -> PairPipelineResult:
            dr_path, nurse_path = dr_paths[0], nurse_paths[0]
            folder = dr_path.parent
            async with sem:
                timings = PatientTimings(patient=folder.name)
                current_timings.set(timings)
                started = time.perf_counter()
                errors: list[RunError] = []
                try:
                    dr_extracted, nurse_extracted = await asyncio.gather(
                        self._extract_all(dr_paths), self._extract_all(nurse_paths)
                    )
                    timings.extract_wall_s = time.perf_counter() - started
                    dr_errors = self._empty_text_errors(dr_paths, dr_extracted)
                    nurse_errors = self._empty_text_errors(nurse_paths, nurse_extracted)
                    errors = [*dr_errors, *nurse_errors]
                    if len(dr_errors) == len(dr_paths) or len(nurse_errors) == len(nurse_paths):
                        return self._failed(dr_paths, nurse_paths, errors)
                    pair = self._build_pair(dr_paths, nurse_paths, dr_extracted, nurse_extracted)
                    classify_started = time.perf_counter()
                    result = await self._pair_clf.classify_pair(pair)
                    timings.classify_wall_s = time.perf_counter() - classify_started
                    return PairPipelineResult(
                        dr_file_path=dr_path,
                        nurse_file_path=nurse_path,
                        dr_paths=dr_paths,
                        nurse_paths=nurse_paths,
                        success=True,
                        result=result,
                        errors=errors,
                    )
                except ClassificationStepError as exc:
                    logger.error(
                        "Pair classification failed for %s + %s: %s", dr_path, nurse_path, exc
                    )
                    return self._failed(dr_paths, nurse_paths, [*errors, *exc.errors])
                except Exception as exc:
                    logger.error("Pair pipeline failed for %s + %s: %s", dr_path, nurse_path, exc)
                    unexpected = RunError(
                        stage=ErrorStage.PIPELINE,
                        code=ErrorCode.UNEXPECTED,
                        message=str(exc) or type(exc).__name__,
                        folder=folder,
                    )
                    return self._failed(dr_paths, nurse_paths, [*errors, unexpected])
                finally:
                    timings.total_s = time.perf_counter() - started
                    timings.log()

        return list(await asyncio.gather(*[process(dr, nurse) for dr, nurse in pairs]))

    async def _extract_all(self, paths: list[Path]) -> list[ExtractedText]:
        """Extract every file of one pair side concurrently."""
        return list(await asyncio.gather(*(self._router.route(p).extract(p) for p in paths)))

    def _build_pair(
        self,
        dr_paths: list[Path],
        nurse_paths: list[Path],
        dr_extracted: list[ExtractedText],
        nurse_extracted: list[ExtractedText],
    ) -> PairDocumentMetadata:
        """Combine each side's text and parse its fields into the classifier's input."""
        return build_pair_metadata(
            self._dr_meta,
            self._nurse_meta,
            dr_paths,
            nurse_paths,
            [x.text for x in dr_extracted],
            [x.text for x in nurse_extracted],
        )

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
        dr_paths: list[Path], nurse_paths: list[Path], errors: list[RunError]
    ) -> PairPipelineResult:
        """A failed pair result; ``error`` joins the messages for CLI/xlsx output."""
        return PairPipelineResult(
            dr_file_path=dr_paths[0],
            nurse_file_path=nurse_paths[0],
            dr_paths=dr_paths,
            nurse_paths=nurse_paths,
            success=False,
            error="; ".join(f"{e.stage}: {e.message}" for e in errors),
            errors=errors,
        )

    def folder_errors(self, root: Path) -> list[RunError]:
        """Leaf folders under root that cannot be processed (empty, or no dr+nurse pair)."""
        return [
            error
            for folder in leaf_folders(root)
            if (error := self._folder_check.check(folder)) is not None
        ]

    async def run_folder(self, root: Path) -> list[PairPipelineResult]:
        """Scan root for dr/nurse pairs and run the pair pipeline."""
        pairs = scan_pairs(root)
        if not pairs:
            logger.warning("No dr/nurse note pairs found in %s", root)
        return await self.run_pairs(pairs)
