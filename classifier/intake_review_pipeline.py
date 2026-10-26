import asyncio
import logging
import time
from pathlib import Path

from classifier.classifiers.intake_verdict import (
    explain,
    final_verdict,
    phase_1_verdict,
    phase_2_verdict,
)
from classifier.classifiers.step_errors import raise_on_step_errors
from classifier.folder_checks import leaf_folders
from classifier.models import (
    ClassificationStepError,
    DocumentMetadata,
    ErrorCode,
    ErrorStage,
    ExtractedText,
    IntakeReviewPipelineResult,
    IntakeReviewResult,
    PairMatch,
    PatientMetadata,
    PhaseVerdict,
    RunError,
)
from classifier.naming import concat_notes, parse_note
from classifier.pair_pipeline import build_pair_metadata, pairs_in_folder
from classifier.patient_pipeline import scan_patient_dr_notes
from classifier.protocols import (
    ContentExtractor,
    DiagnosisExtractor,
    DrValidityClassifier,
    FileRouter,
    FolderCheck,
    NoteExtractor,
    NurseNoteExtractor,
    PairClassifier,
)
from classifier.timing import ExtractTiming, PatientTimings, current_timings, step
from config.settings import settings

logger = logging.getLogger(__name__)


class IntakeReviewPipeline:
    """Two-phase intake review per patient folder.

    Phase I runs the dr-validity classifier on the patient's combined dr notes. Only when
    it passes are the folder's same-category dr + nurse pairs compared (Phase II): the
    pair classifier gives the clinical match and the diagnosis extractor reads the nurse
    note's diagnoses for the explanation. Nurse notes are not extracted when Phase I fails.
    A patient with no same-category nurse note comes out NO_NURSE_NOTE; a folder with no
    dr note is listed with no result.
    """

    def __init__(
        self,
        router: FileRouter,
        validity_classifier: DrValidityClassifier,
        pair_classifier: PairClassifier,
        nurse_diagnosis_extractor: DiagnosisExtractor,
        dr_meta_extractor: NoteExtractor,
        nurse_meta_extractor: NurseNoteExtractor,
        folder_check: FolderCheck,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """``semaphore`` shares one patient limit across runs; else each run gets its own."""
        self._router = router
        self._validity = validity_classifier
        self._pair_clf = pair_classifier
        self._nurse_diagnoses = nurse_diagnosis_extractor
        self._dr_meta = dr_meta_extractor
        self._nurse_meta = nurse_meta_extractor
        self._folder_check = folder_check
        self._semaphore = semaphore

    async def run(self, patients: list[list[Path]]) -> list[IntakeReviewPipelineResult]:
        """Review patients concurrently, up to max_concurrent_files at a time.

        Each entry is one patient's dr notes. A failure yields success=False with
        structured ``errors`` instead of aborting the batch.
        """
        sem = self._semaphore or asyncio.Semaphore(settings.max_concurrent_files)

        async def process(dr_paths: list[Path]) -> IntakeReviewPipelineResult:
            folder = dr_paths[0].parent
            nurse_paths = sorted(f for f in folder.iterdir() if parse_note(f, role="nurse"))
            async with sem:
                timings = PatientTimings(patient=folder.name)
                current_timings.set(timings)
                started = time.perf_counter()
                errors: list[RunError] = []
                try:
                    return await self._review(folder, dr_paths, nurse_paths, timings, errors)
                except ClassificationStepError as exc:
                    logger.error("Intake review failed for %s: %s", folder, exc)
                    return self._failed(folder, dr_paths, nurse_paths, [*errors, *exc.errors])
                except Exception as exc:
                    logger.error("Intake review pipeline failed for %s: %s", folder, exc)
                    unexpected = RunError(
                        stage=ErrorStage.PIPELINE,
                        code=ErrorCode.UNEXPECTED,
                        message=str(exc) or type(exc).__name__,
                        folder=folder,
                    )
                    return self._failed(folder, dr_paths, nurse_paths, [*errors, unexpected])
                finally:
                    timings.total_s = time.perf_counter() - started
                    timings.log()

        return list(await asyncio.gather(*[process(p) for p in patients]))

    async def _review(
        self,
        folder: Path,
        dr_paths: list[Path],
        nurse_paths: list[Path],
        timings: PatientTimings,
        errors: list[RunError],
    ) -> IntakeReviewPipelineResult:
        """Phase I, then Phase II when it passed. Appends non-fatal issues to ``errors``."""
        pair_paths = pairs_in_folder(folder)
        started = time.perf_counter()
        texts = await self._extract_texts(dr_paths, timings)
        timings.extract_wall_s = time.perf_counter() - started
        errors.extend(self._empty_text_errors(dr_paths, texts))
        if all(not texts[p].strip() for p in dr_paths):
            return self._failed(folder, dr_paths, nurse_paths, errors)

        metadata = DocumentMetadata(
            file_path=dr_paths[0],
            raw_text=concat_notes(dr_paths, [texts[p] for p in dr_paths]),
            meta=PatientMetadata(),
        )
        classify_started = time.perf_counter()
        phase_1 = await self._validity.classify(metadata, dr_paths)
        verdict_1, reasons, warnings = phase_1_verdict(phase_1)

        pairs: list[PairMatch] | None = []
        if verdict_1 is PhaseVerdict.PASS:
            pairs = await self._phase_2(pair_paths, texts, timings, errors)
        timings.classify_wall_s = time.perf_counter() - classify_started
        if pairs is None:
            return self._failed(folder, dr_paths, nurse_paths, errors)

        has_pair = bool(pair_paths)
        verdict_2: PhaseVerdict | None = None
        if has_pair:
            verdict_2 = (
                phase_2_verdict(pairs) if verdict_1 is PhaseVerdict.PASS else PhaseVerdict.NOT_RUN
            )
        result = IntakeReviewResult(
            patient_folder=folder,
            phase_1=phase_1,
            verdict_phase_1=verdict_1,
            phase_1_reasons=reasons,
            pairs=pairs,
            verdict_phase_2=verdict_2,
            verdict=final_verdict(verdict_1, verdict_2, has_pair),
            explanation=explain(verdict_1, reasons, phase_1.diagnoses, pairs, has_pair),
            warnings=warnings,
        )
        logger.debug(
            "Intake review %s for %s (phase 1=%s, phase 2=%s)",
            result.verdict,
            folder,
            verdict_1,
            verdict_2,
        )
        return IntakeReviewPipelineResult(
            patient_folder=folder,
            dr_paths=dr_paths,
            nurse_paths=nurse_paths,
            success=True,
            result=result,
            errors=errors,
        )

    async def _phase_2(
        self,
        pair_paths: list[tuple[list[Path], list[Path]]],
        texts: dict[Path, str],
        timings: PatientTimings,
        errors: list[RunError],
    ) -> list[PairMatch] | None:
        """Compare every same-category (dr, nurse) pair; [] when there is none.

        Returns None (patient fails) when every nurse note of some pair has no text, since
        that pair cannot be judged.
        """
        nurse_paths = sorted({p for _, nurse in pair_paths for p in nurse})
        texts.update(await self._extract_texts(nurse_paths, timings))
        nurse_errors = self._empty_text_errors(nurse_paths, texts)
        errors.extend(nurse_errors)
        if any(all(not texts[p].strip() for p in nurse) for _, nurse in pair_paths):
            return None
        return list(
            await asyncio.gather(*(self._match_pair(dr, nurse, texts) for dr, nurse in pair_paths))
        )

    async def _match_pair(
        self, dr_paths: list[Path], nurse_paths: list[Path], texts: dict[Path, str]
    ) -> PairMatch:
        """Clinical match plus the nurse note's diagnoses for one pair."""
        pair = build_pair_metadata(
            self._dr_meta,
            self._nurse_meta,
            dr_paths,
            nurse_paths,
            [texts[p] for p in dr_paths],
            [texts[p] for p in nurse_paths],
        )
        nurse_doc = DocumentMetadata(
            file_path=pair.nurse_file_path, raw_text=pair.nurse_raw_text, meta=PatientMetadata()
        )
        match, nurse_diagnoses = await asyncio.gather(
            step("clinical_match", self._pair_clf.classify_pair(pair)),
            step("nurse_diagnosis", self._nurse_diagnoses.extract_diagnoses(nurse_doc)),
        )
        raise_on_step_errors(
            pair.dr_file_path.parent, {ErrorStage.DIAGNOSIS: nurse_diagnoses.error}
        )
        return PairMatch(
            category=pair.category,
            dr_paths=dr_paths,
            nurse_paths=nurse_paths,
            clinical_verdict=match.clinical_verdict,
            clinical_reasoning=match.clinical_reasoning,
            nurse_diagnoses=nurse_diagnoses.diagnoses,
        )

    async def _extract_texts(self, paths: list[Path], timings: PatientTimings) -> dict[Path, str]:
        """Extract every file concurrently; path -> text."""
        extracted = await asyncio.gather(*(self._timed_extract(p, timings) for p in paths))
        return {p: x.text for p, x in zip(paths, extracted)}

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

    @staticmethod
    def _empty_text_errors(paths: list[Path], texts: dict[Path, str]) -> list[RunError]:
        """One error per note whose extracted text is blank."""
        return [
            RunError(
                stage=ErrorStage.EXTRACT,
                code=ErrorCode.EMPTY_TEXT,
                message="No text extracted from note",
                folder=path.parent,
                file=path,
            )
            for path in paths
            if not texts[path].strip()
        ]

    @staticmethod
    def _failed(
        folder: Path, dr_paths: list[Path], nurse_paths: list[Path], errors: list[RunError]
    ) -> IntakeReviewPipelineResult:
        """A failed patient result; ``error`` joins the messages for CLI/xlsx output."""
        return IntakeReviewPipelineResult(
            patient_folder=folder,
            dr_paths=dr_paths,
            nurse_paths=nurse_paths,
            success=False,
            error="; ".join(f"{e.stage}: {e.message}" for e in errors),
            errors=errors,
        )

    def folder_errors(self, root: Path) -> list[RunError]:
        """Leaf folders under root that cannot be processed (empty, or no dr notes).

        A missing nurse note is not a folder error: that patient is reviewed and comes out
        NO_NURSE_NOTE.
        """
        return [
            error
            for folder in leaf_folders(root)
            if (error := self._folder_check.check(folder)) is not None
        ]

    async def run_folder(self, root: Path) -> list[IntakeReviewPipelineResult]:
        """Review every patient folder with dr notes under root.

        Leaf folders with no dr note (nurse notes only, or empty) are listed after them
        with no result, so every patient folder appears in the report.
        """
        patients = scan_patient_dr_notes(root)
        if not patients:
            logger.warning("No dr notes found in %s", root)
        reviewed = await self.run(patients)
        return [*reviewed, *self._no_dr_note_results(root)]

    @staticmethod
    def _no_dr_note_results(root: Path) -> list[IntakeReviewPipelineResult]:
        """A result-less entry for each leaf folder under root that holds no dr note."""
        results = []
        for folder in leaf_folders(root):
            files = sorted(folder.iterdir())
            if any(parse_note(f, role="dr") for f in files):
                continue
            results.append(
                IntakeReviewPipelineResult(
                    patient_folder=folder,
                    nurse_paths=[f for f in files if parse_note(f, role="nurse")],
                    success=True,
                )
            )
        return results
