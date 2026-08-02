import asyncio
import logging
from pathlib import Path

from classifier.models import DocumentMetadata, PairDocumentMetadata, PairPipelineResult
from classifier.naming import category_order, label, parse_note
from classifier.protocols import FileRouter, NoteExtractor, NurseNoteExtractor, PairClassifier
from config.settings import settings

logger = logging.getLogger(__name__)


def scan_pairs(root: Path) -> list[tuple[list[Path], list[Path]]]:
    """Recursively find dr + nurse pairs in root and its subdirectories.

    Pairing happens per category: hospital_dr_* pairs with hospital_nurse_*, peds_
    with peds_, and uncategorized dr_* with uncategorized nurse_*. One folder can
    therefore yield several pairs. When a folder holds multiple notes for the same
    category and role, ALL of them are included (sorted alphabetically); their
    extracted text is appended downstream.
    """
    pairs: list[tuple[list[Path], list[Path]]] = []
    _collect_pairs(root, pairs)
    return pairs


def _collect_pairs(folder: Path, pairs: list[tuple[list[Path], list[Path]]]) -> None:
    by_category: dict[str | None, dict[str, list[Path]]] = {}
    for f in sorted(folder.iterdir()):
        parsed = parse_note(f)
        if parsed is None:
            continue
        by_category.setdefault(parsed.category, {}).setdefault(parsed.role, []).append(f)

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
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            _collect_pairs(child, pairs)


def _concat_notes(paths: list[Path], texts: list[str]) -> str:
    """Append multiple note texts, each prefixed with a filename header marker."""
    return "\n\n".join(f"--- {p.name} ---\n{t}" for p, t in zip(paths, texts))


class PairPipeline:
    """Orchestrates: route → extract both docs → parse both → classify pair."""

    def __init__(
        self,
        router: FileRouter,
        dr_meta_extractor: NoteExtractor,
        nurse_meta_extractor: NurseNoteExtractor,
        pair_classifier: PairClassifier,
    ) -> None:
        self._router = router
        self._dr_meta = dr_meta_extractor
        self._nurse_meta = nurse_meta_extractor
        self._pair_clf = pair_classifier

    async def run_pairs(
        self, pairs: list[tuple[list[Path], list[Path]]]
    ) -> list[PairPipelineResult]:
        """Process all pairs concurrently up to max_concurrent_files at a time.

        Each pair is (dr_paths, nurse_paths); multiple files per side are extracted
        and their text appended before classification. The first file of each side is
        the representative path used for folder/patient resolution.
        """
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(dr_paths: list[Path], nurse_paths: list[Path]) -> PairPipelineResult:
            dr_path, nurse_path = dr_paths[0], nurse_paths[0]
            async with sem:
                try:
                    dr_extracted = await asyncio.gather(
                        *(self._router.route(p).extract(p) for p in dr_paths)
                    )
                    nurse_extracted = await asyncio.gather(
                        *(self._router.route(p).extract(p) for p in nurse_paths)
                    )
                    dr_full = _concat_notes(dr_paths, [x.text for x in dr_extracted])
                    nurse_full = _concat_notes(nurse_paths, [x.text for x in nurse_extracted])
                    dr_fields = self._dr_meta.extract(dr_full)
                    nurse_fields = self._nurse_meta.extract(nurse_full)
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
                    pair = PairDocumentMetadata(
                        dr_file_path=dr_path,
                        nurse_file_path=nurse_path,
                        dr=dr_meta,
                        nurse=nurse_fields,
                        nurse_raw_text=nurse_full,
                    )
                    result = await self._pair_clf.classify_pair(pair)
                    return PairPipelineResult(
                        dr_file_path=dr_path,
                        nurse_file_path=nurse_path,
                        dr_paths=dr_paths,
                        nurse_paths=nurse_paths,
                        success=True,
                        result=result,
                    )
                except Exception as exc:
                    logger.error("Pair pipeline failed for %s + %s: %s", dr_path, nurse_path, exc)
                    return PairPipelineResult(
                        dr_file_path=dr_path,
                        nurse_file_path=nurse_path,
                        dr_paths=dr_paths,
                        nurse_paths=nurse_paths,
                        success=False,
                        error=str(exc),
                    )

        return list(await asyncio.gather(*[process(dr, nurse) for dr, nurse in pairs]))

    async def run_folder(self, root: Path) -> list[PairPipelineResult]:
        """Scan root for dr/nurse pairs and run the pair pipeline."""
        pairs = scan_pairs(root)
        if not pairs:
            logger.warning("No dr/nurse note pairs found in %s", root)
        return await self.run_pairs(pairs)
