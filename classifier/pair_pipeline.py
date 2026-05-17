import asyncio
import logging
from pathlib import Path

from classifier.models import DocumentMetadata, PairDocumentMetadata, PairPipelineResult
from classifier.protocols import FileRouter, MetadataExtractor, NurseMetadataExtractor, PairClassifier
from config.settings import settings

logger = logging.getLogger(__name__)

_DR_PREFIX = "dr_"
_NURSE_PREFIX = "nurse_"


def scan_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Recursively find dr_* + nurse_* pairs in root and its subdirectories.

    Each folder (including root) that contains at least one dr_* and one nurse_*
    supported file yields one pair (first alphabetically of each prefix).
    """
    pairs: list[tuple[Path, Path]] = []
    _collect_pairs(root, pairs)
    return pairs


def _collect_pairs(folder: Path, pairs: list[tuple[Path, Path]]) -> None:
    exts = set(settings.supported_extensions.keys())
    dr_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.name.lower().startswith(_DR_PREFIX) and f.suffix.lower() in exts
    )
    nurse_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.name.lower().startswith(_NURSE_PREFIX) and f.suffix.lower() in exts
    )
    if dr_files and nurse_files:
        pairs.append((dr_files[0], nurse_files[0]))
        if len(dr_files) > 1 or len(nurse_files) > 1:
            logger.warning(
                "Folder %s has multiple dr_* or nurse_* files — using first of each: %s, %s",
                folder,
                dr_files[0].name,
                nurse_files[0].name,
            )
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            _collect_pairs(child, pairs)


class PairPipeline:
    """Orchestrates: route → extract both docs → parse both → classify pair."""

    def __init__(
        self,
        router: FileRouter,
        dr_meta_extractor: MetadataExtractor,
        nurse_meta_extractor: NurseMetadataExtractor,
        pair_classifier: PairClassifier,
    ) -> None:
        self._router = router
        self._dr_meta = dr_meta_extractor
        self._nurse_meta = nurse_meta_extractor
        self._pair_clf = pair_classifier

    async def run_pairs(self, pairs: list[tuple[Path, Path]]) -> list[PairPipelineResult]:
        """Process all pairs concurrently up to max_concurrent_files at a time."""
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(dr_path: Path, nurse_path: Path) -> PairPipelineResult:
            async with sem:
                try:
                    dr_ext = self._router.route(dr_path)
                    nurse_ext = self._router.route(nurse_path)
                    dr_text, nurse_text = await asyncio.gather(
                        dr_ext.extract(dr_path),
                        nurse_ext.extract(nurse_path),
                    )
                    dr_fields = self._dr_meta.extract(dr_text.text)
                    nurse_fields = self._nurse_meta.extract(nurse_text.text)
                    dr_meta = DocumentMetadata(
                        file_path=dr_path,
                        raw_text=dr_text.text,
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
                    )
                    result = await self._pair_clf.classify_pair(pair)
                    return PairPipelineResult(
                        dr_file_path=dr_path,
                        nurse_file_path=nurse_path,
                        success=True,
                        result=result,
                    )
                except Exception as exc:
                    logger.error("Pair pipeline failed for %s + %s: %s", dr_path, nurse_path, exc)
                    return PairPipelineResult(
                        dr_file_path=dr_path,
                        nurse_file_path=nurse_path,
                        success=False,
                        error=str(exc),
                    )

        return list(await asyncio.gather(*[process(dr, nurse) for dr, nurse in pairs]))

    async def run_folder(self, root: Path) -> list[PairPipelineResult]:
        """Scan root for dr_*/nurse_* pairs and run the pair pipeline."""
        pairs = scan_pairs(root)
        if not pairs:
            logger.warning("No dr_*/nurse_* pairs found in %s", root)
        return await self.run_pairs(pairs)
