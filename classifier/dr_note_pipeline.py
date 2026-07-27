import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Generic, TypeVar

from classifier.models import DocumentMetadata, PatientMetadata
from classifier.pair_pipeline import _DR_PREFIX
from classifier.protocols import FileRouter
from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def scan_dr_notes(folder: Path) -> list[Path]:
    """Return dr_* notes with supported extensions in folder, recursively, sorted.

    Nurse notes (and any non-dr_* files) are excluded.
    """
    return [
        f
        for f in sorted(folder.rglob("*"))
        if f.is_file()
        and f.name.lower().startswith(_DR_PREFIX)
        and f.suffix.lower() in settings.supported_extensions
    ]


class DrNotePipeline(Generic[T]):
    """Generic dr-note pipeline: route → extract text → apply an extractor fn.

    Processes dr_* notes only (nurse notes ignored). ``extract_fn`` receives the
    document metadata (raw text populated) and returns a result of type T. On a
    routing/IO failure, ``error_fn`` produces a fallback T so a bad file never
    aborts the batch.
    """

    def __init__(
        self,
        router: FileRouter,
        extract_fn: Callable[[DocumentMetadata], Awaitable[T]],
        error_fn: Callable[[Path, Exception], T],
    ) -> None:
        self._router = router
        self._extract_fn = extract_fn
        self._error_fn = error_fn

    async def run(self, files: list[Path]) -> list[T]:
        """Process files concurrently, up to max_concurrent_files at a time."""
        sem = asyncio.Semaphore(settings.max_concurrent_files)

        async def process(file_path: Path) -> T:
            async with sem:
                try:
                    content_extractor = self._router.route(file_path)
                    text_result = await content_extractor.extract(file_path)
                    doc_meta = DocumentMetadata(
                        file_path=file_path,
                        raw_text=text_result.text,
                        meta=PatientMetadata(),
                    )
                    return await self._extract_fn(doc_meta)
                except Exception as exc:
                    logger.error("Dr-note pipeline failed for %s: %s", file_path, exc)
                    return self._error_fn(file_path, exc)

        return list(await asyncio.gather(*[process(f) for f in files]))

    async def run_folder(self, folder: Path) -> list[T]:
        """Scan folder recursively for dr_* notes and process them."""
        files = scan_dr_notes(folder)
        if not files:
            logger.warning("No dr_* notes found in %s", folder)
        return await self.run(files)
