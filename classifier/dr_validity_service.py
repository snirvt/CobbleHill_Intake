import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Protocol

from classifier.models import DrValidityPipelineResult, RunError
from classifier.output.dr_validity_report import (
    dr_validity_errors,
    dr_validity_groups,
    dr_validity_patient_dict,
    dr_validity_summary,
)

logger = logging.getLogger(__name__)


class FolderDownloader(Protocol):
    """Downloads a remote folder into local_dir; returns (local_dir, web_url)."""

    async def download(
        self, folder_path: str, local_dir: Path | None = None
    ) -> tuple[Path, str]: ...


class FolderPipeline(Protocol):
    """Runs the per-patient dr-validity check over a local folder."""

    async def run_folder(self, root: Path) -> list[DrValidityPipelineResult]: ...

    def folder_errors(self, root: Path) -> list[RunError]: ...


class DrValidityService:
    """Downloads a SharePoint folder, runs the dr-validity check, returns a JSON-ready report.

    The folder may hold many patient folders or be a single patient folder. Downloaded
    files are always deleted afterwards, even when the run fails.
    """

    def __init__(self, downloader: FolderDownloader, pipeline: FolderPipeline) -> None:
        self._downloader = downloader
        self._pipeline = pipeline

    async def run(self, sharepoint_folder: str) -> dict[str, object]:
        """Return {sharepoint_folder, web_url, summary, groups, patients, errors} for the folder.

        ``errors`` lists everything that went wrong: folders that could not be processed
        (empty, no dr notes) and per-patient failures (blank text, failed LLM steps).
        """
        folder = sharepoint_folder.strip("/")
        # Created here (not by the downloader) so a failed download is cleaned up too.
        local_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="cobblehill_api_"))
        started = time.perf_counter()
        try:
            _, web_url = await self._downloader.download(folder, local_dir)
            downloaded = time.perf_counter()
            results = await self._pipeline.run_folder(local_dir)
            folder_errors = await asyncio.to_thread(self._pipeline.folder_errors, local_dir)
            finished = time.perf_counter()
            logger.info(
                "timing request folder=%s download %.2fs | pipeline %.2fs | total %.2fs "
                "| patients=%d",
                folder,
                downloaded - started,
                finished - downloaded,
                finished - started,
                len(results),
            )
            # Names a patient whose notes sit directly in the temp download dir.
            root_name = PurePosixPath(folder).name
            return {
                "sharepoint_folder": folder,
                "web_url": web_url,
                "summary": dr_validity_summary(results, folder_errors),
                "groups": dr_validity_groups(results),
                "patients": [dr_validity_patient_dict(r, local_dir, root_name) for r in results],
                "errors": dr_validity_errors(results, folder_errors, local_dir, root_name),
            }
        finally:
            await asyncio.to_thread(shutil.rmtree, local_dir, ignore_errors=True)
            logger.debug("Deleted download dir %s", local_dir)
