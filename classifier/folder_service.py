import asyncio
import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol

from classifier.models import RunError

logger = logging.getLogger(__name__)


class FolderDownloader(Protocol):
    """Downloads a remote folder into local_dir; returns (local_dir, web_url)."""

    async def download(
        self, folder_path: str, local_dir: Path | None = None
    ) -> tuple[Path, str]: ...


class FolderPipeline[R](Protocol):
    """Runs a check over a local folder and reports folders it cannot process."""

    async def run_folder(self, root: Path) -> list[R]: ...

    def folder_errors(self, root: Path) -> list[RunError]: ...


# (results, folder_errors, local_root, root_name, web_url) -> JSON-ready report body.
type ReportBuilder[R] = Callable[[list[R], list[RunError], Path, str, str], dict[str, object]]


class FolderReportService[R]:
    """Downloads a SharePoint folder, runs a pipeline on it, returns a JSON-ready report.

    The folder may hold many patient folders or be a single patient folder. The pipeline
    decides what is checked; ``report_fn`` shapes its results. Downloaded files are always
    deleted afterwards, even when the run fails.
    """

    def __init__(
        self,
        downloader: FolderDownloader,
        pipeline: FolderPipeline[R],
        report_fn: ReportBuilder[R],
    ) -> None:
        self._downloader = downloader
        self._pipeline = pipeline
        self._report_fn = report_fn

    async def run(self, sharepoint_folder: str) -> dict[str, object]:
        """Return {sharepoint_folder, web_url, **report} for the folder.

        The report's ``errors`` lists everything that went wrong: folders that could not
        be processed and per-result failures.
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
                "| results=%d",
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
                **self._report_fn(results, folder_errors, local_dir, root_name, web_url),
            }
        finally:
            await asyncio.to_thread(shutil.rmtree, local_dir, ignore_errors=True)
            logger.debug("Deleted download dir %s", local_dir)
