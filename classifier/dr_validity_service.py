import asyncio
import logging
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Protocol

from classifier.models import DrValidityPipelineResult
from classifier.output.dr_validity_report import (
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


class DrValidityService:
    """Downloads a SharePoint folder, runs the dr-validity check, returns a JSON-ready report.

    The folder may hold many patient folders or be a single patient folder. Downloaded
    files are always deleted afterwards, even when the run fails.
    """

    def __init__(self, downloader: FolderDownloader, pipeline: FolderPipeline) -> None:
        self._downloader = downloader
        self._pipeline = pipeline

    async def run(self, sharepoint_folder: str) -> dict[str, object]:
        """Return {sharepoint_folder, web_url, summary, groups, patients} for the folder."""
        folder = sharepoint_folder.strip("/")
        # Created here (not by the downloader) so a failed download is cleaned up too.
        local_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="cobblehill_api_"))
        try:
            _, web_url = await self._downloader.download(folder, local_dir)
            results = await self._pipeline.run_folder(local_dir)
            # Names a patient whose notes sit directly in the temp download dir.
            root_name = PurePosixPath(folder).name
            return {
                "sharepoint_folder": folder,
                "web_url": web_url,
                "summary": dr_validity_summary(results),
                "groups": dr_validity_groups(results),
                "patients": [dr_validity_patient_dict(r, local_dir, root_name) for r in results],
            }
        finally:
            await asyncio.to_thread(shutil.rmtree, local_dir, ignore_errors=True)
            logger.debug("Deleted download dir %s", local_dir)
