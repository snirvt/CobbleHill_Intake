"""SharePoint Online folder downloader/uploader via Microsoft Graph API."""

import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import httpx

try:
    from msal import ConfidentialClientApplication
except ImportError:
    ConfidentialClientApplication = None  # type: ignore[assignment,misc]

from config.settings import settings

logger = logging.getLogger(__name__)

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"


def authenticate_to_graph(client_id: str, client_secret: str, tenant_id: str) -> str:
    """Acquire a Graph API access token via MSAL client credentials flow."""
    if ConfidentialClientApplication is None:
        raise RuntimeError("msal package not installed. Run: uv add msal")
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = ConfidentialClientApplication(
        client_id, authority=authority, client_credential=client_secret
    )
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in result:
        raise RuntimeError(
            f"MSAL authentication failed: {result.get('error_description', result.get('error'))}"
        )
    return result["access_token"]  # type: ignore[return-value]


async def _download_file(
    drive_id: str,
    item_id: str,
    dest: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> None:
    """Download a single file, gated by *semaphore*."""
    url = f"{GRAPH_ROOT}/drives/{drive_id}/items/{item_id}/content"
    async with semaphore:
        response = await client.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        dest.write_bytes(response.content)
    logger.info("Downloaded: %s", dest.name)


async def download_folder(
    drive_id: str,
    folder_path: str,
    local_dir: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> None:
    """Recursively download all supported files from a SharePoint drive folder."""
    local_dir.mkdir(parents=True, exist_ok=True)

    url = f"{GRAPH_ROOT}/drives/{drive_id}/root:/{folder_path}:/children"
    response = await client.get(url, timeout=30)
    response.raise_for_status()

    supported = set(settings.supported_extensions.keys())

    tasks = []
    for item in response.json()["value"]:
        name: str = item["name"]

        if "folder" in item:
            logger.info("Entering folder: %s/%s", folder_path, name)
            tasks.append(
                download_folder(
                    drive_id,
                    f"{folder_path}/{name}",
                    local_dir / name,
                    client,
                    semaphore,
                )
            )
        else:
            if Path(name).suffix.lower() not in supported:
                logger.debug("Skipping unsupported file: %s", name)
                continue
            tasks.append(
                _download_file(drive_id, item["id"], local_dir / name, client, semaphore)
            )

    await asyncio.gather(*tasks)


async def upload_file(
    drive_id: str,
    folder_path: str,
    local_file: Path,
    client: httpx.AsyncClient,
) -> str:
    """Upload *local_file* to *folder_path* on SharePoint, returning the remote filename.

    A timestamp suffix is appended to avoid overwriting existing files.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    remote_name = f"{local_file.stem}_{timestamp}{local_file.suffix}"
    url = f"{GRAPH_ROOT}/drives/{drive_id}/root:/{folder_path}/{remote_name}:/content"
    response = await client.put(url, content=local_file.read_bytes(), timeout=60)
    response.raise_for_status()
    logger.info("Uploaded: %s → %s/%s", local_file.name, folder_path, remote_name)
    return remote_name


class SharePointFileUploader:
    """Uploads files to a SharePoint Online drive folder."""

    def __init__(self, client_id: str, client_secret: str, tenant_id: str, drive_id: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._drive_id = drive_id

    async def upload(self, local_file: Path, folder_path: str) -> str:
        """Upload *local_file* to *folder_path*, returning the remote filename."""
        access_token = authenticate_to_graph(self._client_id, self._client_secret, self._tenant_id)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/octet-stream",
        }
        async with httpx.AsyncClient(headers=headers) as client:
            return await upload_file(self._drive_id, folder_path, local_file, client)


class SharePointFolderDownloader:
    """Downloads files from a SharePoint Online drive folder."""

    def __init__(self, client_id: str, client_secret: str, tenant_id: str, drive_id: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._drive_id = drive_id

    async def download(self, folder_path: str, local_dir: Path | None = None) -> Path:
        """Download all supported files from *folder_path* into *local_dir*.

        *folder_path* is the drive-relative path, e.g.
        ``"Patient Encounters/Medical Notes/Non-Admits"``.

        If *local_dir* is None a temporary directory is created; the caller is
        responsible for deleting it when done.
        """
        if local_dir is None:
            local_dir = Path(tempfile.mkdtemp(prefix="cobblehill_sp_"))

        access_token = authenticate_to_graph(self._client_id, self._client_secret, self._tenant_id)
        headers = {"Authorization": f"Bearer {access_token}"}
        semaphore = asyncio.Semaphore(settings.sharepoint_download_concurrency)

        logger.info("Downloading SharePoint folder '%s' → %s", folder_path, local_dir)
        async with httpx.AsyncClient(headers=headers) as client:
            await download_folder(self._drive_id, folder_path, local_dir, client, semaphore)
        return local_dir
