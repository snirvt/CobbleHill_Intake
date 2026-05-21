"""SharePoint Online folder downloader."""

import logging
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from config.settings import settings

try:
    from office365.sharepoint.client_context import ClientContext
except ImportError:  # package not installed — only needed at runtime, not in tests
    ClientContext = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def is_sharepoint_url(value: str) -> bool:
    """Return True if value is an http(s) URL (treat as SharePoint)."""
    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _parse_sharepoint_url(url: str) -> tuple[str, str]:
    """Return (site_url, folder_server_relative_path) from a SharePoint AllItems URL.

    Expects the standard SharePoint web UI URL where the folder path is in the ``id``
    query parameter, e.g.:
        https://<tenant>.sharepoint.com/sites/<site>/…/AllItems.aspx?id=%2Fsites%2F…
    """
    parsed = urlparse(url)

    # Folder path comes from the ?id= query param
    qs = parse_qs(parsed.query)
    id_values = qs.get("id", [])
    if not id_values:
        raise ValueError(
            f"Cannot determine SharePoint folder: no 'id' query parameter in URL: {url}"
        )
    folder_rel_path = unquote(id_values[0])  # e.g. /sites/Roga/Shared Documents/...

    # Site URL: scheme + netloc + /sites/<site-name>
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) >= 2 and path_parts[0] == "sites":
        site_url = f"{parsed.scheme}://{parsed.netloc}/sites/{path_parts[1]}"
    else:
        raise ValueError(
            f"Cannot extract site URL from SharePoint path: {parsed.path!r}"
        )

    return site_url, folder_rel_path


class SharePointFolderDownloader:
    """Downloads all supported files from a SharePoint Online folder to a temp directory."""

    def __init__(self, client_id: str) -> None:
        self._client_id = client_id

    def download_to_temp(self, folder_url: str) -> Path:
        """Download supported files from *folder_url* and return path to a new temp directory.

        The caller is responsible for deleting the directory when done.
        """
        if ClientContext is None:
            raise RuntimeError(
                "office365 package not installed. Run: uv add Office365-REST-Python-Client"
            )

        site_url, folder_path = _parse_sharepoint_url(folder_url)
        logger.info("Connecting to SharePoint site: %s", site_url)

        ctx = ClientContext(site_url).with_interactive(
            tenant="common", client_id=self._client_id
        )

        temp_dir = Path(tempfile.mkdtemp(prefix="cobblehill_sp_"))
        logger.info("Downloading files from %s → %s", folder_path, temp_dir)

        folder = ctx.web.get_folder_by_server_relative_url(folder_path)
        files = folder.files
        ctx.load(files)
        ctx.execute_query()

        supported = set(settings.supported_extensions.keys())
        for sp_file in files:
            name: str = sp_file.properties["Name"]
            if Path(name).suffix.lower() not in supported:
                logger.debug("Skipping unsupported file: %s", name)
                continue
            local_path = temp_dir / name
            with open(local_path, "wb") as fh:
                sp_file.download(fh)
                ctx.execute_query()
            logger.info("Downloaded: %s", name)

        return temp_dir
