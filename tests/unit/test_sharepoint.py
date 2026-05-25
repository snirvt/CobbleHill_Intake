"""Unit tests for classifier.ingest.sharepoint."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from classifier.ingest.sharepoint import (
    SharePointFolderDownloader,
    authenticate_to_graph,
    download_folder,
)

# ---------------------------------------------------------------------------
# authenticate_to_graph
# ---------------------------------------------------------------------------


@patch("classifier.ingest.sharepoint.ConfidentialClientApplication")
def test_authenticate_to_graph_returns_token(mock_app_cls: MagicMock) -> None:
    mock_app = MagicMock()
    mock_app.acquire_token_for_client.return_value = {"access_token": "tok123"}
    mock_app_cls.return_value = mock_app

    token = authenticate_to_graph("cid", "csecret", "tid")

    assert token == "tok123"
    mock_app_cls.assert_called_once_with(
        "cid",
        authority="https://login.microsoftonline.com/tid",
        client_credential="csecret",
    )


@patch("classifier.ingest.sharepoint.ConfidentialClientApplication")
def test_authenticate_to_graph_raises_on_error(mock_app_cls: MagicMock) -> None:
    mock_app = MagicMock()
    mock_app.acquire_token_for_client.return_value = {
        "error": "invalid_client",
        "error_description": "bad secret",
    }
    mock_app_cls.return_value = mock_app

    with pytest.raises(RuntimeError, match="MSAL authentication failed"):
        authenticate_to_graph("cid", "csecret", "tid")


# ---------------------------------------------------------------------------
# download_folder
# ---------------------------------------------------------------------------

DRIVE_ID = "test-drive"


def _file_item(name: str, item_id: str = "fid") -> dict:  # type: ignore[type-arg]
    return {"name": name, "id": item_id}


def _folder_item(name: str) -> dict:  # type: ignore[type-arg]
    return {"name": name, "id": f"did-{name}", "folder": {}}


def _make_client(url_responses: dict[str, object]) -> MagicMock:
    """Return an async-capable mock client whose get() dispatches by URL fragment."""

    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        for fragment, payload in url_responses.items():
            if fragment in url:
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                if isinstance(payload, dict):
                    resp.json.return_value = payload
                else:
                    resp.content = payload
                return resp
        raise AssertionError(f"Unexpected URL in test: {url}")

    client = MagicMock()
    client.get = fake_get
    return client


async def test_download_folder_downloads_supported_files(tmp_path: Path) -> None:
    semaphore = asyncio.Semaphore(5)
    client = _make_client(
        {
            "children": {"value": [
                _file_item("report.pdf", "id1"),
                _file_item("note.txt", "id2"),
                _file_item("ignore.docx", "id3"),
            ]},
            "items/id1": b"pdf-data",
            "items/id2": b"txt-data",
        }
    )

    await download_folder(DRIVE_ID, "Folder/Sub", tmp_path, client, semaphore)

    assert (tmp_path / "report.pdf").read_bytes() == b"pdf-data"
    assert (tmp_path / "note.txt").read_bytes() == b"txt-data"
    assert not (tmp_path / "ignore.docx").exists()


async def test_download_folder_recurses_into_subfolders(tmp_path: Path) -> None:
    semaphore = asyncio.Semaphore(5)
    client = _make_client(
        {
            "root:/Root:": {"value": [_folder_item("sub")]},
            "root:/Root/sub:": {"value": [_file_item("file.pdf", "fid1")]},
            "items/fid1": b"bytes",
        }
    )

    await download_folder(DRIVE_ID, "Root", tmp_path, client, semaphore)

    assert (tmp_path / "sub" / "file.pdf").read_bytes() == b"bytes"


# ---------------------------------------------------------------------------
# SharePointFolderDownloader
# ---------------------------------------------------------------------------

_KWARGS = dict(
    client_id="cid",
    client_secret="csecret",
    tenant_id="tid",
    drive_id="did",
)


@patch("classifier.ingest.sharepoint.authenticate_to_graph", return_value="tok")
@patch("classifier.ingest.sharepoint.download_folder", new_callable=AsyncMock)
async def test_downloader_uses_provided_local_dir(
    mock_dl: AsyncMock, _mock_auth: MagicMock, tmp_path: Path
) -> None:
    downloader = SharePointFolderDownloader(**_KWARGS)
    result = await downloader.download("Some/Folder", local_dir=tmp_path)

    assert result == tmp_path
    mock_dl.assert_called_once()
    args = mock_dl.call_args.args
    assert args[0] == "did"
    assert args[1] == "Some/Folder"
    assert args[2] == tmp_path


@patch("classifier.ingest.sharepoint.authenticate_to_graph", return_value="tok")
@patch("classifier.ingest.sharepoint.download_folder", new_callable=AsyncMock)
async def test_downloader_creates_tmp_dir_when_no_local_dir(
    mock_dl: AsyncMock, _mock_auth: MagicMock
) -> None:
    downloader = SharePointFolderDownloader(**_KWARGS)
    result = await downloader.download("Some/Folder")

    assert result.exists()
    assert result.name.startswith("cobblehill_sp_")
    result.rmdir()
