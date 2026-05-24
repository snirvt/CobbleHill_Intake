"""Unit tests for classifier.ingest.sharepoint."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

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
HEADERS = {"Authorization": "Bearer tok"}


def _file_item(name: str, item_id: str = "fid") -> dict:  # type: ignore[type-arg]
    return {"name": name, "id": item_id}


def _folder_item(name: str) -> dict:  # type: ignore[type-arg]
    return {"name": name, "id": f"did-{name}", "folder": {}}


@patch("classifier.ingest.sharepoint.requests.get")
def test_download_folder_downloads_supported_files(
    mock_get: MagicMock, tmp_path: Path
) -> None:
    list_resp = MagicMock()
    list_resp.json.return_value = {
        "value": [
            _file_item("report.pdf", "id1"),
            _file_item("note.txt", "id2"),
            _file_item("ignore.docx", "id3"),
        ]
    }
    dl_pdf = MagicMock()
    dl_pdf.content = b"pdf-data"
    dl_txt = MagicMock()
    dl_txt.content = b"txt-data"

    mock_get.side_effect = [list_resp, dl_pdf, dl_txt]

    download_folder(DRIVE_ID, "Folder/Sub", tmp_path, HEADERS)

    assert (tmp_path / "report.pdf").read_bytes() == b"pdf-data"
    assert (tmp_path / "note.txt").read_bytes() == b"txt-data"
    assert not (tmp_path / "ignore.docx").exists()


@patch("classifier.ingest.sharepoint.requests.get")
def test_download_folder_recurses_into_subfolders(
    mock_get: MagicMock, tmp_path: Path
) -> None:
    root_resp = MagicMock()
    root_resp.json.return_value = {"value": [_folder_item("sub")]}

    sub_resp = MagicMock()
    sub_resp.json.return_value = {"value": [_file_item("file.pdf", "fid1")]}

    dl_resp = MagicMock()
    dl_resp.content = b"bytes"

    mock_get.side_effect = [root_resp, sub_resp, dl_resp]

    download_folder(DRIVE_ID, "Root", tmp_path, HEADERS)

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
@patch("classifier.ingest.sharepoint.download_folder")
def test_downloader_uses_provided_local_dir(
    mock_dl: MagicMock, _mock_auth: MagicMock, tmp_path: Path
) -> None:
    downloader = SharePointFolderDownloader(**_KWARGS)
    result = downloader.download("Some/Folder", local_dir=tmp_path)

    assert result == tmp_path
    mock_dl.assert_called_once_with("did", "Some/Folder", tmp_path, {"Authorization": "Bearer tok"})


@patch("classifier.ingest.sharepoint.authenticate_to_graph", return_value="tok")
@patch("classifier.ingest.sharepoint.download_folder")
def test_downloader_creates_tmp_dir_when_no_local_dir(
    mock_dl: MagicMock, _mock_auth: MagicMock
) -> None:
    downloader = SharePointFolderDownloader(**_KWARGS)
    result = downloader.download("Some/Folder")

    assert result.exists()
    assert result.name.startswith("cobblehill_sp_")
    result.rmdir()
