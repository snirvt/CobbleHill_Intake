"""API tests: real FastAPI app, fake service (no SharePoint, no LLM)."""

import logging
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from classifier.api import create_app, main
from classifier.ingest.sharepoint import SharePointAuthError
from config.settings import settings

_REPORT = {
    "sharepoint_folder": "Pending",
    "web_url": "https://sp/Pending",
    "summary": {
        "total": 1, "valid": 1, "invalid": 0, "unknown_age": 0, "failed": 0,
        "skipped_folders": 1, "errors": 1,
    },
    "groups": [{"main_diagnosis": "GERD", "count": 1}],
    "patients": [{"patient": "DOE, JANE", "success": True, "verdict": "VALID", "errors": []}],
    "errors": [
        {
            "patient": "EMPTY",
            "file": None,
            "stage": "folder",
            "code": "empty_folder",
            "message": "Folder has no files",
        }
    ],
}


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://graph")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status))


@pytest.fixture
def service() -> MagicMock:
    svc = MagicMock()
    svc.run = AsyncMock(return_value=_REPORT)
    return svc


@pytest.fixture
def client(service: MagicMock) -> Iterator[TestClient]:
    # Explicitly open (no key), regardless of any COBBLEHILL_API_KEY in the environment.
    with TestClient(create_app(service=service, api_key=None)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_dr_validity_returns_report(client: TestClient, service: MagicMock) -> None:
    resp = client.post("/dr-validity", json={"sharepoint_folder": "/Pending/"})

    assert resp.status_code == 200
    assert resp.json() == _REPORT
    service.run.assert_awaited_once_with("Pending")


@pytest.mark.parametrize("body", [{"sharepoint_folder": "  "}, {"sharepoint_folder": "/"}, {}])
def test_dr_validity_rejects_blank_folder(client: TestClient, body: dict[str, str]) -> None:
    assert client.post("/dr-validity", json=body).status_code == 422


def test_missing_sharepoint_folder_is_404(client: TestClient, service: MagicMock) -> None:
    service.run.side_effect = _http_error(404)
    resp = client.post("/dr-validity", json={"sharepoint_folder": "Nope"})
    assert resp.status_code == 404
    assert "Nope" in resp.json()["detail"]


def test_other_sharepoint_error_is_502(client: TestClient, service: MagicMock) -> None:
    service.run.side_effect = _http_error(500)
    assert client.post("/dr-validity", json={"sharepoint_folder": "Pending"}).status_code == 502


def test_sharepoint_unreachable_is_502(client: TestClient, service: MagicMock) -> None:
    service.run.side_effect = httpx.ConnectError("no route")
    assert client.post("/dr-validity", json={"sharepoint_folder": "Pending"}).status_code == 502


def test_sharepoint_auth_failure_is_502(client: TestClient, service: MagicMock) -> None:
    service.run.side_effect = SharePointAuthError("bad secret")
    resp = client.post("/dr-validity", json={"sharepoint_folder": "Pending"})
    assert resp.status_code == 502
    assert "bad secret" not in resp.json()["detail"]


# ---------------------------------------------------------------------------
# main(): --host / --port
# ---------------------------------------------------------------------------

@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", None)


@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_key", SecretStr("s3cret"))


@pytest.mark.usefixtures("no_api_key")
def test_main_defaults_to_localhost_without_key(caplog: pytest.LogCaptureFixture) -> None:
    with patch("classifier.api.uvicorn.run") as run, caplog.at_level(logging.WARNING):
        main([])

    assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 8010}
    assert "X-API-Key" not in caplog.text


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("   ")])
def test_main_refuses_network_host_without_key(
    key: SecretStr | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "api_key", key)
    with patch("classifier.api.uvicorn.run") as run, pytest.raises(SystemExit) as exc:
        main(["--host", "0.0.0.0"])

    run.assert_not_called()
    assert "API_TOKEN" in str(exc.value.code)


@pytest.mark.usefixtures("with_api_key")
def test_main_network_host_with_key_starts(caplog: pytest.LogCaptureFixture) -> None:
    with patch("classifier.api.uvicorn.run") as run, caplog.at_level(logging.WARNING):
        main(["--host", "0.0.0.0", "--port", "9000"])

    assert run.call_args.kwargs == {"host": "0.0.0.0", "port": 9000}
    assert "X-API-Key required" in caplog.text
    assert "s3cret" not in caplog.text


# ---------------------------------------------------------------------------
# X-API-Key enforcement
# ---------------------------------------------------------------------------

@pytest.fixture
def keyed_client(service: MagicMock) -> Iterator[TestClient]:
    with TestClient(create_app(service=service, api_key="s3cret")) as c:
        yield c


_BODY = {"sharepoint_folder": "Pending"}


def test_correct_api_key_is_accepted(keyed_client: TestClient) -> None:
    resp = keyed_client.post("/dr-validity", json=_BODY, headers={"X-API-Key": "s3cret"})
    assert resp.status_code == 200
    assert resp.json() == _REPORT


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_missing_or_wrong_api_key_is_401(
    keyed_client: TestClient, service: MagicMock, headers: dict[str, str]
) -> None:
    resp = keyed_client.post("/dr-validity", json=_BODY, headers=headers)

    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid or missing API key"}
    service.run.assert_not_awaited()


def test_health_needs_no_api_key(keyed_client: TestClient) -> None:
    assert keyed_client.get("/health").status_code == 200


def test_app_uses_settings_key_by_default(
    service: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "api_key", SecretStr("from-settings"))
    with TestClient(create_app(service=service)) as c:
        assert c.post("/dr-validity", json=_BODY).status_code == 401
        ok = c.post("/dr-validity", json=_BODY, headers={"X-API-Key": "from-settings"})
        assert ok.status_code == 200


@pytest.mark.parametrize("key", [SecretStr(""), SecretStr("  ")])
def test_blank_settings_key_means_open_only_on_localhost(
    key: SecretStr, service: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank key is treated as unset (and main() refuses to expose it, tested above)."""
    monkeypatch.setattr(settings, "api_key", key)
    with TestClient(create_app(service=service)) as c:
        assert c.post("/dr-validity", json=_BODY).status_code == 200


def test_settings_api_key_defaults_to_secrets_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from _SECRETS._secrets import API_TOKEN
    from config.settings import Settings

    monkeypatch.delenv("COBBLEHILL_API_KEY", raising=False)
    key = Settings().api_key
    assert key is not None and key.get_secret_value() == API_TOKEN
