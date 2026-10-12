"""API tests: real FastAPI app, fake service (no SharePoint, no LLM)."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from classifier.api import create_app
from classifier.ingest.sharepoint import SharePointAuthError

_REPORT = {
    "sharepoint_folder": "Pending",
    "web_url": "https://sp/Pending",
    "summary": {"total": 1, "valid": 1, "invalid": 0, "unknown_age": 0, "failed": 0},
    "groups": [{"main_diagnosis": "GERD", "count": 1}],
    "patients": [{"patient": "DOE, JANE", "success": True, "verdict": "VALID"}],
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
    with TestClient(create_app(service=service)) as c:
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
