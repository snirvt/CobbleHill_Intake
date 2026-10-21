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
    "category_groups": [{"diagnosis_category": "Reflux / GERD", "count": 1}],
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
    with TestClient(create_app(service=service, match_service=service, api_key=None)) as c:
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
# POST /dr-nurse-match
# ---------------------------------------------------------------------------

_MATCH_REPORT = {
    "sharepoint_folder": "Pending",
    "web_url": "https://sp/Pending",
    "summary": {
        "total": 1, "match": 1, "mismatch": 0, "undecided": 0, "failed": 0,
        "skipped_folders": 1, "errors": 1,
    },
    "groups": [{"main_diagnosis": "GERD", "count": 1}],
    "category_groups": [{"diagnosis_category": "Reflux / GERD", "count": 1}],
    "pairs": [{"patient": "DOE, JANE", "success": True, "overall": "MATCH", "errors": []}],
    "errors": [
        {
            "patient": "DR, ONLY",
            "file": None,
            "stage": "folder",
            "code": "missing_required_notes",
            "message": "Folder has no nurse notes",
        }
    ],
}


@pytest.fixture
def match_service() -> MagicMock:
    svc = MagicMock()
    svc.run = AsyncMock(return_value=_MATCH_REPORT)
    return svc


@pytest.fixture
def match_client(service: MagicMock, match_service: MagicMock) -> Iterator[TestClient]:
    with TestClient(
        create_app(service=service, match_service=match_service, api_key=None)
    ) as c:
        yield c


def test_dr_nurse_match_returns_report(
    match_client: TestClient, service: MagicMock, match_service: MagicMock
) -> None:
    resp = match_client.post("/dr-nurse-match", json={"sharepoint_folder": "/Pending/"})

    assert resp.status_code == 200
    assert resp.json() == _MATCH_REPORT
    match_service.run.assert_awaited_once_with("Pending")
    service.run.assert_not_awaited()


@pytest.mark.parametrize("body", [{"sharepoint_folder": "  "}, {}])
def test_dr_nurse_match_rejects_blank_folder(
    match_client: TestClient, body: dict[str, str]
) -> None:
    assert match_client.post("/dr-nurse-match", json=body).status_code == 422


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (_http_error(404), 404),
        (_http_error(500), 502),
        (httpx.ConnectError("no route"), 502),
        (SharePointAuthError("bad secret"), 502),
    ],
)
def test_dr_nurse_match_maps_sharepoint_errors(
    match_client: TestClient, match_service: MagicMock, error: Exception, status: int
) -> None:
    match_service.run.side_effect = error
    resp = match_client.post("/dr-nurse-match", json={"sharepoint_folder": "Nope"})
    assert resp.status_code == status


def test_dr_nurse_match_requires_api_key(
    service: MagicMock, match_service: MagicMock
) -> None:
    app = create_app(service=service, match_service=match_service, api_key="s3cret")
    with TestClient(app) as c:
        assert c.post("/dr-nurse-match", json={"sharepoint_folder": "P"}).status_code == 401
        ok = c.post(
            "/dr-nurse-match", json={"sharepoint_folder": "P"}, headers={"X-API-Key": "s3cret"}
        )
        assert ok.status_code == 200
    match_service.run.assert_awaited_once()


def test_real_services_share_one_set_of_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from classifier import main as wiring

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        wiring, "build_dr_validity_service", lambda limits: seen.setdefault("validity", limits)
    )
    monkeypatch.setattr(
        wiring, "build_dr_nurse_match_service", lambda limits: seen.setdefault("match", limits)
    )
    with TestClient(create_app(api_key=None)):
        pass
    assert isinstance(seen["validity"], wiring.ConcurrencyLimits)
    assert seen["validity"] is seen["match"]


# ---------------------------------------------------------------------------
# GET .../example
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("route", "items_key"),
    [("/dr-validity/example", "patients"), ("/dr-nurse-match/example", "pairs")],
)
def test_example_routes_return_sample_without_running(
    match_client: TestClient,
    service: MagicMock,
    match_service: MagicMock,
    route: str,
    items_key: str,
) -> None:
    resp = match_client.get(route)

    assert resp.status_code == 200
    body = resp.json()
    assert list(body) == [
        "sharepoint_folder", "web_url", "summary", "groups", "category_groups", items_key,
        "errors",
    ]
    assert "EXAMPLE" in body["sharepoint_folder"]
    service.run.assert_not_awaited()
    match_service.run.assert_not_awaited()


@pytest.mark.parametrize("route", ["/dr-validity/example", "/dr-nurse-match/example"])
def test_example_routes_require_api_key(keyed_client: TestClient, route: str) -> None:
    assert keyed_client.get(route).status_code == 401
    assert keyed_client.get(route, headers={"X-API-Key": "s3cret"}).status_code == 200

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
    with TestClient(create_app(service=service, match_service=service, api_key="s3cret")) as c:
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


def test_version_reports_injected_commit_and_start_time(service: MagicMock) -> None:
    app = create_app(service=service, match_service=service, api_key=None, commit="abc1234")
    with TestClient(app) as c:
        body = c.get("/version").json()
    assert body["commit"] == "abc1234"
    assert body["started_at"].endswith("+00:00")


def test_version_requires_api_key(keyed_client: TestClient) -> None:
    assert keyed_client.get("/version").status_code == 401
    assert keyed_client.get("/version", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_app_uses_settings_key_by_default(
    service: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "api_key", SecretStr("from-settings"))
    with TestClient(create_app(service=service, match_service=service)) as c:
        assert c.post("/dr-validity", json=_BODY).status_code == 401
        ok = c.post("/dr-validity", json=_BODY, headers={"X-API-Key": "from-settings"})
        assert ok.status_code == 200


@pytest.mark.parametrize("key", [SecretStr(""), SecretStr("  ")])
def test_blank_settings_key_means_open_only_on_localhost(
    key: SecretStr, service: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank key is treated as unset (and main() refuses to expose it, tested above)."""
    monkeypatch.setattr(settings, "api_key", key)
    with TestClient(create_app(service=service, match_service=service)) as c:
        assert c.post("/dr-validity", json=_BODY).status_code == 200


def test_settings_api_key_defaults_to_secrets_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from _SECRETS._secrets import API_TOKEN
    from config.settings import Settings

    monkeypatch.delenv("COBBLEHILL_API_KEY", raising=False)
    key = Settings().api_key
    assert key is not None and key.get_secret_value() == API_TOKEN
