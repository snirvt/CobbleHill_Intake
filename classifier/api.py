"""FastAPI server: run a check on a SharePoint folder, return JSON.

Routes (all take ``{"sharepoint_folder": "..."}`` and block until done):

- ``POST /dr-validity``: every patient's dr notes against the intake rules.
- ``POST /dr-nurse-match``: every same-category dr + nurse note pair compared. Folders
  without such a pair are reported under ``errors``, not processed.
- ``POST /intake-review``: per patient, Phase I (dr-validity rules) then — only if it
  passed — Phase II (dr + nurse notes must match); ``verdict_phase_1``,
  ``verdict_phase_2``, a final ACCEPT / REJECT / INCOMPLETE ``verdict`` and an
  ``explanation``.

``GET /<route>/example`` returns a sample response of each in exactly the same format,
built from fake data (no SharePoint, no LLM, no patient data).

Run: ``uv run cobblehill-api`` (binds settings.api_host:api_port, localhost by default).

Exposing it on the local network requires an API key (the server refuses to start
without one). The key is ``API_TOKEN`` in ``_SECRETS/_secrets.py`` (override with the
COBBLEHILL_API_KEY env var)::

    uv run cobblehill-api --host 0.0.0.0

Clients then send it on every request except GET /health::

    curl -H "X-API-Key: <key>" -H "Content-Type: application/json" \
         -d '{"sharepoint_folder": "..."}' http://<host>:8010/dr-validity

``source scripts/api_env.sh`` sets ``$KEY`` and ``$HOST`` for these curl calls.

The response lists every problem met along the way under ``errors`` (top level) and per
patient/pair under ``patients[].errors`` / ``pairs[].errors``: empty folders, folders
without the required notes, notes with no extractable text, and LLM steps that failed
after retries. SharePoint failures still return HTTP 404/502.
"""

import argparse
import logging
import secrets
import sys
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

from classifier.folder_service import FolderReportService
from classifier.ingest.sharepoint import SharePointAuthError
from classifier.models import (
    DrValidityPipelineResult,
    IntakeReviewPipelineResult,
    PairPipelineResult,
)
from classifier.output.examples import (
    dr_nurse_match_example,
    dr_validity_example,
    intake_review_example,
)
from classifier.version import git_commit
from config.settings import settings

logger = logging.getLogger(__name__)


class FolderRequest(BaseModel):
    """Body of POST /dr-validity, /dr-nurse-match and /intake-review."""

    sharepoint_folder: str

    @field_validator("sharepoint_folder")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        folder = value.strip().strip("/")
        if not folder:
            raise ValueError("sharepoint_folder must not be blank")
        return folder


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
# Distinguishes "use settings.api_key" from an explicit "no key" (None).
_FROM_SETTINGS = object()


def _settings_api_key() -> str | None:
    """The configured key, or None when unset or blank (an empty env var is not a key)."""
    if settings.api_key is None:
        return None
    return settings.api_key.get_secret_value().strip() or None


async def _run_report(report: Awaitable[dict[str, Any]], folder: str) -> dict[str, Any]:
    """Await a service run, mapping SharePoint failures to HTTP 404/502."""
    try:
        return await report
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.error("SharePoint error %s for %r: %s", status, folder, exc)
        if status == 404:
            raise HTTPException(404, f"SharePoint folder not found: {folder}") from exc
        raise HTTPException(502, f"SharePoint request failed ({status})") from exc
    except SharePointAuthError as exc:
        logger.error("SharePoint auth failed: %s", exc)
        raise HTTPException(502, "SharePoint authentication failed") from exc
    except httpx.HTTPError as exc:
        logger.error("SharePoint unreachable for %r: %s", folder, exc)
        raise HTTPException(502, f"SharePoint request failed: {exc}") from exc


def create_app(
    service: FolderReportService[DrValidityPipelineResult] | None = None,
    api_key: str | None | object = _FROM_SETTINGS,
    match_service: FolderReportService[PairPipelineResult] | None = None,
    commit: str | None = None,
    review_service: FolderReportService[IntakeReviewPipelineResult] | None = None,
) -> FastAPI:
    """Build the app; the real services are created once at startup unless injected.

    ``commit`` is what /version reports; defaults to the repo's git HEAD at startup.

    ``service`` backs /dr-validity, ``match_service`` backs /dr-nurse-match and
    ``review_service`` backs /intake-review. Real services share one set of limits
    (classifier.main.ConcurrencyLimits), so all concurrent requests to every route together
    stay within max_concurrent_llm_calls, max_concurrent_files and
    sharepoint_download_concurrency.

    api_key defaults to settings.api_key. When set, every route except /health requires a
    matching X-API-Key header; when None, the API is open (localhost-only use).
    """
    expected_key = (
        _settings_api_key() if api_key is _FROM_SETTINGS else cast("str | None", api_key) or None
    )

    async def require_api_key(provided: str | None = Security(_API_KEY_HEADER)) -> None:
        """Reject the request unless it carries the configured API key."""
        if expected_key is None:
            return
        if provided is None or not secrets.compare_digest(
            provided.encode(), expected_key.encode()
        ):
            raise HTTPException(401, "Invalid or missing API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from classifier.main import (
            build_dr_nurse_match_service,
            build_dr_validity_service,
            build_intake_review_service,
            make_concurrency_limits,
        )

        limits = make_concurrency_limits()
        app.state.service = service or build_dr_validity_service(limits)
        app.state.match_service = match_service or build_dr_nurse_match_service(limits)
        app.state.review_service = review_service or build_intake_review_service(limits)
        app.state.commit = commit or git_commit(settings.base_path)
        app.state.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        yield

    app = FastAPI(title="CobbleHill intake checks", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @app.get("/version", dependencies=[Depends(require_api_key)])
    async def version(request: Request) -> dict[str, str]:
        """Commit the running server was started from, and when it started."""
        return {"commit": request.app.state.commit, "started_at": request.app.state.started_at}

    @app.post("/dr-validity", dependencies=[Depends(require_api_key)])
    async def dr_validity(body: FolderRequest, request: Request) -> dict[str, Any]:
        """Check every patient's dr notes in the SharePoint folder; blocks until done."""
        svc: FolderReportService[DrValidityPipelineResult] = request.app.state.service
        logger.info("dr-validity request for %r", body.sharepoint_folder)
        return await _run_report(svc.run(body.sharepoint_folder), body.sharepoint_folder)

    @app.post("/dr-nurse-match", dependencies=[Depends(require_api_key)])
    async def dr_nurse_match(body: FolderRequest, request: Request) -> dict[str, Any]:
        """Compare every dr + nurse note pair in the SharePoint folder; blocks until done."""
        svc: FolderReportService[PairPipelineResult] = request.app.state.match_service
        logger.info("dr-nurse-match request for %r", body.sharepoint_folder)
        return await _run_report(svc.run(body.sharepoint_folder), body.sharepoint_folder)

    @app.post("/intake-review", dependencies=[Depends(require_api_key)])
    async def intake_review(body: FolderRequest, request: Request) -> dict[str, Any]:
        """Two-phase accept/reject review of every patient in the folder; blocks until done."""
        svc: FolderReportService[IntakeReviewPipelineResult] = request.app.state.review_service
        logger.info("intake-review request for %r", body.sharepoint_folder)
        return await _run_report(svc.run(body.sharepoint_folder), body.sharepoint_folder)

    @app.get("/dr-validity/example", dependencies=[Depends(require_api_key)])
    async def dr_validity_example_route() -> dict[str, Any]:
        """Sample /dr-validity response built from fake data, to show the output format."""
        return dr_validity_example()

    @app.get("/dr-nurse-match/example", dependencies=[Depends(require_api_key)])
    async def dr_nurse_match_example_route() -> dict[str, Any]:
        """Sample /dr-nurse-match response built from fake data, to show the output format."""
        return dr_nurse_match_example()

    @app.get("/intake-review/example", dependencies=[Depends(require_api_key)])
    async def intake_review_example_route() -> dict[str, Any]:
        """Sample /intake-review response built from fake data, to show the output format."""
        return intake_review_example()

    return app


app = create_app()


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``uv run cobblehill-api [--host HOST] [--port PORT]``."""
    parser = argparse.ArgumentParser(description="CobbleHill intake-checks API server")
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help="Interface to bind; use 0.0.0.0 to listen on all interfaces (local network)",
    )
    parser.add_argument("--port", type=int, default=settings.api_port, help="Port to listen on")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.host not in _LOOPBACK_HOSTS:
        if _settings_api_key() is None:
            sys.exit(
                f"Refusing to listen on {args.host}: the API returns patient data and no API "
                "key is set. Set API_TOKEN in _SECRETS/_secrets.py (or COBBLEHILL_API_KEY)."
            )
        logger.warning(
            "API reachable from the network on %s:%s; X-API-Key required", args.host, args.port
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
