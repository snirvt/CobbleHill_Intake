"""FastAPI server: run the dr-validity check on a SharePoint folder, return JSON.

Run: ``uv run cobblehill-api`` (binds settings.api_host:api_port, localhost by default).

Exposing it on the local network requires an API key (the server refuses to start
without one). The key is ``API_TOKEN`` in ``_SECRETS/_secrets.py`` (override with the
COBBLEHILL_API_KEY env var)::

    uv run cobblehill-api --host 0.0.0.0

Clients then send it on every request except GET /health::

    curl -H "X-API-Key: <key>" -H "Content-Type: application/json" \
         -d '{"sharepoint_folder": "..."}' http://<host>:8010/dr-validity
"""

import argparse
import logging
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

from classifier.dr_validity_service import DrValidityService
from classifier.ingest.sharepoint import SharePointAuthError
from config.settings import settings

logger = logging.getLogger(__name__)


class DrValidityRequest(BaseModel):
    """Body of POST /dr-validity."""

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


def create_app(
    service: DrValidityService | None = None, api_key: str | None | object = _FROM_SETTINGS
) -> FastAPI:
    """Build the app; the real service is created once at startup unless one is injected.

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
        if service is not None:
            app.state.service = service
        else:
            from classifier.main import build_dr_validity_service

            app.state.service = build_dr_validity_service()
        yield

    app = FastAPI(title="CobbleHill dr-validity", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @app.post("/dr-validity", dependencies=[Depends(require_api_key)])
    async def dr_validity(body: DrValidityRequest, request: Request) -> dict[str, Any]:
        """Check every patient's dr notes in the SharePoint folder; blocks until done."""
        svc: DrValidityService = request.app.state.service
        logger.info("dr-validity request for %r", body.sharepoint_folder)
        try:
            return await svc.run(body.sharepoint_folder)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("SharePoint error %s for %r: %s", status, body.sharepoint_folder, exc)
            if status == 404:
                raise HTTPException(
                    404, f"SharePoint folder not found: {body.sharepoint_folder}"
                ) from exc
            raise HTTPException(502, f"SharePoint request failed ({status})") from exc
        except SharePointAuthError as exc:
            logger.error("SharePoint auth failed: %s", exc)
            raise HTTPException(502, "SharePoint authentication failed") from exc
        except httpx.HTTPError as exc:
            logger.error("SharePoint unreachable for %r: %s", body.sharepoint_folder, exc)
            raise HTTPException(502, f"SharePoint request failed: {exc}") from exc

    return app


app = create_app()


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``uv run cobblehill-api [--host HOST] [--port PORT]``."""
    parser = argparse.ArgumentParser(description="CobbleHill dr-validity API server")
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
