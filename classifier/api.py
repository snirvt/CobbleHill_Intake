"""FastAPI server: run the dr-validity check on a SharePoint folder, return JSON.

Run: ``uv run cobblehill-api`` (binds settings.api_host:api_port, localhost by default).
There is no auth, so keep it bound to localhost.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
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


def create_app(service: DrValidityService | None = None) -> FastAPI:
    """Build the app; the real service is created once at startup unless one is injected."""

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

    @app.post("/dr-validity")
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


def main() -> None:
    """Entry point for ``uv run cobblehill-api``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
