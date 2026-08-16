from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, HTTPException, Response, status

from gta_ai.config import Settings
from gta_ai.media_audit import MediaAuditCoordinatorPort, MediaAuditError
from gta_ai.schemas import MediaAuditAccepted, MediaAuditRequest, MediaAuditState


def create_media_audit_router(
    settings: Settings,
    coordinator: MediaAuditCoordinatorPort,
) -> APIRouter:
    router = APIRouter(prefix="/v1/media-audits", tags=["media-audit"])

    def authorize(authorization: str | None) -> None:
        if not settings.media_audit_enabled:
            raise HTTPException(status_code=503, detail="media audit is disabled")
        expected_secret = settings.media_audit_api_token
        if expected_secret is None:
            raise HTTPException(status_code=503, detail="media audit token is not configured")
        expected = f"Bearer {expected_secret.get_secret_value()}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid media audit credential")

    @router.post("", response_model=MediaAuditAccepted, status_code=status.HTTP_202_ACCEPTED)
    async def submit(
        request: MediaAuditRequest,
        authorization: str | None = Header(default=None),
    ) -> MediaAuditAccepted:
        authorize(authorization)
        try:
            return await coordinator.submit(request)
        except MediaAuditError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.delete("/{request_id}", response_model=MediaAuditState)
    async def cancel(
        request_id: str,
        authorization: str | None = Header(default=None),
    ) -> MediaAuditState:
        authorize(authorization)
        return await coordinator.cancel(request_id)

    @router.get("/{request_id}", response_model=MediaAuditState)
    async def get_status(
        request_id: str,
        response: Response,
        authorization: str | None = Header(default=None),
    ) -> MediaAuditState:
        authorize(authorization)
        current = await coordinator.status(request_id)
        if current.state == "unknown":
            response.status_code = status.HTTP_404_NOT_FOUND
        return current

    return router
