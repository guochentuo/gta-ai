from __future__ import annotations

import hmac

from fastapi import APIRouter, Header, HTTPException

from gta_ai.config import Settings
from gta_ai.material_search import MaterialSearchError, MaterialSearchPort
from gta_ai.schemas.material_search import MaterialSearchRequest, MaterialSearchResult


def create_material_search_router(
    settings: Settings, search_service: MaterialSearchPort
) -> APIRouter:
    router = APIRouter(prefix="/v1/material-search", tags=["material-search"])

    def authorize(authorization: str | None) -> None:
        if not settings.material_search_enabled:
            raise HTTPException(status_code=503, detail="material search is disabled")
        secret = settings.media_audit_api_token
        if secret is None:
            raise HTTPException(status_code=503, detail="material search token is not configured")
        expected = f"Bearer {secret.get_secret_value()}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid material search credential")

    @router.post("", response_model=MaterialSearchResult)
    async def search(
        request: MaterialSearchRequest,
        authorization: str | None = Header(default=None),
    ) -> MaterialSearchResult:
        authorize(authorization)
        try:
            return await search_service.search(request)
        except MaterialSearchError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return router
