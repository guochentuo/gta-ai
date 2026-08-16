from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Response, status

from gta_ai import __version__
from gta_ai.clients.local_model import LocalModelClient
from gta_ai.config import Settings, get_settings
from gta_ai.internal_inference_api import create_internal_inference_router
from gta_ai.material_search import ElasticsearchMaterialSearch, MaterialSearchPort
from gta_ai.material_search_api import create_material_search_router
from gta_ai.media_audit_kafka import KafkaMediaAuditWorker, MediaAuditWorkerPort
from gta_ai.schemas import HealthState, ProviderHealth, ServiceHealth
from gta_ai.web import create_web_router


def create_app(
    *,
    settings: Settings | None = None,
    local_model_client: LocalModelClient | None = None,
    media_audit_worker: MediaAuditWorkerPort | None = None,
    material_search_service: MaterialSearchPort | None = None,
    chat_transport: httpx.AsyncBaseTransport | None = None,
    inference_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    runtime_local_client = local_model_client or LocalModelClient(runtime_settings)
    runtime_audit = media_audit_worker or KafkaMediaAuditWorker(
        runtime_settings, runtime_local_client
    )
    runtime_search = material_search_service or ElasticsearchMaterialSearch(runtime_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime_audit.start()
        try:
            yield
        finally:
            await runtime_audit.close()

    application = FastAPI(
        title=runtime_settings.app_name,
        version=__version__,
        description="Internal AI analysis and decision-support service.",
        lifespan=lifespan,
    )
    application.include_router(create_web_router(runtime_settings, transport=chat_transport))
    application.include_router(
        create_internal_inference_router(
            runtime_settings, transport=inference_transport
        )
    )
    application.include_router(create_material_search_router(runtime_settings, runtime_search))

    @application.get("/health/live", response_model=ServiceHealth)
    async def live() -> ServiceHealth:
        return ServiceHealth(version=__version__, state=HealthState.OK)

    @application.get("/health/ready", response_model=ServiceHealth)
    async def ready(response: Response) -> ServiceHealth:
        local_health = await runtime_local_client.probe()
        openai_health = ProviderHealth(
            provider="openai",
            state=(
                HealthState.CONFIGURED
                if runtime_settings.openai_is_configured
                else HealthState.NOT_CONFIGURED
            ),
            model=runtime_settings.openai_model,
        )
        ready_state = (
            HealthState.OK if local_health.state is HealthState.OK else HealthState.DEGRADED
        )
        if ready_state is not HealthState.OK:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ServiceHealth(
            version=__version__,
            state=ready_state,
            providers=[local_health, openai_health],
        )

    return application


app = create_app()
