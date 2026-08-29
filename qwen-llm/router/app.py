# ruff: noqa: F401
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .request_state import PriorityWorkloadGate, RequestRegistry, RuntimeState, TrackedRequest
from .retrieval import ElasticsearchRetriever, RetrievalConfig, RetrievalResult
from .router_config import RouterConfig
from .runtime_logging import configure as configure_logging
from .runtime_logging import error as log_error
from .runtime_logging import info as log_info
from .runtime_logging import shutdown as shutdown_logging
from .services.chat_pipeline_service import (
    COMPANY_RECOMMENDATION_PROMPT,
    CONFIRMATION_FOLLOWUP_PROMPT,
    LANGUAGE_MATCH_PROMPT,
    PLAYFUL_QUERY_PROMPT,
    QUESTION_HISTORY_PROMPT,
    RETRIEVAL_PROMPT,
    SIMPLE_CHAT_PROMPT,
    SYSTEM_PROMPT,
    TRAVEL_SUPPORT_PROMPT,
    VERIFIED_BUSINESS_FACT_PROMPT,
    _asks_about_identity,
    _clean_decorative_symbols,
    _clean_response_identity,
    _IdentityPrefixFilter,
    _image_marker_sse_event,
    _ImageMarkerStreamFilter,
    _latest_user_text,
    _response_text,
    _response_usage,
    contact_request_event,
    contextual_retrieval_query,
    conversation_may_need_retrieval,
    direct_retrieval_query,
    generate_handoff_offer,
    handoff_offer_event,
    inject_knowledge_tool,
    inject_persona,
    language_style_prompt,
    normalize_retrieval_query,
    query_explicitly_requests_contact,
    retrieval_context,
    retrieval_image_event,
    retrieval_tool_query,
    simple_chat_query,
    stream_tool_state,
)
from .services.context_classification_service import (
    business_fact_evidence_found,
    is_company_recommendation_query,
    is_confirmation_followup,
    is_playful_or_impossible_travel_query,
    is_travel_support_query,
    is_user_question_history_query,
    requires_verified_business_fact,
)
from .services.gpu_admission_service import (
    LeaseAcquireRequest,
    LeaseHeartbeatRequest,
    PersistentGpuAdmission,
    _gpu_memory_snapshot,
    _gpu_summary,
    _memory_gb,
    _metric_values,
)
from .services.pricing_service import TripPricingService, parse_quote_intent, pricing_context
from .services.welcome_localization_service import (
    WelcomeLocalizationConfig,
    WelcomeLocalizationService,
    normalize_welcome_locale,
    welcome_handoff_event,
    welcome_handoff_payload,
)
from .services.welcome_template_service import WelcomeTemplateService
from .summarization import ConversationSummarizer, SummarizationConfig, estimate_tokens

WORKLOAD_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@/-]{1,160}$")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@/-]{1,160}$")


class RequestCancelled(Exception):
    pass


async def _wait_for_request_cancel(request: Request, cancel_event: asyncio.Event) -> None:
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            break
        await asyncio.sleep(0.05)


async def _await_cancelable(awaitable, cancel_event: asyncio.Event):
    operation = asyncio.ensure_future(awaitable)
    cancellation = asyncio.create_task(cancel_event.wait())
    try:
        done, _ = await asyncio.wait({operation, cancellation}, return_when=asyncio.FIRST_COMPLETED)
        # If acquisition and cancellation become ready in the same loop turn,
        # the acquired resource must be returned to the caller so its acquired
        # flag is set and the single cleanup path can release it.
        if operation in done:
            cancellation.cancel()
            return await operation
        if cancellation in done:
            operation.cancel()
            with suppress(BaseException):
                await operation
            raise RequestCancelled("inference request was cancelled or disconnected")
        raise RequestCancelled("inference request was cancelled or disconnected")
    except BaseException:
        if not operation.done():
            operation.cancel()
        if not cancellation.done():
            cancellation.cancel()
        raise


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


async def _acquire_inference_gate(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o660)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _release_inference_gate(descriptor: int | None) -> None:
    if descriptor is None:
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def create_router_app(
    config: RouterConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    memory_provider: Callable[[], tuple[int, int]] = _gpu_memory_snapshot,
) -> FastAPI:
    runtime_config = config or RouterConfig.from_environment()
    runtime_state = RuntimeState(runtime_config.state_path)
    workload_gate = PriorityWorkloadGate(runtime_config.max_concurrent_workloads)
    request_registry = RequestRegistry()
    admission = PersistentGpuAdmission(
        runtime_config.admission_db_path,
        poll_seconds=runtime_config.admission_poll_seconds,
        minimum_free_mb=runtime_config.admission_minimum_free_mb,
        maximum_active=runtime_config.admission_max_active,
        memory_provider=memory_provider,
    )
    retriever = ElasticsearchRetriever(
        RetrievalConfig(
            enabled=runtime_config.retrieval_enabled,
            embedding_url=runtime_config.retrieval_embedding_url,
            embedding_model=runtime_config.retrieval_embedding_model,
            elasticsearch_url=runtime_config.retrieval_elasticsearch_url,
            elasticsearch_username=runtime_config.retrieval_elasticsearch_username,
            elasticsearch_password=runtime_config.retrieval_elasticsearch_password,
            elasticsearch_indices=runtime_config.retrieval_elasticsearch_indices,
            verify_tls=runtime_config.retrieval_verify_tls,
            timeout_seconds=runtime_config.retrieval_timeout_seconds,
            top_k=runtime_config.retrieval_top_k,
            context_top_k=runtime_config.retrieval_context_top_k,
            num_candidates=runtime_config.retrieval_num_candidates,
            max_query_chars=runtime_config.retrieval_max_query_chars,
            max_context_chars=runtime_config.retrieval_max_context_chars,
            max_document_chars=runtime_config.retrieval_max_document_chars,
            min_score=runtime_config.retrieval_min_score,
            image_min_score=runtime_config.retrieval_image_min_score,
        ),
        transport=transport,
    )
    pricing_service = TripPricingService(
        runtime_config.quote_url,
        runtime_config.quote_timeout_seconds,
        transport=transport,
    )
    welcome_templates = WelcomeTemplateService(
        embedding_url=runtime_config.retrieval_embedding_url,
        embedding_model=runtime_config.retrieval_embedding_model,
        elasticsearch_url=runtime_config.retrieval_elasticsearch_url,
        elasticsearch_username=runtime_config.retrieval_elasticsearch_username,
        elasticsearch_password=runtime_config.retrieval_elasticsearch_password,
        elasticsearch_index=runtime_config.welcome_template_index,
        verify_tls=runtime_config.retrieval_verify_tls,
        timeout_seconds=runtime_config.welcome_template_timeout_seconds,
        transport=transport,
    )
    welcome_localizer = WelcomeLocalizationService(
        WelcomeLocalizationConfig(
            enabled=runtime_config.welcome_localization_enabled,
            url=runtime_config.welcome_localization_url,
            model=runtime_config.welcome_localization_model,
            timeout_seconds=runtime_config.welcome_localization_timeout_seconds,
            max_tokens=runtime_config.welcome_localization_max_tokens,
            cache_key_prefix=runtime_config.welcome_localization_cache_key_prefix,
            cache_ttl_seconds=runtime_config.welcome_localization_cache_ttl_seconds,
            redis_socket_timeout_seconds=runtime_config.redis_socket_timeout_seconds,
            redis_cluster_nodes=runtime_config.redis_cluster_nodes,
            redis_password=runtime_config.redis_password,
            prewarm_locales=runtime_config.welcome_localization_prewarm_locales,
            prewarm_keyword=runtime_config.welcome_localization_prewarm_keyword,
        ),
        transport=transport,
    )

    async def enrich_with_pricing(result: RetrievalResult, user_text: str) -> str:
        """核价异常只降级，不影响正常知识回答。"""
        intent = parse_quote_intent(user_text)
        if not intent.requested:
            return ""
        quote = None
        if intent.travel_date and intent.person_count and result.quote_candidates:
            try:
                quote = await pricing_service.quote(result.quote_candidates[0], intent)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                log_error(
                    "行程实时核价失败，已降级为询价引导",
                    error_type=type(exc).__name__,
                )
        return pricing_context(intent, quote)

    summarizer = ConversationSummarizer(
        SummarizationConfig(
            enabled=runtime_config.summarization_enabled,
            url=runtime_config.summarization_url,
            model=runtime_config.summarization_model,
            timeout_seconds=runtime_config.summarization_timeout_seconds,
            max_summary_chars=runtime_config.summarization_max_summary_chars,
            max_context_tokens=runtime_config.summarization_max_context_tokens,
            recent_turns=runtime_config.summarization_recent_turns,
            redis_key_prefix=runtime_config.redis_key_prefix,
            redis_ttl_seconds=runtime_config.redis_ttl_seconds,
            redis_socket_timeout_seconds=runtime_config.redis_socket_timeout_seconds,
            redis_cluster_nodes=runtime_config.redis_cluster_nodes,
            redis_password=runtime_config.redis_password,
        ),
        transport=transport,
    )
    summary_tasks: set[asyncio.Task[None]] = set()

    async def update_summary(
        session_id: str,
        request_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        started = time.monotonic()
        log_info(
            f"开始更新会话摘要：session_id={session_id}，request_id={request_id}",
            console=True,
            request_id=request_id,
            session_id=session_id,
        )
        try:
            places = await retriever.extract_place_entities(user_text + "\n" + assistant_text)
            if places:
                summarizer.remember_places(session_id, places)
                log_info(
                    f"会话地点记忆已更新：session_id={session_id}，places={','.join(places)}",
                    request_id=request_id,
                    session_id=session_id,
                    place_count=len(places),
                )
            summary = await summarizer.update(session_id)
            if summary:
                log_info(
                    "会话摘要更新完成："
                    f"session_id={session_id}，chars={len(summary)}，"
                    f"elapsed={round((time.monotonic() - started) * 1000)} ms",
                    console=True,
                    request_id=request_id,
                    session_id=session_id,
                    summary_chars=len(summary),
                )
        except (httpx.HTTPError, ValueError, KeyError, redis.RedisError):
            log_error(
                "会话摘要更新失败，继续使用旧摘要",
                exc_info=True,
                request_id=request_id,
                session_id=session_id,
            )

    async def monitor_model() -> None:
        """每5秒输出模型、GPU和吞吐心跳, 首次就绪时执行一次短测速。"""
        previous_generation_tokens: float | None = None
        previous_prompt_tokens: float | None = None
        previous_time = time.monotonic()
        model_announced = False
        benchmark_attempted = False
        async with httpx.AsyncClient(timeout=2) as client:
            while True:
                gpu = await asyncio.to_thread(_gpu_summary)
                try:
                    models = await client.get(f"{runtime_config.backend_url}/v1/models")
                    models.raise_for_status()
                    model_data = models.json().get("data", [])
                    if model_data and not model_announced:
                        loaded = model_data[0]
                        log_info(
                            "27B模型加载成功："
                            f"model={loaded.get('id', runtime_config.internal_model)}，"
                            f"context={loaded.get('max_model_len', 'unknown')}，"
                            f"backend={runtime_config.backend_url}",
                            console=True,
                            model=loaded.get("id", runtime_config.internal_model),
                            max_model_len=loaded.get("max_model_len"),
                            backend=runtime_config.backend_url,
                        )
                        model_announced = True

                    if runtime_config.startup_benchmark_enabled and not benchmark_attempted:
                        benchmark_attempted = True
                        benchmark_started = time.monotonic()
                        benchmark = await client.post(
                            f"{runtime_config.backend_url}/v1/chat/completions",
                            json={
                                "model": runtime_config.internal_model,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "请只输出从1到20的数字序列，用空格分隔。",
                                    }
                                ],
                                "max_tokens": runtime_config.startup_benchmark_max_tokens,
                                "temperature": 0,
                            },
                            timeout=runtime_config.request_timeout_seconds,
                        )
                        benchmark.raise_for_status()
                        benchmark_elapsed = max(0.001, time.monotonic() - benchmark_started)
                        completion_tokens = int(
                            benchmark.json().get("usage", {}).get("completion_tokens", 0)
                        )
                        benchmark_rate = completion_tokens / benchmark_elapsed
                        log_info(
                            "27B启动测速完成："
                            f"tokens={completion_tokens}，elapsed={benchmark_elapsed:.2f}s，"
                            f"output={benchmark_rate:.1f} token/s",
                            console=True,
                            completion_tokens=completion_tokens,
                            elapsed_seconds=round(benchmark_elapsed, 3),
                            output_tokens_per_second=round(benchmark_rate, 2),
                        )

                    response = await client.get(f"{runtime_config.backend_url}/metrics")
                    response.raise_for_status()
                    metrics = _metric_values(response.text)
                    now = time.monotonic()
                    generation_tokens = metrics.get("vllm:generation_tokens_total", 0.0)
                    prompt_tokens = metrics.get("vllm:prompt_tokens_total", 0.0)
                    running = round(metrics.get("vllm:num_requests_running", 0.0))
                    waiting = round(metrics.get("vllm:num_requests_waiting", 0.0))
                    if (
                        previous_generation_tokens is not None
                        and previous_prompt_tokens is not None
                    ):
                        elapsed = max(0.001, now - previous_time)
                        output_rate = (
                            max(0.0, generation_tokens - previous_generation_tokens) / elapsed
                        )
                        input_rate = max(0.0, prompt_tokens - previous_prompt_tokens) / elapsed
                        log_info(
                            "27B心跳："
                            f"gpu={gpu.get('gpu')}，utilization="
                            f"{gpu.get('gpu_utilization_percent', 0)}%，"
                            f"memory={_memory_gb(gpu)}，"
                            f"temperature={gpu.get('temperature_celsius', 0)}°C，"
                            f"power={gpu.get('power_watts', 0)}/"
                            f"{gpu.get('power_limit_watts', 0)} W，"
                            f"input={input_rate:.1f} token/s，"
                            f"output={output_rate:.1f} token/s，"
                            f"running={running}，waiting={waiting}",
                            console=True,
                            connected=True,
                            **gpu,
                            input_tokens_per_second=round(input_rate, 2),
                            output_tokens_per_second=round(output_rate, 2),
                            running=running,
                            waiting=waiting,
                        )
                    previous_generation_tokens = generation_tokens
                    previous_prompt_tokens = prompt_tokens
                    previous_time = now
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    log_info(
                        "27B心跳异常：模型后端不可用，"
                        f"backend={runtime_config.backend_url}，"
                        f"gpu={gpu.get('gpu')}，utilization="
                        f"{gpu.get('gpu_utilization_percent', 0)}%，"
                        f"memory={_memory_gb(gpu)}，"
                        f"temperature={gpu.get('temperature_celsius', 0)}°C，"
                        f"power={gpu.get('power_watts', 0)}/"
                        f"{gpu.get('power_limit_watts', 0)} W，"
                        f"error={type(exc).__name__}",
                        console=True,
                        connected=False,
                        backend=runtime_config.backend_url,
                        error_type=type(exc).__name__,
                        **gpu,
                    )
                    model_announced = False
                    previous_generation_tokens = None
                    previous_prompt_tokens = None
                    previous_time = time.monotonic()
                await asyncio.sleep(5)

    async def prewarm_welcome_locales() -> None:
        if (
            not runtime_config.welcome_localization_enabled
            or not runtime_config.welcome_localization_prewarm_locales
        ):
            return
        try:
            template = await welcome_templates.retrieve(
                runtime_config.welcome_localization_prewarm_keyword
            )
            completed = await welcome_localizer.prewarm(template)
            log_info(
                "欢迎模板热门语言预热完成："
                f"template_id={template.template_id}，locales={','.join(completed) or 'none'}",
                template_id=template.template_id,
                locales=completed,
            )
        except (httpx.HTTPError, ValueError, LookupError, KeyError, redis.RedisError):
            log_error("欢迎模板热门语言预热失败，长尾语言继续按需生成", exc_info=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime_state.initialize()
        gpu = await asyncio.to_thread(_gpu_summary)
        log_info(
            "27B路由配置："
            f"listen={runtime_config.host}:{runtime_config.port}，"
            f"backend={runtime_config.backend_url}，"
            f"model={runtime_config.internal_model}，"
            f"concurrency={runtime_config.max_concurrent_workloads}，"
            f"admission={runtime_config.admission_max_active}",
            console=True,
            listen=f"{runtime_config.host}:{runtime_config.port}",
            backend=runtime_config.backend_url,
            model=runtime_config.internal_model,
            concurrency=runtime_config.max_concurrent_workloads,
            admission=runtime_config.admission_max_active,
        )
        log_info(
            "GPU状态："
            f"name={gpu.get('gpu')}，driver={gpu.get('driver', 'unknown')}，"
            f"memory={_memory_gb(gpu)}，"
            f"utilization={gpu.get('gpu_utilization_percent', 0)}%，"
            f"temperature={gpu.get('temperature_celsius', 0)}°C，"
            f"power={gpu.get('power_watts', 0)}/"
            f"{gpu.get('power_limit_watts', 0)} W",
            console=True,
            **gpu,
        )
        monitor = None if transport is not None else asyncio.create_task(monitor_model())
        welcome_prewarm = (
            None if transport is not None else asyncio.create_task(prewarm_welcome_locales())
        )
        try:
            yield
        finally:
            if welcome_prewarm is not None:
                welcome_prewarm.cancel()
                with suppress(asyncio.CancelledError):
                    await welcome_prewarm
            if monitor is not None:
                monitor.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor
            log_info("27B路由已停止", console=True)

    application = FastAPI(
        title="gta-ai inference router",
        description="Inference-first router for preemptible idle fine-tuning.",
        lifespan=lifespan,
    )

    async def backend_ready(client: httpx.AsyncClient, backend_url: str) -> bool:
        try:
            response = await client.get(f"{backend_url}/v1/models", timeout=1)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_for_backend(client: httpx.AsyncClient, backend_url: str) -> None:
        deadline = time.monotonic() + runtime_config.wake_timeout_seconds
        while time.monotonic() < deadline:
            if await backend_ready(client, backend_url):
                return
            await asyncio.sleep(runtime_config.backend_poll_seconds)
        raise HTTPException(
            status_code=503,
            detail="推理服务未能在训练抢占窗口内恢复",
            headers={"Retry-After": "5"},
        )

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"state": "ok"}

    @application.get("/_gta/runtime")
    async def runtime() -> dict[str, object]:
        state = await runtime_state.snapshot()
        state.update(await workload_gate.snapshot())
        state.update(await request_registry.snapshot())
        return state

    @application.get("/_gta/admission")
    async def admission_status() -> dict[str, object]:
        return await admission.snapshot()

    @application.post("/_gta/admission/leases")
    async def acquire_lease(request: LeaseAcquireRequest) -> dict[str, object]:
        try:
            return await admission.acquire(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc

    @application.post("/_gta/admission/leases/{lease_id}/heartbeat")
    async def heartbeat_lease(lease_id: str, request: LeaseHeartbeatRequest) -> dict[str, object]:
        try:
            return await admission.heartbeat(lease_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="lease not found") from exc

    @application.delete("/_gta/admission/leases/{lease_id}")
    async def release_lease(lease_id: str, owner: str) -> dict[str, bool]:
        if not owner:
            raise HTTPException(status_code=422, detail="owner is required")
        released = await admission.release(lease_id, owner)
        if not released:
            raise HTTPException(status_code=404, detail="lease not found")
        return {"released": True}

    @application.delete("/_gta/requests/{request_id}")
    async def cancel_request(request_id: str) -> dict[str, object]:
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise HTTPException(status_code=422, detail="invalid request id")
        cancelled = await request_registry.cancel(request_id)
        log_info("推理请求已取消", request_id=request_id, cancelled=cancelled)
        return {"request_id": request_id, "cancelled": cancelled}

    @application.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request) -> Response:
        started_at = time.monotonic()
        upstream_path = "/" + path
        is_workload = request.method in {"POST", "PUT", "PATCH"} and (
            upstream_path in WORKLOAD_PATHS or upstream_path.startswith("/v1/")
        )

        request_body = await request.body()
        request_id = request.headers.get("x-gta-request-id", "").strip()
        session_id = request.headers.get("x-gta-session-id", "").strip()
        locale_hint = request.headers.get("x-gta-locale", "").strip()
        country_hint = request.headers.get("x-gta-country", "").strip()
        if session_id and not SESSION_ID_PATTERN.fullmatch(session_id):
            raise HTTPException(status_code=422, detail="invalid X-GTA-Session-ID")
        if (
            runtime_config.welcome_template_enabled
            and request.method == "POST"
            and upstream_path == "/v1/chat/completions"
        ):
            try:
                welcome_request = json.loads(request_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                welcome_request = None
            welcome_messages = (
                welcome_request.get("messages") if isinstance(welcome_request, dict) else None
            )
            welcome_keyword_value = (
                welcome_request.get("keyword")
                if isinstance(welcome_request, dict) and "keyword" in welcome_request
                else None
            )
            welcome_request_candidate = isinstance(welcome_request, dict) and (
                welcome_messages is None or welcome_messages == []
            )
            keyword = ""
            if welcome_request_candidate and welcome_keyword_value is not None:
                if not isinstance(welcome_keyword_value, str):
                    raise HTTPException(status_code=422, detail="keyword must be a string")
                keyword = welcome_keyword_value.strip()[:500]
                if not keyword:
                    welcome_request_candidate = False
            if welcome_request_candidate and keyword:
                try:
                    template = await welcome_templates.retrieve(keyword)
                except (httpx.HTTPError, ValueError, LookupError, KeyError) as exc:
                    log_error(
                        "欢迎模板检索失败",
                        exc_info=True,
                        request_id=request_id,
                        keyword=keyword,
                    )
                    raise HTTPException(status_code=503, detail="欢迎内容暂时不可用") from exc
                target_locale = normalize_welcome_locale(locale_hint, country_hint)
                response_id = f"chatcmpl-welcome-{uuid.uuid4().hex}"
                model_name = str(welcome_request.get("model") or "green-travel-ai")
                if welcome_request.get("stream") is True:

                    async def welcome_stream() -> AsyncIterator[bytes]:
                        role_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                            ],
                        }
                        finish_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode()
                        yield (
                            b'data: {"type":"welcome_localization","status":"generating"}\n\n'
                        )
                        localized_template = template
                        try:
                            localized_template = await welcome_localizer.localize(
                                template,
                                locale_hint=locale_hint,
                                country_hint=country_hint,
                            )
                        except (httpx.HTTPError, ValueError, KeyError, redis.RedisError):
                            log_error(
                                "欢迎模板本地化失败，已降级为中文母版",
                                exc_info=True,
                                request_id=request_id,
                                locale=target_locale,
                                country=country_hint,
                            )
                        content = localized_template.content.strip()
                        if (
                            localized_template.title
                            and localized_template.title not in content[:200]
                        ):
                            content = f"## {localized_template.title}\n\n{content}".strip()
                        image_event = b""
                        if localized_template.images:
                            image_event = retrieval_image_event(
                                RetrievalResult(
                                    query=keyword,
                                    context="",
                                    hit_count=1,
                                    elapsed_ms=0,
                                    images=tuple(
                                        (image["title"], image["path"])
                                        for image in localized_template.images
                                    ),
                                )
                            )
                        log_info(
                            "返回ES欢迎模板："
                            f"template_id={localized_template.template_id}，"
                            f"version={localized_template.version}，"
                            f"keyword={keyword or '<default>'}，locale={target_locale}",
                            request_id=request_id,
                            template_id=localized_template.template_id,
                            template_version=localized_template.version,
                            keyword=keyword,
                            locale=target_locale,
                        )
                        if localized_template.ui_text:
                            ui_event = {
                                "type": "welcome_ui",
                                "ui_text": localized_template.ui_text,
                            }
                            yield (f"data: {json.dumps(ui_event, ensure_ascii=False)}\n\n").encode()
                        yield b'data: {"type":"review_section"}\n\n'
                        if image_event:
                            yield image_event
                        content_parts: list[str] = []
                        content_cursor = 0
                        image_marker = "[[IMAGE_GROUP_1]]"
                        while content_cursor < len(content):
                            if content.startswith(image_marker, content_cursor):
                                content_parts.append(image_marker)
                                content_cursor += len(image_marker)
                            else:
                                content_parts.append(content[content_cursor])
                                content_cursor += 1
                        for character in content_parts:
                            content_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": character},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield (
                                f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
                            ).encode()
                            if runtime_config.welcome_template_character_interval_ms:
                                await asyncio.sleep(
                                    runtime_config.welcome_template_character_interval_ms / 1000
                                )
                        if localized_template.suggested_questions:
                            questions_event = {
                                "type": "suggested_questions",
                                "questions": list(localized_template.suggested_questions),
                            }
                            yield (
                                f"data: {json.dumps(questions_event, ensure_ascii=False)}\n\n"
                            ).encode()
                        yield welcome_handoff_event(localized_template.ui_text)
                        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode()
                        yield b"data: [DONE]\n\n"

                    return StreamingResponse(
                        welcome_stream(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                try:
                    template = await welcome_localizer.localize(
                        template,
                        locale_hint=locale_hint,
                        country_hint=country_hint,
                    )
                except (httpx.HTTPError, ValueError, KeyError, redis.RedisError):
                    log_error(
                        "欢迎模板本地化失败，已降级为中文母版",
                        exc_info=True,
                        request_id=request_id,
                        locale=target_locale,
                        country=country_hint,
                    )
                content = template.content.strip()
                if template.title and template.title not in content[:200]:
                    content = f"## {template.title}\n\n{content}".strip()
                return Response(
                    content=json.dumps(
                        {
                            "id": response_id,
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": content},
                                    "finish_reason": "stop",
                                }
                            ],
                            "gta_welcome": {
                                "template_id": template.template_id,
                                "version": template.version,
                                "images": list(template.images),
                                "suggested_questions": list(template.suggested_questions),
                                "ui_text": template.ui_text,
                                "review_section": True,
                                "handoff_offer": welcome_handoff_payload(template.ui_text),
                            },
                        },
                        ensure_ascii=False,
                    ),
                    media_type="application/json",
                )
        try:
            current_query = _latest_user_text(request_body)
            memory_summary = summarizer.get(session_id, current_query=current_query)
            if memory_summary:
                log_info(
                    "装载有界会话记忆："
                    f"session_id={session_id}，estimated_tokens={estimate_tokens(memory_summary)}，"
                    f"limit={runtime_config.summarization_max_context_tokens}",
                    request_id=request_id,
                    session_id=session_id,
                    memory_tokens=estimate_tokens(memory_summary),
                    memory_token_limit=runtime_config.summarization_max_context_tokens,
                )
        except redis.RedisError:
            memory_summary = ""
            log_error(
                "读取会话记忆失败，本轮继续使用当前消息",
                exc_info=True,
                request_id=request_id,
                session_id=session_id,
            )
        original_chat_body = request_body
        identity_query = _asks_about_identity(current_query)
        verified_business_fact_required = requires_verified_business_fact(current_query)
        verified_business_fact_found = False
        contextual_query = contextual_retrieval_query(current_query, memory_summary)
        playful_query = is_playful_or_impossible_travel_query(current_query)
        if playful_query:
            # 玩笑不能被旧旅游摘要重新拉回检索或营销流程。
            memory_summary = ""
            contextual_query = ""
        # 图片只能由当前旅游问题或明确的旅游追问触发。
        # 不允许从模型回答中偶然出现的地名反向触发图片。
        image_response_allowed = bool(
            not playful_query
            and (conversation_may_need_retrieval(original_chat_body) or contextual_query)
        )
        if identity_query:
            # 身份或底层模型问题是明确话题切换，不让旅游摘要压过当前问题。
            memory_summary = ""
        history_message_limit = 1 if identity_query else runtime_config.max_history_messages
        history_char_limit = (
            min(1000, runtime_config.max_history_chars)
            if identity_query
            else runtime_config.max_history_chars
        )
        native_tool_enabled = False
        fast_retrieval_query = ""
        retrieval_images_event = b""
        initial_image_candidates: tuple[tuple[str, str], ...] = ()
        image_search_query = current_query
        image_search_query_normalized = bool(re.search(r"[\u4e00-\u9fff]", image_search_query))
        stream_requested = False
        contact_request_required = False
        handoff_offer_required = False
        handoff_offer_task: asyncio.Task[dict[str, str]] | None = None
        explicit_contact_request = False
        additional_system_prompt = ""
        cpu_simple_route = False
        upstream_backend_url = runtime_config.backend_url
        response_model_label = "27B"
        if is_workload and not request_id:
            request_id = uuid.uuid4().hex
        if request.method == "POST" and upstream_path == "/v1/chat/completions":
            try:
                stream_requested = json.loads(original_chat_body).get("stream") is True
                explicit_contact_request = query_explicitly_requests_contact(current_query)
                contact_request_required = stream_requested and explicit_contact_request
                handoff_offer_required = (
                    stream_requested and image_response_allowed and not explicit_contact_request
                )
                cpu_simple_route = bool(
                    runtime_config.simple_chat_enabled and simple_chat_query(original_chat_body)
                )
                additional_system_prompt = "\n\n".join(
                    part
                    for part in (
                        SIMPLE_CHAT_PROMPT if cpu_simple_route else "",
                        language_style_prompt(current_query, locale_hint, country_hint),
                        (
                            COMPANY_RECOMMENDATION_PROMPT
                            if is_company_recommendation_query(current_query)
                            else ""
                        ),
                        (VERIFIED_BUSINESS_FACT_PROMPT if verified_business_fact_required else ""),
                        (
                            CONFIRMATION_FOLLOWUP_PROMPT
                            if is_confirmation_followup(current_query) and memory_summary
                            else ""
                        ),
                        TRAVEL_SUPPORT_PROMPT if is_travel_support_query(current_query) else "",
                        PLAYFUL_QUERY_PROMPT if playful_query else "",
                        (
                            QUESTION_HISTORY_PROMPT
                            if is_user_question_history_query(current_query)
                            else ""
                        ),
                    )
                    if part
                )
                if cpu_simple_route:
                    upstream_backend_url = runtime_config.simple_chat_url
                    response_model_label = "4B"
                elif runtime_config.retrieval_enabled and not playful_query:
                    fast_retrieval_query = (
                        current_query
                        if verified_business_fact_required
                        else ("" if identity_query else direct_retrieval_query(original_chat_body))
                    ) or contextual_query
                request_body = inject_persona(
                    request_body,
                    internal_model=(
                        runtime_config.simple_chat_model
                        if cpu_simple_route
                        else runtime_config.internal_model
                    ),
                    standard_max_tokens=(
                        runtime_config.simple_chat_max_tokens
                        if cpu_simple_route
                        else runtime_config.standard_max_tokens
                    ),
                    persona_prompt_enabled=runtime_config.persona_prompt_enabled,
                    max_history_messages=history_message_limit,
                    max_history_chars=history_char_limit,
                    memory_summary=memory_summary,
                    include_priority=not cpu_simple_route,
                    additional_system_prompt=additional_system_prompt,
                    max_tokens_cap=(
                        runtime_config.simple_chat_max_tokens if cpu_simple_route else None
                    ),
                )
                if (
                    runtime_config.retrieval_enabled
                    and not fast_retrieval_query
                    and conversation_may_need_retrieval(original_chat_body)
                ):
                    request_body = inject_knowledge_tool(request_body)
                    native_tool_enabled = True
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        tracked: TrackedRequest | None = None
        if is_workload:
            if not REQUEST_ID_PATTERN.fullmatch(request_id):
                raise HTTPException(status_code=422, detail="invalid X-GTA-Request-ID")
            tracked = await request_registry.register(request_id)
            if tracked is None:
                log_info("拒绝重复的推理请求", request_id=request_id, path=upstream_path)
                raise HTTPException(
                    status_code=409,
                    detail="request id is already active",
                    headers={"Retry-After": "1"},
                )
            await runtime_state.begin()
            log_info(
                f"收到{response_model_label}推理请求：request_id={request_id}",
                console=True,
                request_id=request_id,
                path=upstream_path,
                input_bytes=len(request_body),
                inference_model=response_model_label,
            )

        inference_gate: int | None = None
        workload_gate_acquired = False
        admission_lease: dict[str, object] | None = None
        admission_heartbeat: asyncio.Task[None] | None = None
        admission_owner = "router:" + hashlib.sha256(request_id.encode()).hexdigest()[:32]
        timeout = httpx.Timeout(runtime_config.request_timeout_seconds, connect=2)
        client = httpx.AsyncClient(timeout=timeout, transport=transport)
        if handoff_offer_required:
            handoff_offer_task = asyncio.create_task(
                generate_handoff_offer(
                    current_query,
                    locale_hint=locale_hint,
                    country_hint=country_hint,
                    url=runtime_config.welcome_localization_url,
                    model=runtime_config.welcome_localization_model,
                    timeout_seconds=min(
                        runtime_config.welcome_localization_timeout_seconds,
                        runtime_config.request_timeout_seconds,
                    ),
                    transport=transport,
                )
            )
        disconnect_watcher = (
            asyncio.create_task(_wait_for_request_cancel(request, tracked.cancel_event))
            if is_workload
            else None
        )
        cleaned = False
        cleanup_lock = asyncio.Lock()

        async def cleanup() -> None:
            nonlocal cleaned, inference_gate, admission_lease, admission_heartbeat
            nonlocal handoff_offer_task
            async with cleanup_lock:
                if cleaned:
                    return
                cleaned = True
                if admission_heartbeat is not None:
                    admission_heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await admission_heartbeat
                    admission_heartbeat = None
                if disconnect_watcher is not None:
                    disconnect_watcher.cancel()
                if handoff_offer_task is not None and not handoff_offer_task.done():
                    handoff_offer_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await handoff_offer_task
                    handoff_offer_task = None
                await client.aclose()
                _release_inference_gate(inference_gate)
                inference_gate = None
                if admission_lease is not None:
                    await admission.release(str(admission_lease["lease_id"]), admission_owner)
                    admission_lease = None
                if workload_gate_acquired:
                    await workload_gate.release()
                if is_workload:
                    await runtime_state.end()
                    await request_registry.unregister(request_id)

        try:
            if fast_retrieval_query:
                try:
                    normalized_retrieval_query = await normalize_retrieval_query(
                        fast_retrieval_query,
                        url=runtime_config.summarization_url,
                        model=runtime_config.summarization_model,
                        timeout_seconds=runtime_config.summarization_timeout_seconds,
                        transport=transport,
                    )
                    image_search_query = normalized_retrieval_query
                    image_search_query_normalized = True
                    async with asyncio.timeout(runtime_config.retrieval_timeout_seconds):
                        retrieval = await retriever.retrieve(normalized_retrieval_query)
                    initial_image_candidates = retrieval.images
                    if verified_business_fact_required:
                        verified_business_fact_found = business_fact_evidence_found(
                            current_query,
                            retrieval.context,
                        )
                    quote_context = await enrich_with_pricing(
                        retrieval,
                        contextual_query or current_query,
                    )
                    request_body = inject_persona(
                        original_chat_body,
                        internal_model=runtime_config.internal_model,
                        standard_max_tokens=runtime_config.standard_max_tokens,
                        persona_prompt_enabled=runtime_config.persona_prompt_enabled,
                        max_history_messages=history_message_limit,
                        max_history_chars=history_char_limit,
                        memory_summary=memory_summary,
                        additional_system_prompt=additional_system_prompt,
                        retrieval_context="\n\n".join(
                            part
                            for part in (
                                retrieval_context(
                                    retrieval,
                                    use_image_marker=stream_requested,
                                    include_images=False,
                                ),
                                quote_context,
                            )
                            if part
                        ),
                    )
                    log_info(
                        "明确旅游意图直接调用ES知识检索："
                        f"query={fast_retrieval_query}，"
                        f"normalized_query={normalized_retrieval_query}，"
                        f"hits={retrieval.hit_count}，"
                        f"images={len(retrieval.images)}，elapsed={retrieval.elapsed_ms} ms",
                        request_id=request_id,
                        retrieval_hits=retrieval.hit_count,
                        retrieval_images=len(retrieval.images),
                        retrieval_elapsed_ms=retrieval.elapsed_ms,
                    )
                except (TimeoutError, httpx.HTTPError, ValueError, TypeError) as exc:
                    request_body = inject_knowledge_tool(request_body)
                    native_tool_enabled = True
                    log_error(
                        "直接知识检索失败，降级为模型工具判断",
                        request_id=request_id,
                        error_type=type(exc).__name__,
                    )
            if is_workload and not cpu_simple_route:
                priority = workload_gate.parse_priority(request.headers.get("x-gta-priority"))
                await request_registry.set_state(request_id, "WAITING_WORKLOAD")
                await _await_cancelable(workload_gate.acquire(priority), tracked.cancel_event)
                workload_gate_acquired = True
                await request_registry.set_state(request_id, "WAITING_ADMISSION")
                admission_started = time.monotonic()
                workload_class = request.headers.get("x-gta-workload-class") or (
                    workload_gate.workload_class(priority)
                )
                try:
                    admission_lease = await _await_cancelable(
                        admission.acquire(
                            LeaseAcquireRequest(
                                owner=admission_owner,
                                workload_class=workload_class,
                                requested_memory_mb=0,
                                ttl_seconds=15,
                                wait_seconds=runtime_config.request_timeout_seconds,
                            )
                        ),
                        tracked.cancel_event,
                    )
                except (TimeoutError, ValueError) as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"GPU admission rejected inference: {exc}",
                        headers={"Retry-After": "1"},
                    ) from exc
                admission_wait_ms = round((time.monotonic() - admission_started) * 1000)
                admission_message = (
                    "GPU通道已分配"
                    if admission_wait_ms <= 200
                    else f"GPU通道排队完成：wait={admission_wait_ms} ms"
                )
                log_info(
                    admission_message,
                    console=admission_wait_ms > 200,
                    admission_wait_ms=admission_wait_ms,
                )

                async def keep_admission_alive() -> None:
                    while admission_lease is not None:
                        await asyncio.sleep(5)
                        try:
                            await admission.heartbeat(
                                str(admission_lease["lease_id"]),
                                LeaseHeartbeatRequest(owner=admission_owner, ttl_seconds=15),
                            )
                        except KeyError:
                            log_error("GPU租约意外失效，当前推理将被取消")
                            tracked.cancel_event.set()
                            return
                        except redis.RedisError:
                            log_error("GPU租约续期失败，将在下个心跳重试", exc_info=True)

                admission_heartbeat = asyncio.create_task(keep_admission_alive())
                # 共享锁覆盖整个推理响应。训练持有独占锁时, 新请求先登记为 active,
                # 然后等待调度器抢占训练并释放 GPU; 绝不会撞上正在退出的 vLLM。
                await request_registry.set_state(request_id, "WAITING_GPU_LOCK")
                inference_gate = await _await_cancelable(
                    _acquire_inference_gate(runtime_config.gpu_lock_path),
                    tracked.cancel_event,
                )
                await request_registry.set_state(request_id, "WAITING_BACKEND")
                await _await_cancelable(
                    wait_for_backend(client, upstream_backend_url), tracked.cancel_event
                )
            upstream_request = client.build_request(
                request.method,
                f"{upstream_backend_url}{upstream_path}",
                params=request.query_params,
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
                },
                # 27B Router统一注入品牌身份，调用者无需额外传入system消息。
                content=request_body,
            )
            if is_workload:
                await request_registry.set_state(request_id, "INFERENCE")
                upstream = await _await_cancelable(
                    client.send(upstream_request, stream=True),
                    tracked.cancel_event,
                )
            else:
                upstream = await client.send(upstream_request, stream=True)
        except RequestCancelled as exc:
            await cleanup()
            log_info(
                "推理请求已中断",
                request_id=request_id,
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        except asyncio.CancelledError:
            await cleanup()
            raise
        except HTTPException as exc:
            await cleanup()
            log_error(
                "推理请求失败",
                exc_info=True,
                request_id=request_id,
                status=exc.status_code,
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            raise
        except httpx.HTTPError as exc:
            await cleanup()
            log_error(
                "无法连接27B模型",
                exc_info=True,
                request_id=request_id,
                error_type=type(exc).__name__,
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            raise HTTPException(
                status_code=502, detail=f"推理后端连接失败: {type(exc).__name__}"
            ) from exc
        except BaseException:
            await cleanup()
            log_error(
                "27B路由发生未处理异常",
                exc_info=True,
                request_id=request_id,
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )
            raise

        async def stream() -> AsyncIterator[bytes]:
            nonlocal upstream, request_body, retrieval_images_event
            nonlocal initial_image_candidates, image_search_query, image_search_query_normalized
            completed = False
            first_response_logged = False
            response_capture = bytearray()
            identity_filter = _IdentityPrefixFilter(
                content_type=upstream.headers.get("content-type", ""),
                allow_identity=_asks_about_identity(_latest_user_text(original_chat_body)),
            )
            generated_answer = ""
            answer_image_search_attempts = 0
            next_answer_image_search_chars = 180
            answer_images_inserted = False
            collected_image_candidates: list[tuple[str, str]] = []
            contact_request_inserted = False
            handoff_offer_inserted = False

            async def process_chunk(chunk: bytes, *, final: bool = False) -> bytes:
                nonlocal generated_answer
                nonlocal answer_image_search_attempts
                nonlocal next_answer_image_search_chars
                nonlocal answer_images_inserted
                nonlocal collected_image_candidates
                nonlocal retrieval_images_event
                nonlocal initial_image_candidates
                nonlocal image_search_query
                nonlocal image_search_query_normalized
                nonlocal contact_request_inserted
                nonlocal handoff_offer_inserted
                filtered = identity_filter.feed(chunk, final=final)
                filtered = _clean_decorative_symbols(
                    filtered,
                    upstream.headers.get("content-type", ""),
                )
                done_event = b""
                done_position = filtered.find(b"data: [DONE]")
                if done_position >= 0:
                    done_end = filtered.find(b"\n\n", done_position)
                    done_end = len(filtered) if done_end < 0 else done_end + 2
                    done_event = filtered[done_position:done_end]
                    filtered = filtered[:done_position] + filtered[done_end:]
                    final = True
                fragment = _response_text(filtered)
                if fragment:
                    generated_answer += fragment
                should_search_images = (
                    stream_requested
                    and image_response_allowed
                    and not answer_images_inserted
                    and not identity_query
                    and answer_image_search_attempts < 4
                    and (final or len(generated_answer) >= next_answer_image_search_chars)
                    and (final or any(mark in fragment for mark in "。！？!?\n"))
                    and not re.search(
                        r"(?:^|\n)\s*(?:\d{1,2}[.)、]|[-*])\s*$",
                        generated_answer,
                    )
                )
                if should_search_images:
                    answer_image_search_attempts += 1
                    next_answer_image_search_chars = len(generated_answer) + 180
                    try:
                        if not image_search_query_normalized:
                            image_search_query = await normalize_retrieval_query(
                                image_search_query,
                                url=runtime_config.summarization_url,
                                model=runtime_config.summarization_model,
                                timeout_seconds=runtime_config.summarization_timeout_seconds,
                                transport=transport,
                            )
                            image_search_query_normalized = True
                        async with asyncio.timeout(runtime_config.retrieval_timeout_seconds):
                            images = await retriever.retrieve_images_from_answer(
                                generated_answer,
                                image_search_query,
                            )
                    except (TimeoutError, httpx.HTTPError, ValueError, TypeError):
                        images = ()
                        log_error(
                            "根据AI回答检索景点图片失败，本轮继续输出正文",
                            request_id=request_id,
                            exc_info=True,
                        )
                    known_paths = {path for _, path in collected_image_candidates}
                    for candidate in (*images, *initial_image_candidates):
                        if candidate[1] not in known_paths:
                            collected_image_candidates.append(candidate)
                            known_paths.add(candidate[1])
                    images = tuple(collected_image_candidates[:3])
                    if len(images) >= 3:
                        answer_images_inserted = True
                        retrieval_images_event = retrieval_image_event(
                            RetrievalResult(
                                query="generated-answer",
                                context="",
                                hit_count=len(images),
                                elapsed_ms=0,
                                images=images,
                            )
                        )
                        filtered += retrieval_images_event + _image_marker_sse_event()
                        log_info(
                            "根据AI回答中的地域实体插入三图组："
                            f"request_id={request_id}，"
                            f"images={','.join(title for title, _ in images[:3])}",
                            request_id=request_id,
                            retrieval_images=len(images[:3]),
                            image_source="generated_answer",
                        )
                    elif final:
                        log_info(
                            "相关横图不足三张，本轮不插入图片组："
                            f"request_id={request_id}，images={len(images)}",
                            request_id=request_id,
                            retrieval_images=len(images),
                            image_source="generated_answer",
                        )
                if final and contact_request_required and not contact_request_inserted:
                    contact_request_inserted = True
                    filtered += contact_request_event()
                    log_info(
                        f"用户主动询问联系方式：request_id={request_id}，session_id={session_id}",
                        request_id=request_id,
                        session_id=session_id,
                        conversion_event="contact_request",
                    )
                elif final and handoff_offer_required and not handoff_offer_inserted:
                    handoff_offer_inserted = True
                    try:
                        if handoff_offer_task is None:
                            raise ValueError("人工服务文案任务未创建")
                        handoff_copy = await handoff_offer_task
                        filtered += handoff_offer_event(handoff_copy)
                        log_info(
                            "27B已生成本地化转人工事件："
                            f"request_id={request_id}，session_id={session_id}",
                            request_id=request_id,
                            session_id=session_id,
                            conversion_event="handoff_offer",
                        )
                    except (httpx.HTTPError, ValueError, KeyError, asyncio.CancelledError):
                        log_error(
                            "27B生成本地化转人工文案失败，本轮不输出硬编码文案",
                            request_id=request_id,
                            session_id=session_id,
                            exc_info=True,
                        )
                if is_workload and len(response_capture) < 2 * 1024 * 1024:
                    remaining = 2 * 1024 * 1024 - len(response_capture)
                    response_capture.extend(filtered[:remaining])
                return filtered + done_event

            try:
                if retrieval_images_event:
                    yield retrieval_images_event
                iterator = upstream.aiter_raw().__aiter__()
                if native_tool_enabled and upstream.headers.get("content-type", "").startswith(
                    "text/event-stream"
                ):
                    probe = bytearray()
                    tool_seen = False
                    while True:
                        try:
                            if is_workload:
                                chunk = await _await_cancelable(
                                    iterator.__anext__(), tracked.cancel_event
                                )
                            else:
                                chunk = await iterator.__anext__()
                        except StopAsyncIteration:
                            break
                        probe.extend(chunk)
                        detected_tool, content_seen, _ = stream_tool_state(bytes(probe))
                        tool_seen = tool_seen or detected_tool
                        if content_seen and not tool_seen:
                            # 普通回答直接进入实时透传，不再发起第二次27B请求。
                            if is_workload and not first_response_logged:
                                first_response_logged = True
                                first_response_ms = round((time.monotonic() - started_at) * 1000)
                                log_info(
                                    f"{response_model_label}首次回复："
                                    f"request_id={request_id}，latency={first_response_ms} ms",
                                    console=True,
                                    request_id=request_id,
                                    first_response_ms=first_response_ms,
                                )
                            filtered = await process_chunk(bytes(probe))
                            if filtered:
                                yield filtered
                            probe.clear()
                            break
                    if tool_seen:
                        _, _, retrieval_query = stream_tool_state(bytes(probe))
                        await upstream.aclose()
                        context = ""
                        try:
                            if not retrieval_query:
                                raise ValueError("27B工具调用缺少query参数")
                            async with asyncio.timeout(runtime_config.retrieval_timeout_seconds):
                                retrieval = await retriever.retrieve(retrieval_query)
                            initial_image_candidates = retrieval.images
                            image_search_query = retrieval_query
                            image_search_query_normalized = bool(
                                re.search(r"[\u4e00-\u9fff]", retrieval_query)
                            )
                            context = retrieval_context(
                                retrieval,
                                use_image_marker=stream_requested,
                                include_images=False,
                            )
                            quote_context = await enrich_with_pricing(
                                retrieval,
                                contextual_query or current_query,
                            )
                            if quote_context:
                                context = "\n\n".join((context, quote_context))
                            log_info(
                                "27B实时调用ES知识检索工具："
                                f"query={retrieval_query}，hits={retrieval.hit_count}，"
                                f"images={len(retrieval.images)}，"
                                f"elapsed={retrieval.elapsed_ms} ms",
                                request_id=request_id,
                                retrieval_hits=retrieval.hit_count,
                                retrieval_images=len(retrieval.images),
                                retrieval_elapsed_ms=retrieval.elapsed_ms,
                            )
                        except (TimeoutError, httpx.HTTPError, ValueError, TypeError) as exc:
                            log_error(
                                "27B实时知识检索工具执行失败，已降级为普通回答",
                                request_id=request_id,
                                error_type=type(exc).__name__,
                            )
                        request_body = inject_persona(
                            original_chat_body,
                            internal_model=runtime_config.internal_model,
                            standard_max_tokens=runtime_config.standard_max_tokens,
                            persona_prompt_enabled=runtime_config.persona_prompt_enabled,
                            max_history_messages=history_message_limit,
                            max_history_chars=history_char_limit,
                            memory_summary=memory_summary,
                            additional_system_prompt=additional_system_prompt,
                            retrieval_context=context,
                        )
                        followup_request = client.build_request(
                            request.method,
                            f"{runtime_config.backend_url}{upstream_path}",
                            params=request.query_params,
                            headers={
                                key: value
                                for key, value in request.headers.items()
                                if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
                            },
                            content=request_body,
                        )
                        upstream = await _await_cancelable(
                            client.send(followup_request, stream=True),
                            tracked.cancel_event,
                        )
                        iterator = upstream.aiter_raw().__aiter__()
                        if retrieval_images_event:
                            yield retrieval_images_event
                    elif probe:
                        # 没有正文也没有工具调用时，保持后端原始响应。
                        filtered = await process_chunk(bytes(probe))
                        if filtered:
                            yield filtered
                        probe.clear()
                while True:
                    try:
                        if is_workload:
                            chunk = await _await_cancelable(
                                iterator.__anext__(), tracked.cancel_event
                            )
                        else:
                            chunk = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    if is_workload and not first_response_logged and chunk:
                        first_response_logged = True
                        first_response_ms = round((time.monotonic() - started_at) * 1000)
                        log_info(
                            f"{response_model_label}首次回复："
                            f"request_id={request_id}，latency={first_response_ms} ms",
                            console=True,
                            request_id=request_id,
                            first_response_ms=first_response_ms,
                        )
                    filtered = await process_chunk(chunk)
                    if filtered:
                        yield filtered
                tail = await process_chunk(b"", final=True)
                if tail:
                    yield tail
                completed = True
            except RequestCancelled:
                log_info(
                    "推理响应已中断",
                    request_id=request_id,
                    duration_ms=round((time.monotonic() - started_at) * 1000),
                )
                # 响应头已经发送后，客户端断开或主动停止属于正常流结束。
                # 此处不能再抛异常，否则Starlette会记录无意义的ASGI堆栈。
                return
            except Exception:
                log_error(
                    "读取27B模型响应失败",
                    exc_info=True,
                    request_id=request_id,
                    duration_ms=round((time.monotonic() - started_at) * 1000),
                )
                raise
            finally:
                await upstream.aclose()
                await cleanup()
                if is_workload and completed:
                    duration_seconds = time.monotonic() - started_at
                    usage = _response_usage(bytes(response_capture))
                    output_tokens_per_second = (
                        usage["output_tokens"] / duration_seconds if duration_seconds > 0 else 0.0
                    )
                    log_info(
                        f"{response_model_label}回复完成：request_id={request_id}，"
                        f"input={usage['input_tokens']} tokens，"
                        f"output={usage['output_tokens']} tokens，"
                        f"total={usage['total_tokens']} tokens，"
                        f"duration={duration_seconds:.2f}s，"
                        f"output_speed={output_tokens_per_second:.2f} token/s",
                        console=True,
                        request_id=request_id,
                        status=upstream.status_code,
                        duration_ms=round(duration_seconds * 1000),
                        output_tokens_per_second=round(output_tokens_per_second, 2),
                        **usage,
                        inference_model=response_model_label,
                    )
                    if session_id and runtime_config.summarization_enabled:
                        user_text = _latest_user_text(original_chat_body)
                        assistant_text = (
                            _response_text(bytes(response_capture))
                            .replace("[[IMAGE_GROUP_1]]", "")
                        )
                        if user_text and assistant_text:
                            memory_claim_trusted = (
                                not verified_business_fact_required or verified_business_fact_found
                            )
                            if not memory_claim_trusted:
                                log_info(
                                    "企业事实缺少知识库证据，本轮回答不写入会话记忆",
                                    request_id=request_id,
                                    session_id=session_id,
                                    memory_write_skipped=True,
                                )
                            else:
                                try:
                                    summarizer.record_turn(
                                        session_id,
                                        user_text,
                                        assistant_text,
                                    )
                                except redis.RedisError:
                                    log_error(
                                        "保存最近会话消息失败，不影响本轮回复",
                                        exc_info=True,
                                        request_id=request_id,
                                        session_id=session_id,
                                    )
                                else:
                                    task = asyncio.create_task(
                                        update_summary(
                                            session_id,
                                            request_id,
                                            user_text,
                                            assistant_text,
                                        )
                                    )
                                    summary_tasks.add(task)
                                    task.add_done_callback(summary_tasks.discard)

        response_headers = _forward_headers(upstream.headers)
        if is_workload:
            response_headers["x-gta-request-id"] = request_id
        if session_id:
            response_headers["x-gta-session-id"] = session_id
        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return application


app = create_router_app()


def main() -> None:
    config = RouterConfig.from_environment()
    configure_logging("gta-ai-router")
    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="warning",
            access_log=False,
            log_config=None,
        )
    except BaseException:
        log_error("27B路由启动失败", exc_info=True)
        raise
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
