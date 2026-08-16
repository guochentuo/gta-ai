from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import httpx

from gta_ai.config import Settings
from gta_ai.schemas.material_search import MaterialSearchRequest, MaterialSearchResult


class MaterialSearchError(RuntimeError):
    pass


class MaterialSearchPort(Protocol):
    async def search(self, request: MaterialSearchRequest) -> MaterialSearchResult: ...


def _text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result else None


def _integer(value: object) -> int | None:
    try:
        return int(value) if value is not None and str(value) else None
    except (TypeError, ValueError):
        return None


def _readable_size(value: int | None) -> str | None:
    if value is None:
        return None
    number = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if number < 1024 or unit == "TB":
            return f"{number:.0f} {unit}" if unit == "B" else f"{number:.2f} {unit}"
        number /= 1024
    return None


def _local_datetime(value: object) -> str | None:
    text = _text(value)
    if text is None:
        return None
    return text[:19].replace("T", " ")


class ElasticsearchMaterialSearch:
    """Read material projections while hiding ES details from Java."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        embedding_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._embedding_transport = embedding_transport

    def _auth(self) -> httpx.BasicAuth | None:
        if not self._settings.media_audit_elasticsearch_username:
            return None
        password = self._settings.media_audit_elasticsearch_password
        return httpx.BasicAuth(
            self._settings.media_audit_elasticsearch_username,
            "" if password is None else password.get_secret_value(),
        )

    async def search(self, request: MaterialSearchRequest) -> MaterialSearchResult:
        body = await self._request_body(request)
        base = str(self._settings.media_audit_elasticsearch_url).rstrip("/")
        url = f"{base}/{self._settings.media_audit_material_alias}/_search"
        timeout = httpx.Timeout(self._settings.material_search_timeout_seconds, connect=3)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            auth=self._auth(),
            verify=self._settings.media_audit_elasticsearch_verify_tls,
        ) as client:
            response = await client.post(url, json=body)
        try:
            response.raise_for_status()
            raw = response.json()
            hits_node = raw["hits"]
            raw_hits = hits_node["hits"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise MaterialSearchError("failed to query material projection") from exc
        total_node = hits_node.get("total", 0)
        total = total_node.get("value", 0) if isinstance(total_node, dict) else total_node
        result: list[dict[str, Any]] = []
        for hit in raw_hits:
            source = hit.get("_source") if isinstance(hit, dict) else None
            if isinstance(source, Mapping):
                result.append(self._to_wire_item(source))
        return MaterialSearchResult(total=max(0, int(total)), list=result)

    async def _request_body(self, request: MaterialSearchRequest) -> dict[str, Any]:
        constraints = self._constraints(request)
        keyword = (request.keyword or "").strip()
        lexical = self._lexical_query(keyword, constraints) if keyword else None
        body: dict[str, Any] = {
            "from": (request.page_num - 1) * request.page_size,
            "size": request.page_size,
            "track_total_hits": True,
            "query": lexical or constraints,
            "sort": [],
        }
        if keyword:
            body["sort"].append({"_score": {"order": "desc"}})
            vector = await self._embed(keyword)
            if vector and lexical is not None:
                body["knn"] = {
                    "field": "visual_text_embedding.vector",
                    "query_vector": vector,
                    "k": max(100, request.page_size),
                    "num_candidates": 500,
                    "boost": 1.0,
                    # Accuracy first: vector scores only rerank lexical candidates.
                    "filter": lexical,
                }
        body["sort"].extend(
            [
                {"ranking.sort_index": {"order": "desc", "missing": "_last"}},
                {"created_at": {"order": "desc", "missing": "_last"}},
            ]
        )
        return body

    def _constraints(self, request: MaterialSearchRequest) -> dict[str, Any]:
        filters: list[dict[str, Any]] = [
            {"term": {"document_kind": "asset"}},
            {"term": {"deletion_state": "ACTIVE"}},
        ]
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []
        if request.type:
            media_type = "video" if request.type in {"02", "video"} else "image"
            filters.append({"term": {"media_type": media_type}})
        if request.user_id is not None:
            filters.append({"term": {"created_by": str(request.user_id)}})
        if request.orientation:
            filters.append({"term": {"orientation": request.orientation}})
        if request.biz_id:
            filters.append({"term": {"status": "PUBLISHED"}})
            must_not.append(
                {
                    "nested": {
                        "path": "business_refs",
                        "query": {"term": {"business_refs.biz_id": request.biz_id}},
                    }
                }
            )
        elif request.status:
            filters.append({"term": {"status": request.status}})
        if request.is_reject:
            filters.append(
                {"term": {"is_rejected": request.is_reject.lower() in {"1", "true"}}}
            )
        if request.use_type:
            values = [item.strip() for item in request.use_type.split(",") if item.strip()]
            if values:
                filters.append({"terms": {"use_types": values}})
        score_range: dict[str, int] = {}
        if request.min_score is not None:
            score_range["gte"] = request.min_score
        if request.max_score is not None:
            score_range["lte"] = request.max_score
        if score_range:
            filters.append({"range": {"review_score": score_range}})
        date_range: dict[str, str] = {}
        if request.start_date:
            date_range["gte"] = request.start_date[:10] + "T00:00:00"
        if request.end_date:
            date_range["lte"] = request.end_date[:10] + "T23:59:59"
        if date_range:
            filters.append({"range": {"created_at": date_range}})
        if request.remark:
            must.append({"match_phrase": {"remark": request.remark}})
        if request.original_author:
            must.append(
                {"match_phrase": {"source.original_author": request.original_author}}
            )
        return {"bool": {"filter": filters, "must": must, "must_not": must_not}}

    @staticmethod
    def _lexical_query(keyword: str, constraints: dict[str, Any]) -> dict[str, Any]:
        fields = [
            "origin_name^12",
            "title^10",
            "file_name^7",
            "remark^5",
            "ai.caption^5",
            "ai.ocr_text^3",
            "ai.transcript^4",
            "ai.visual_description^6",
            "video_understanding.summary^6",
            "video_understanding.main_topic^7",
            "video_understanding.places^8",
            "visual_text^6",
            "ocr_text^3",
            "transcript^4",
        ]
        keyword_query = {
            "bool": {
                "should": [
                    {"term": {"origin_name.exact": {"value": keyword, "boost": 20}}},
                    {"multi_match": {"query": keyword, "fields": fields, "type": "phrase"}},
                    {
                        "nested": {
                            "path": "business_refs",
                            "query": {
                                "multi_match": {
                                    "query": keyword,
                                    "fields": [
                                        "business_refs.description^4",
                                        "business_refs.alt^4",
                                    ],
                                    "type": "phrase",
                                }
                            },
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
        return {"bool": {"filter": [constraints], "must": [keyword_query]}}

    async def _embed(self, keyword: str) -> list[float]:
        if not self._settings.material_search_embedding_enabled:
            return []
        timeout = httpx.Timeout(self._settings.material_search_timeout_seconds, connect=3)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, transport=self._embedding_transport
            ) as client:
                response = await client.post(
                    str(self._settings.material_search_embedding_url),
                    json={
                        "model": self._settings.material_search_embedding_model,
                        "input": [keyword],
                    },
                )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
            if (
                not isinstance(vector, list)
                or len(vector) != self._settings.material_search_dimensions
            ):
                return []
            return [float(value) for value in vector]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return []

    @staticmethod
    def _to_wire_item(source: Mapping[str, Any]) -> dict[str, Any]:
        covers = source.get("covers") if isinstance(source.get("covers"), Mapping) else {}
        playback = source.get("playback") if isinstance(source.get("playback"), Mapping) else {}
        metrics = source.get("metrics") if isinstance(source.get("metrics"), Mapping) else {}
        origin = source.get("source") if isinstance(source.get("source"), Mapping) else {}
        size = _integer(source.get("size_bytes"))
        quote_count = _integer(metrics.get("quote_count")) or 0
        deletion_state = _text(source.get("deletion_state")) or "ACTIVE"
        master_url = _text(playback.get("master_url"))
        use_types = source.get("use_types", [])
        if not isinstance(use_types, list):
            use_types = []
        return {
            "id": _text(source.get("asset_id")),
            "type": "02" if source.get("media_type") == "video" else "01",
            "useType": ",".join(str(v) for v in use_types),
            "originalUrl": master_url or _text(playback.get("original_url")),
            "coverUrl": _text(covers.get("default_url")),
            "originName": _text(source.get("origin_name")),
            "fileName": _text(source.get("file_name")),
            "format": (_text(source.get("format")) or "").upper() or None,
            "contentType": _text(source.get("content_type")),
            "width": _integer(source.get("width")),
            "height": _integer(source.get("height")),
            "orientation": _text(source.get("orientation")),
            "duration": str((_integer(source.get("duration_ms")) or 0) // 1000),
            "size": size,
            "sizeTxt": _readable_size(size),
            "status": _text(source.get("status")),
            "isReject": "1" if source.get("is_rejected") is True else "0",
            "score": _text(source.get("review_score")),
            "processingState": _text(source.get("processing_state")),
            "auditState": _text(source.get("audit_state")),
            "auditAttemptCount": _integer(source.get("audit_attempt_count")),
            "deletionState": deletion_state,
            "currentAuditRequestId": _text(source.get("current_audit_request_id")),
            "canReturn": quote_count == 0 and deletion_state == "ACTIVE",
            "dashUrl": _text(playback.get("dash_url")),
            "createdAt": _local_datetime(source.get("created_at")),
            "updatedAt": _local_datetime(source.get("updated_at")),
            "quoteCount": quote_count,
            "createBy": _integer(source.get("created_by")),
            "sourceName": _text(origin.get("name")),
            "sourceUrl": _text(origin.get("page_url")),
            "originalAuthor": _text(origin.get("original_author")),
            "originalLicense": _text(origin.get("license")),
        }
