from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .services.pricing_service import QuoteCandidate


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = False
    embedding_url: str = "http://192.168.80.7:7101/v1/embeddings"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    elasticsearch_url: str = "https://192.168.80.130:9200"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    elasticsearch_indices: tuple[str, ...] = (
        "gta_article_admin",
        "gta_trip_admin",
        "gta_video_admin",
        "gta_province_admin",
        "gta_city_admin",
        "gta_scenic_admin",
    )
    verify_tls: bool = False
    timeout_seconds: float = 1.5
    top_k: int = 12
    context_top_k: int = 3
    num_candidates: int = 100
    max_query_chars: int = 1000
    max_context_chars: int = 2400
    max_document_chars: int = 600
    min_score: float = 0.72
    image_min_score: float = 0.78


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    context: str
    hit_count: int
    elapsed_ms: int
    images: tuple[tuple[str, str], ...] = ()
    quote_candidates: tuple[QuoteCandidate, ...] = ()


class ElasticsearchRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    async def retrieve(self, query: str) -> RetrievalResult:
        started = time.monotonic()
        if not self._config.enabled or not query:
            return RetrievalResult(query=query, context="", hit_count=0, elapsed_ms=0)

        timeout = httpx.Timeout(self._config.timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            verify=self._config.verify_tls,
        ) as client:
            embedding_response = await client.post(
                self._config.embedding_url,
                json={
                    "model": self._config.embedding_model,
                    "input": [query],
                    "encoding_format": "float",
                },
            )
            embedding_response.raise_for_status()
            embedding = _read_embedding(embedding_response.json())

            indices = ",".join(self._config.elasticsearch_indices)
            search_response = await client.post(
                f"{self._config.elasticsearch_url}/{indices}/_search",
                auth=(
                    self._config.elasticsearch_username,
                    self._config.elasticsearch_password,
                ),
                json={
                    "size": self._config.top_k,
                    "min_score": self._config.min_score,
                    "knn": [
                        {
                            "field": "business_text_vector",
                            "query_vector": embedding,
                            "k": self._config.top_k,
                            "num_candidates": self._config.num_candidates,
                            "filter": {"term": {"status": "PUBLISHED"}},
                        },
                        {
                            "field": "business_text_vector",
                            "query_vector": embedding,
                            "k": min(4, self._config.top_k),
                            "num_candidates": self._config.num_candidates,
                            "boost": 1.15,
                            "filter": {
                                "bool": {
                                    "filter": [
                                        {"term": {"status": "PUBLISHED"}},
                                        {"term": {"document_kind": "trip"}},
                                        {"term": {"sale_summary.inquiry_eligible": True}},
                                    ]
                                }
                            },
                        },
                    ],
                    "_source": [
                        "document_id",
                        "document_kind",
                        "title",
                        "tags",
                        "summary",
                        "business_search_text",
                        "cover_url",
                        "display_url",
                        "province_name",
                        "city_name",
                        "scenic_name",
                        "trip_id",
                        "duration",
                        "min_price",
                        "itinerary_days",
                        "packages",
                        "sale_summary",
                        "width",
                        "height",
                    ],
                },
            )
            search_response.raise_for_status()
            scenic_hits: list[dict[str, Any]] = []
            if _query_needs_images(query):
                scenic_response = await client.post(
                    f"{self._config.elasticsearch_url}/gta_scenic_admin/_search",
                    auth=(
                        self._config.elasticsearch_username,
                        self._config.elasticsearch_password,
                    ),
                    json={
                        "size": 12,
                        "query": {
                            "bool": {
                                "filter": [{"term": {"status": "PUBLISHED"}}],
                                "must": [
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": [
                                                "scenic_name^5",
                                                "title^4",
                                                "city_name^3",
                                                "province_name^2",
                                                "tags^2",
                                                "business_search_text",
                                            ],
                                        }
                                    }
                                ],
                            }
                        },
                        "_source": [
                            "document_id",
                            "document_kind",
                            "title",
                            "tags",
                            "cover_url",
                            "province_name",
                            "city_name",
                            "scenic_name",
                            "width",
                            "height",
                        ],
                    },
                )
                scenic_response.raise_for_status()
                scenic_hits = [
                    hit
                    for hit in scenic_response.json().get("hits", {}).get("hits", [])
                    if isinstance(hit, dict)
                ]
            site_context = await self._retrieve_site_context(client, query)

        hits = search_response.json().get("hits", {}).get("hits", [])
        product_hits = [
            hit
            for hit in hits
            if isinstance(hit, dict)
            and isinstance(hit.get("_source"), dict)
            and _text(hit["_source"].get("document_kind")) == "trip"
            and isinstance(hit["_source"].get("sale_summary"), dict)
            and hit["_source"]["sale_summary"].get("inquiry_eligible") is True
        ]
        product_hits.sort(key=lambda hit: _product_intent_score(query, hit), reverse=True)
        product_documents = [
            _format_trip_product(hit, self._config.max_document_chars)
            for hit in product_hits[:2]
        ]
        product_documents = [document for document in product_documents if document]
        quote_candidates = tuple(
            candidate
            for hit in product_hits[:3]
            if (candidate := _quote_candidate(hit)) is not None
        )
        documents = [
            _format_hit(hit, self._config.max_document_chars)
            for hit in hits[: self._config.context_top_k]
            if isinstance(hit, dict)
            and _text((hit.get("_source") or {}).get("document_kind")) != "trip"
        ]
        documents = [document for document in documents if document]
        context_sections: list[str] = []
        if site_context:
            context_sections.append(site_context)
        if product_documents and not site_context:
            context_sections.append(
                "【绿色旅行网可推荐行程产品】\n" + "\n\n".join(product_documents)
            )
        if documents and not site_context:
            context_sections.append("【补充知识资料】\n" + "\n\n".join(documents))
        context = "\n\n".join(context_sections)[: self._config.max_context_chars]
        images: list[tuple[str, str]] = []
        destination_matched = any(
            isinstance(hit, dict)
            and isinstance(hit.get("_source"), dict)
            and _image_matches_query(query, hit["_source"])
            for hit in hits
        )
        if _query_allows_images(query) and (_query_needs_images(query) or destination_matched):
            seen_paths: set[str] = set()
            # 文本向量相关不等于封面画面相关。图片采用独立的更高阈值并要求
            # 标题、标签或目的地名称与用户问题存在明确主题词重合。
            image_hits = [*scenic_hits, *hits]
            for hit in sorted(image_hits, key=_image_rank, reverse=True):
                source = hit.get("_source") if isinstance(hit, dict) else None
                if not isinstance(source, dict):
                    continue
                if float(hit.get("_score") or 0) < self._config.image_min_score:
                    continue
                if not _image_matches_query(query, source):
                    continue
                if not _is_landscape(source):
                    continue
                path = _text(source.get("cover_url"))
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                image_title = (
                    _text(source.get("scenic_name"))
                    or _text(source.get("city_name"))
                    or _text(source.get("province_name"))
                    or _text(source.get("title"))
                    or "知识库图片"
                )
                images.append((image_title, path))
                if len(images) == 3:
                    break
        elapsed_ms = round((time.monotonic() - started) * 1000)
        return RetrievalResult(
            query=query,
            context=context,
            hit_count=(1 if site_context else len(product_documents) + len(documents)),
            elapsed_ms=elapsed_ms,
            images=tuple(images),
            quote_candidates=quote_candidates,
        )

    async def _retrieve_site_context(self, client: httpx.AsyncClient, query: str) -> str:
        modules = _site_modules_for_query(query)
        if not modules:
            return ""
        urls = ["/about" if module == "about" else "/help-center" for module in modules]
        response = await client.post(
            f"{self._config.elasticsearch_url}/gta_site/_search",
            auth=(self._config.elasticsearch_username, self._config.elasticsearch_password),
            json={
                "size": len(urls),
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"document_kind": "site_page"}},
                            {"term": {"status": True}},
                            {"terms": {"url": urls}},
                        ]
                    }
                },
                "_source": [
                    "name",
                    "url",
                    "resolved_sections_json",
                    "faqs.question",
                    "faqs.answer",
                    "faqs.category",
                ],
            },
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        sections: list[str] = []
        for hit in hits:
            source = hit.get("_source") if isinstance(hit, dict) else None
            if not isinstance(source, dict):
                continue
            if source.get("url") == "/about":
                about = _format_about_page(source)
                if about:
                    sections.append(about)
            elif source.get("url") == "/help-center":
                help_text = _format_help_page(source, query)
                if help_text:
                    sections.append(help_text)
        return "\n\n".join(sections)

    async def retrieve_images_from_answer(
        self, answer_text: str, user_query: str = ""
    ) -> tuple[tuple[str, str], ...]:
        """Select image domain from the answer subject, then require exact assets."""
        text = answer_text[-2000:].strip()
        if not self._config.enabled or len(text) < 40:
            return ()
        image_theme = _answer_image_theme(user_query, text)
        timeout = httpx.Timeout(self._config.timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            verify=self._config.verify_tls,
        ) as client:
            if image_theme:
                theme_images = await self._retrieve_theme_images(
                    client,
                    image_theme,
                    user_query,
                )
                if theme_images:
                    return theme_images
            if _query_needs_images(user_query):
                if re.search(r"[\u4e00-\u9fff]", user_query):
                    keyword_images = await self._retrieve_keyword_images(client, user_query)
                    if keyword_images:
                        return keyword_images
                semantic_images = await self._retrieve_semantic_scenic_images(
                    client,
                    user_query,
                )
                if semantic_images:
                    return semantic_images
            scenic_response = await client.post(
                f"{self._config.elasticsearch_url}/"
                "gta_province_admin,gta_city_admin,gta_scenic_admin/_search",
                auth=(
                    self._config.elasticsearch_username,
                    self._config.elasticsearch_password,
                ),
                json={
                    "size": 100,
                    "query": {
                        "bool": {
                            "filter": [{"term": {"status": "PUBLISHED"}}],
                            "must": [
                                {
                                    "multi_match": {
                                        "query": text,
                                        "fields": [
                                            "scenic_name^6",
                                            "title^4",
                                            "city_name^3",
                                            "province_name^2",
                                            "tags^2",
                                        ],
                                    }
                                }
                            ],
                        }
                    },
                    "_source": [
                        "title",
                        "scenic_name",
                        "city_name",
                        "province_name",
                        "cover_url",
                        "width",
                        "height",
                    ],
                },
            )
            scenic_response.raise_for_status()
            scenic_hits = scenic_response.json().get("hits", {}).get("hits", [])

            mentioned_names: list[str] = []
            images: list[tuple[str, str]] = []
            seen_paths: set[str] = set()
            ranked: list[tuple[int, int, dict[str, Any]]] = []
            for hit in scenic_hits:
                source = hit.get("_source") if isinstance(hit, dict) else None
                if not isinstance(source, dict):
                    continue
                name = (
                    _text(source.get("scenic_name"))
                    or _text(source.get("city_name"))
                    or _text(source.get("province_name"))
                )
                if len(name) < 2 or name not in text:
                    continue
                entity_priority = (
                    0
                    if _text(source.get("scenic_name"))
                    else 1
                    if _text(source.get("city_name"))
                    else 2
                )
                ranked.append((entity_priority, text.find(name), source))
            for _, _, source in sorted(ranked, key=lambda item: (item[0], item[1])):
                name = (
                    _text(source.get("scenic_name"))
                    or _text(source.get("city_name"))
                    or _text(source.get("province_name"))
                )
                if name in mentioned_names:
                    continue
                mentioned_names.append(name)
                path = _text(source.get("cover_url"))
                if _is_landscape(source) and path and path not in seen_paths:
                    seen_paths.add(path)
                    images.append((name, path))
                if len(images) == 3:
                    return tuple(images)
        return tuple(images)

    async def _retrieve_keyword_images(
        self,
        client: httpx.AsyncClient,
        query: str,
    ) -> tuple[tuple[str, str], ...]:
        """Retrieve exact Chinese destination assets from every business index."""
        response = await client.post(
            f"{self._config.elasticsearch_url}/{','.join(self._config.elasticsearch_indices)}"
            "/_search",
            auth=(
                self._config.elasticsearch_username,
                self._config.elasticsearch_password,
            ),
            json={
                "size": 50,
                "query": {
                    "bool": {
                        "filter": [{"term": {"status": "PUBLISHED"}}],
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "scenic_name^7",
                                        "city_name^6",
                                        "province_name^5",
                                        "title^5",
                                        "tags^3",
                                        "business_search_text",
                                    ],
                                }
                            }
                        ],
                    }
                },
                "_source": [
                    "title",
                    "tags",
                    "scenic_name",
                    "city_name",
                    "province_name",
                    "cover_url",
                    "width",
                    "height",
                ],
            },
        )
        response.raise_for_status()
        images: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for hit in response.json().get("hits", {}).get("hits", []):
            source = hit.get("_source") if isinstance(hit, dict) else None
            if not isinstance(source, dict):
                continue
            if not _image_matches_query(query, source) or not _is_landscape(source):
                continue
            path = _text(source.get("cover_url"))
            if not path or path in seen_paths:
                continue
            title = (
                _text(source.get("scenic_name"))
                or _text(source.get("city_name"))
                or _text(source.get("province_name"))
                or _text(source.get("title"))
            )
            if not title:
                continue
            seen_paths.add(path)
            images.append((title, path))
            if len(images) == 3:
                break
        return tuple(images)

    async def _retrieve_semantic_scenic_images(
        self,
        client: httpx.AsyncClient,
        query: str,
    ) -> tuple[tuple[str, str], ...]:
        """Use multilingual vectors when query words cannot match Chinese ES names."""
        embedding_response = await client.post(
            self._config.embedding_url,
            json={
                "model": self._config.embedding_model,
                "input": [query[: self._config.max_query_chars]],
                "encoding_format": "float",
            },
        )
        embedding_response.raise_for_status()
        embedding = _read_embedding(embedding_response.json())
        response = await client.post(
            f"{self._config.elasticsearch_url}/"
            "gta_province_admin,gta_city_admin,gta_scenic_admin/_search",
            auth=(
                self._config.elasticsearch_username,
                self._config.elasticsearch_password,
            ),
            json={
                "size": 12,
                # Cross-language similarity is lower than same-language similarity.
                # City anchoring below provides the strict relevance boundary.
                "min_score": max(0.55, self._config.image_min_score - 0.20),
                "knn": {
                    "field": "business_text_vector",
                    "query_vector": embedding,
                    "k": 12,
                    "num_candidates": self._config.num_candidates,
                    "filter": {"term": {"status": "PUBLISHED"}},
                },
                "_source": [
                    "title",
                    "scenic_name",
                    "city_name",
                    "province_name",
                    "cover_url",
                    "width",
                    "height",
                ],
            },
        )
        response.raise_for_status()
        sources = [
            hit.get("_source")
            for hit in response.json().get("hits", {}).get("hits", [])
            if isinstance(hit, dict) and isinstance(hit.get("_source"), dict)
        ]
        anchor = next(
            (
                _text(source.get("city_name")) or _text(source.get("province_name"))
                for source in sources
                if _text(source.get("city_name")) or _text(source.get("province_name"))
            ),
            "",
        )
        images: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for source in sources:
            if anchor:
                title = _text(source.get("title"))
                scenic_name = _text(source.get("scenic_name"))
                city_name = _text(source.get("city_name"))
                province_name = _text(source.get("province_name"))
                belongs_to_anchor = (
                    city_name == anchor
                    or province_name == anchor
                    or scenic_name.startswith(anchor)
                    or title.startswith(anchor)
                    or f"位于{anchor}" in title
                    or f"坐落于{anchor}" in title
                )
                if not belongs_to_anchor:
                    continue
            if not _is_landscape(source):
                continue
            path = _text(source.get("cover_url"))
            if not path or path in seen_paths:
                continue
            title = (
                _text(source.get("scenic_name"))
                or _text(source.get("city_name"))
                or _text(source.get("province_name"))
                or _text(source.get("title"))
            )
            if not title:
                continue
            seen_paths.add(path)
            images.append((title, path))
            if len(images) == 3:
                break
        return tuple(images)

    async def _retrieve_theme_images(
        self,
        client: httpx.AsyncClient,
        image_theme: tuple[str, tuple[str, ...]],
        user_query: str,
    ) -> tuple[tuple[str, str], ...]:
        """Search business media for one subject; never fall back to scenic covers."""
        theme_name, terms = image_theme
        subject_terms = _theme_query_subjects(user_query, terms)
        business_indices = tuple(
            index
            for index in self._config.elasticsearch_indices
            if index
            not in {
                "gta_province_admin",
                "gta_city_admin",
                "gta_scenic_admin",
            }
        )
        if not business_indices:
            return ()
        query = " ".join((user_query.strip(), *terms)).strip()
        response = await client.post(
            f"{self._config.elasticsearch_url}/{','.join(business_indices)}/_search",
            auth=(
                self._config.elasticsearch_username,
                self._config.elasticsearch_password,
            ),
            json={
                "size": 30,
                "query": {
                    "bool": {
                        "filter": [{"term": {"status": "PUBLISHED"}}],
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "title^6",
                                        "tags^5",
                                        "summary^3",
                                        "business_search_text^2",
                                    ],
                                }
                            }
                        ],
                    }
                },
                "_source": [
                    "document_kind",
                    "title",
                    "tags",
                    "summary",
                    "business_search_text",
                    "cover_url",
                    "width",
                    "height",
                ],
            },
        )
        response.raise_for_status()
        images: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for hit in response.json().get("hits", {}).get("hits", []):
            source = hit.get("_source") if isinstance(hit, dict) else None
            if not isinstance(source, dict) or not _source_matches_theme(source, terms):
                continue
            if subject_terms and not _source_matches_subject(source, subject_terms):
                continue
            path = _text(source.get("cover_url"))
            if not _is_landscape(source) or not path or path in seen_paths:
                continue
            seen_paths.add(path)
            title = _text(source.get("title")) or theme_name
            images.append((title, path))
            if len(images) == 3:
                break
        return tuple(images)

    async def extract_place_entities(self, text: str) -> tuple[str, ...]:
        """Extract only ES-confirmed place names that occur verbatim in a turn."""
        content = text[-6000:].strip()
        if not self._config.enabled or len(content) < 2:
            return ()
        timeout = httpx.Timeout(self._config.timeout_seconds)
        indices = "gta_province_admin,gta_city_admin,gta_scenic_admin"
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            verify=self._config.verify_tls,
        ) as client:
            response = await client.post(
                f"{self._config.elasticsearch_url}/{indices}/_search",
                auth=(
                    self._config.elasticsearch_username,
                    self._config.elasticsearch_password,
                ),
                json={
                    "size": 100,
                    "query": {
                        "bool": {
                            "filter": [{"term": {"status": "PUBLISHED"}}],
                            "must": [
                                {
                                    "multi_match": {
                                        "query": content,
                                        "fields": [
                                            "province_name^6",
                                            "city_name^6",
                                            "scenic_name^7",
                                            "title^2",
                                        ],
                                    }
                                }
                            ],
                        }
                    },
                    "_source": ["province_name", "city_name", "scenic_name"],
                },
            )
            response.raise_for_status()
        found: list[tuple[int, str]] = []
        for hit in response.json().get("hits", {}).get("hits", []):
            source = hit.get("_source") if isinstance(hit, dict) else None
            if not isinstance(source, dict):
                continue
            for field in ("province_name", "city_name", "scenic_name"):
                name = _text(source.get(field))
                if len(name) >= 2 and name in content:
                    found.append((content.find(name), name))
        names: list[str] = []
        for _, name in sorted(found, key=lambda item: item[0]):
            if name not in names:
                names.append(name)
        return tuple(names)


def _read_embedding(payload: Any) -> list[float]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("Embedding响应data数量不正确")
    values = data[0].get("embedding") if isinstance(data[0], dict) else None
    if not isinstance(values, list) or len(values) != 1024:
        actual = len(values) if isinstance(values, list) else 0
        raise ValueError(f"Embedding响应维度不正确: actual={actual}, expected=1024")
    return [float(value) for value in values]


def _image_rank(hit: dict[str, Any]) -> float:
    source = hit.get("_source")
    if not isinstance(source, dict):
        return -1.0
    score = float(hit.get("_score") or 0)
    kind = _text(source.get("document_kind"))
    title = _text(source.get("title"))
    if kind in {"scenic", "city", "province"}:
        score += 0.08
    elif kind == "article":
        score += 0.03
    if any(term in title for term in ("景点", "风景", "园林", "古镇", "四季", "全攻略")):
        score += 0.035
    if any(term in title for term in ("包团", "定制", "男友", "好友", "产品", "线路")):
        score -= 0.08
    return score


def _is_landscape(source: dict[str, Any]) -> bool:
    try:
        width = int(source.get("width") or 0)
        height = int(source.get("height") or 0)
    except (TypeError, ValueError):
        return False
    return width > height > 0


_ANSWER_IMAGE_THEMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "包车用车",
        ("包车", "用车", "车辆", "车型", "轿车", "商务车", "大巴", "司机", "接送", "接机", "送机"),
    ),
    ("导游服务", ("导游", "领队", "讲解员", "中文导游", "英文导游", "地陪")),
    ("酒店住宿", ("酒店", "住宿", "客房", "房型", "民宿", "度假村")),
    ("餐饮美食", ("餐厅", "餐饮", "美食", "菜品", "下午茶", "宴席")),
    ("旅游活动", ("演出", "游船", "漂流", "滑雪", "温泉", "骑行", "徒步")),
)


def _answer_image_theme(user_query: str, answer_text: str) -> tuple[str, tuple[str, ...]] | None:
    """Choose a product image theme only when the current user request names it."""
    query = user_query.strip()
    for theme in _ANSWER_IMAGE_THEMES:
        if any(term in query for term in theme[1]):
            return theme
    return None


def _source_matches_theme(source: dict[str, Any], terms: tuple[str, ...]) -> bool:
    text = _source_search_text(source)
    return any(term in text for term in terms)


def _theme_query_subjects(user_query: str, theme_terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", user_query)
    noise = (*theme_terms, "推荐", "介绍", "一下", "怎么", "如何", "服务", "旅游", "旅行")
    for term in sorted(noise, key=len, reverse=True):
        normalized = normalized.replace(term, " ")
    return tuple(term for term in normalized.split() if len(term) >= 2)


def _source_matches_subject(source: dict[str, Any], subjects: tuple[str, ...]) -> bool:
    text = _source_search_text(source)
    return any(subject in text for subject in subjects)


def _source_search_text(source: dict[str, Any]) -> str:
    values = (
        source.get("title"),
        source.get("tags"),
        source.get("summary"),
        source.get("business_search_text"),
    )
    text = " ".join(
        " ".join(_text(item) for item in value) if isinstance(value, list) else _text(value)
        for value in values
    )
    return text


_IMAGE_INTENT_TERMS = (
    "旅游",
    "旅行",
    "想去",
    "出游",
    "度假",
    "游玩",
    "攻略",
    "景点",
    "行程",
    "路线",
    "怎么玩",
    "去哪里",
    "自由行",
    "周边游",
    "图文",
    "图片",
    "配图",
    "照片",
)
_IMAGE_INTENT_CASEFOLD_TERMS = (
    "travel",
    "trip",
    "tour",
    "itinerary",
    "sightseeing",
    "attraction",
    "vacation",
    "holiday",
    "travel guide",
    "things to do",
    "旅遊",
    "観光",
    "ツアー",
    "여행",
    "관광",
    "투어",
    "voyage",
    "viaje",
    "viagem",
    "viaggio",
    "reizen",
    "reise",
    "podróż",
    "seyahat",
    "туризм",
    "путешествие",
    "سفر",
    "رحلة",
    "यात्रा",
    "ท่องเที่ยว",
    "du lịch",
    "perjalanan",
)
_NO_IMAGE_QUERY_TERMS = (
    "多少钱",
    "价格",
    "报价",
    "费用",
    "预算",
    "赚钱",
    "利润",
    "哪个公司",
    "公司介绍",
    "政策",
    "合同",
)
_IMAGE_QUERY_NOISE = (
    "旅游攻略",
    "旅行攻略",
    "旅游",
    "旅行",
    "攻略",
    "景点",
    "行程",
    "路线",
    "怎么玩",
    "去哪里",
    "自由行",
    "周边游",
    "图文",
    "图片",
    "配图",
    "照片",
    "推荐一下",
    "帮我推荐",
    "帮我输出",
    "帮我",
    "我需要",
    "我想去",
    "想去",
    "需要多少钱",
    "多少钱",
    "几天",
    "三天",
    "两天",
    "一天",
    "介绍一下",
    "介绍",
    "推荐",
    "一下",
    "请问",
)
_GENERIC_IMAGE_MATCHES = {
    "旅游",
    "旅行",
    "攻略",
    "景点",
    "行程",
    "路线",
    "推荐",
    "需要",
    "怎么",
    "可以",
    "一个",
    "我们",
    "他们",
    "什么",
    "三天",
    "两天",
    "一天",
    "图文",
    "图片",
}


def _query_needs_images(query: str) -> bool:
    """Prefer images for destination travel queries, excluding business queries."""
    folded = query.casefold()
    return _query_allows_images(query) and (
        any(term in query for term in _IMAGE_INTENT_TERMS)
        or any(term.casefold() in folded for term in _IMAGE_INTENT_CASEFOLD_TERMS)
    )


def _query_allows_images(query: str) -> bool:
    return not any(term in query for term in _NO_IMAGE_QUERY_TERMS)


def _image_matches_query(query: str, source: dict[str, Any]) -> bool:
    values = [
        source.get("title"),
        source.get("tags"),
        source.get("province_name"),
        source.get("city_name"),
        source.get("scenic_name"),
    ]
    haystack = " ".join(
        " ".join(_text(item) for item in value) if isinstance(value, list) else _text(value)
        for value in values
        if value is not None
    )
    if not haystack:
        return False
    # Multilingual vector relevance handles non-Chinese queries because ES place
    # fields are Chinese. Only explicit place assets pass; score and shape remain gated.
    if (
        not re.search(r"[\u4e00-\u9fff]", query)
        and _query_needs_images(query)
        and any(_text(source.get(field)) for field in ("scenic_name", "city_name", "province_name"))
    ):
        return True
    normalized_query = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", query).casefold()
    for noise in sorted(_IMAGE_QUERY_NOISE, key=len, reverse=True):
        normalized_query = normalized_query.replace(noise.casefold(), " ")
    fragments = [fragment for fragment in normalized_query.split() if len(fragment) >= 2]
    normalized_haystack = haystack.casefold()
    for fragment in fragments:
        if fragment in normalized_haystack:
            return True
        max_size = min(6, len(fragment))
        for size in range(max_size, 1, -1):
            for offset in range(len(fragment) - size + 1):
                candidate = fragment[offset : offset + size]
                if candidate in _GENERIC_IMAGE_MATCHES:
                    continue
                if candidate in normalized_haystack:
                    return True
    return False


def _format_trip_product(hit: dict[str, Any], max_chars: int) -> str:
    source = hit.get("_source")
    if not isinstance(source, dict) or _text(source.get("document_kind")) != "trip":
        return ""
    title = _text(source.get("title"))
    if not title:
        return ""

    summary = _text(source.get("summary"))[:max_chars]
    itinerary = source.get("itinerary_days")
    itinerary_lines: list[str] = []
    if isinstance(itinerary, list):
        for day in itinerary[:5]:
            if not isinstance(day, dict):
                continue
            day_number = day.get("day_number")
            day_title = _text(day.get("title"))
            route = " → ".join(
                value
                for value in (_text(day.get("from_address")), _text(day.get("to_address")))
                if value
            )
            detail = "; ".join(value for value in (day_title, route) if value)
            if detail:
                itinerary_lines.append(f"第{day_number}天: {detail}")

    package_lines: list[str] = []
    default_party_lines: list[str] = []
    packages = source.get("packages")
    if isinstance(packages, list):
        for package in packages[:3]:
            if not isinstance(package, dict):
                continue
            name = _text(package.get("sku_name")) or _text(package.get("package_name"))
            people = _text(package.get("applicable_people"))
            min_pax = package.get("min_pax")
            pricing_mode = _text(package.get("pricing_mode"))
            description = "; ".join(
                value
                for value in (
                    name,
                    f"适用人群: {people}" if people else "",
                    f"至少{min_pax}人" if isinstance(min_pax, int) and min_pax > 0 else "",
                    f"计价方式: {pricing_mode}" if pricing_mode else "",
                )
                if value
            )
            if description:
                package_lines.append(description)
            default_party = "; ".join(
                value
                for value in (
                    people,
                    f"默认核价人数不得低于{min_pax}人"
                    if isinstance(min_pax, int) and min_pax > 0
                    else "",
                )
                if value
            )
            if default_party and default_party not in default_party_lines:
                default_party_lines.append(default_party)

    sale_summary = source.get("sale_summary")
    price_line = ""
    quote_line = ""
    if isinstance(sale_summary, dict):
        price = sale_summary.get("reference_start_price")
        currency = _text(sale_summary.get("currency"))
        if isinstance(price, int | float) and not isinstance(price, bool):
            price_line = f"参考起价: {price:g} {currency}".rstrip() + " (不是实时报价)"
        if sale_summary.get("requires_realtime_quote") is True:
            quote_line = "准确价格需要根据日期、人数和套餐实时询价"

    fields = [
        f"产品名称: {title}",
        f"产品编号: {_text(source.get('trip_id'))}",
        f"产品简介: {summary}",
        f"行程天数: {_text(source.get('duration'))}",
        "行程安排: " + " | ".join(itinerary_lines) if itinerary_lines else "",
        "可选套餐: " + " | ".join(package_lines) if package_lines else "",
        "默认人数与核价口径: " + " | ".join(default_party_lines)
        if default_party_lines
        else "",
        price_line,
        quote_line,
        f"产品页面: {_text(source.get('display_url'))}",
        f"产品相关度: {float(hit.get('_score') or 0):.4f}",
    ]
    return "\n".join(field for field in fields if field and not field.endswith(": "))


def _quote_candidate(hit: dict[str, Any]) -> QuoteCandidate | None:
    source = hit.get("_source")
    if not isinstance(source, dict):
        return None
    packages = source.get("packages")
    if not isinstance(packages, list):
        return None
    for package in packages:
        if not isinstance(package, dict):
            continue
        spu_id = _text(package.get("spu_id"))
        if not spu_id:
            continue
        min_pax = package.get("min_pax")
        return QuoteCandidate(
            title=_text(source.get("title")),
            trip_id=_text(source.get("trip_id")),
            spu_id=spu_id,
            min_pax=min_pax if isinstance(min_pax, int) and min_pax > 0 else 1,
            applicable_people=_text(package.get("applicable_people")),
            display_url=_text(source.get("display_url")),
        )
    return None


_ABOUT_TERMS = (
    "关于我们",
    "公司介绍",
    "你们公司",
    "哪家公司",
    "所属公司",
    "谁开发",
    "谁研发",
    "谁运营",
    "谁创建",
    "开发者",
    "运营方",
    "绿色旅行网是谁",
    "旅行社资质",
    "company",
    "about us",
    "who are you",
    "greentourasia",
)
_ABOUT_RECOMMENDATION_PATTERN = re.compile(
    r"(?:旅游公司|旅行公司|旅行社|旅游平台).{0,16}(?:推荐|最好|靠谱|选择|哪家|比较)|"
    r"(?:推荐|最好|靠谱|选择|哪家|比较).{0,16}(?:旅游公司|旅行公司|旅行社|旅游平台)|"
    r"(?:旅游|旅行).{0,20}(?:推荐|最好|靠谱|选择|哪家|比较).{0,12}(?:公司|旅行社|平台)",
    re.I,
)
_HELP_TERMS = (
    "帮助中心",
    "怎么预订",
    "如何预订",
    "付款",
    "支付",
    "退款",
    "取消订单",
    "投诉",
    "预订政策",
    "help",
    "booking",
    "payment",
    "refund",
    "cancel",
)


def _site_modules_for_query(query: str) -> tuple[str, ...]:
    folded = query.casefold()
    modules: list[str] = []
    if (
        any(term.casefold() in folded for term in _ABOUT_TERMS)
        or _ABOUT_RECOMMENDATION_PATTERN.search(query)
    ):
        modules.append("about")
    if any(term.casefold() in folded for term in _HELP_TERMS):
        modules.append("help")
    return tuple(modules)


def _format_about_page(source: dict[str, Any]) -> str:
    raw = source.get("resolved_sections_json")
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        sections = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    lines: list[str] = []
    for section in sections if isinstance(sections, list) else []:
        if not isinstance(section, dict):
            continue
        title = _text(section.get("section_name"))
        description = _text(section.get("section_sub_name"))
        if description:
            lines.append("\n".join(value for value in (title, description) if value))
        if len("\n".join(lines)) >= 1800:
            break
    if not lines:
        return ""
    return "【绿色旅行网公司资料】\n" + "\n\n".join(lines)[:2000]


def _format_help_page(source: dict[str, Any], query: str) -> str:
    faqs = source.get("faqs")
    if not isinstance(faqs, list):
        return ""
    terms = set(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,6}", query.casefold()))

    def score(item: dict[str, Any]) -> int:
        text = " ".join(
            _text(item.get(field)).casefold() for field in ("question", "category", "answer")
        )
        return sum(1 for term in terms if term in text)

    ranked = sorted(
        (item for item in faqs if isinstance(item, dict)), key=score, reverse=True
    )
    selected = [item for item in ranked[:3] if score(item) > 0]
    if not selected:
        selected = ranked[:2]
    lines = [
        f"问题: {_text(item.get('question'))}\n答案: {_text(item.get('answer'))}"
        for item in selected
    ]
    return "【绿色旅行网帮助中心】\n" + "\n\n".join(lines)[:1800] if lines else ""


def _product_intent_score(query: str, hit: dict[str, Any]) -> float:
    source = hit.get("_source")
    if not isinstance(source, dict):
        return float(hit.get("_score") or 0)
    title = _text(source.get("title"))
    summary = _text(source.get("summary"))
    duration = _text(source.get("duration"))
    tags = source.get("tags")
    tag_text = " ".join(_text(tag) for tag in tags) if isinstance(tags, list) else _text(tags)
    packages = source.get("packages")
    package_text = " ".join(
        " ".join(
            _text(package.get(field))
            for field in ("package_name", "sku_name", "applicable_people", "features")
        )
        for package in packages or []
        if isinstance(package, dict)
    )
    query_folded = query.casefold()
    weighted_fields = (
        (title.casefold(), 5.0),
        (tag_text.casefold(), 3.0),
        (duration.casefold(), 3.0),
        (package_text.casefold(), 2.0),
        (summary.casefold(), 1.0),
    )
    terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,8}", query_folded))
    # 中文目的地通常黏在整句中。补充二至四字片段以强化地域和天数匹配。
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", query_folded):
        for size in (4, 3, 2):
            terms.update(chunk[offset : offset + size] for offset in range(len(chunk) - size + 1))
    lexical_score = sum(
        weight
        for term in terms
        if term not in _IMAGE_QUERY_NOISE
        for field, weight in weighted_fields
        if term and term in field
    )
    return float(hit.get("_score") or 0) + lexical_score


def _format_hit(hit: dict[str, Any], max_chars: int) -> str:
    source = hit.get("_source")
    if not isinstance(source, dict):
        return ""
    title = _text(source.get("title")) or _text(source.get("document_id"))
    summary = _text(source.get("summary"))
    knowledge = _text(source.get("business_search_text"))
    if summary and knowledge.startswith(summary):
        knowledge = knowledge[len(summary) :].lstrip()
    content = "\n".join(part for part in (summary, knowledge) if part)[:max_chars]
    fields = [
        f"标题: {title}",
        f"类型: {_text(source.get('document_kind'))}",
        f"内容: {content}",
        f"页面: {_text(source.get('display_url'))}",
        f"相关度: {float(hit.get('_score') or 0):.4f}",
    ]
    return "\n".join(field for field in fields if not field.endswith(": "))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
