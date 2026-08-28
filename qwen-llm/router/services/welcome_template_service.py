"""广告关键词欢迎模板的ES向量检索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class WelcomeTemplate:
    template_id: str
    title: str
    content: str
    images: tuple[dict[str, str], ...]
    suggested_questions: tuple[str, ...]
    ui_text: dict[str, str]
    version: int


class WelcomeTemplateService:
    def __init__(
        self,
        *,
        embedding_url: str,
        embedding_model: str,
        elasticsearch_url: str,
        elasticsearch_username: str,
        elasticsearch_password: str,
        elasticsearch_index: str,
        verify_tls: bool,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._embedding_url = embedding_url
        self._embedding_model = embedding_model
        self._elasticsearch_url = elasticsearch_url.rstrip("/")
        self._username = elasticsearch_username
        self._password = elasticsearch_password
        self._index = elasticsearch_index
        self._verify_tls = verify_tls
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def retrieve(self, keyword: str) -> WelcomeTemplate:
        query = keyword.strip()[:500]
        timeout = httpx.Timeout(self._timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            verify=self._verify_tls,
        ) as client:
            if query:
                embedding = await self._embedding(client, query)
                result = await self._search(
                    client,
                    {
                        "size": 1,
                        "knn": {
                            "field": "keyword_vector",
                            "query_vector": embedding,
                            "k": 1,
                            "num_candidates": 100,
                            "filter": {"term": {"status": "PUBLISHED"}},
                        },
                        "query": {
                            "bool": {
                                "should": [
                                    {"term": {"keywords.exact": {"value": query, "boost": 4}}},
                                    {"match": {"keywords": {"query": query, "boost": 2}}},
                                ]
                            }
                        },
                    },
                )
                if result is not None:
                    return result
            fallback = await self._search(
                client,
                {
                    "size": 1,
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"status": "PUBLISHED"}},
                                {"term": {"is_default": True}},
                            ]
                        }
                    },
                    "sort": [{"version": "desc"}],
                },
            )
            if fallback is None:
                raise LookupError("没有可用的欢迎模板")
            return fallback

    async def _embedding(self, client: httpx.AsyncClient, text: str) -> list[float]:
        response = await client.post(
            self._embedding_url,
            json={
                "model": self._embedding_model,
                "input": [text],
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        payload = response.json()
        values = payload.get("data", [{}])[0].get("embedding", [])
        if not isinstance(values, list) or len(values) != 1024:
            raise ValueError("欢迎模板查询向量必须是1024维")
        return [float(value) for value in values]

    async def _search(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
    ) -> WelcomeTemplate | None:
        response = await client.post(
            f"{self._elasticsearch_url}/{self._index}/_search",
            auth=(self._username, self._password),
            json=body,
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        if not hits:
            return None
        source = hits[0].get("_source", {})
        images = tuple(
            {
                "title": str(image.get("title", "")),
                "path": str(image.get("path", "")),
            }
            for image in source.get("images", [])
            if isinstance(image, dict) and image.get("path")
        )
        questions = tuple(
            str(question).strip()
            for question in source.get("suggested_questions", [])
            if str(question).strip()
        )
        raw_ui_text = source.get("ui_text", {})
        ui_text = (
            {
                str(key).strip(): str(value).strip()
                for key, value in raw_ui_text.items()
                if str(key).strip() and str(value).strip()
            }
            if isinstance(raw_ui_text, dict)
            else {}
        )
        return WelcomeTemplate(
            template_id=str(source.get("template_id", hits[0].get("_id", ""))),
            title=str(source.get("title", "")),
            content=str(source.get("content", "")),
            images=images,
            suggested_questions=questions,
            ui_text=ui_text,
            version=int(source.get("version", 1)),
        )
