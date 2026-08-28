from __future__ import annotations

from typing import Any

import httpx

GOOGLE_API_KEY = "AIzaSyCjAyPb-6c2X6da1FHXXbIMF7ncTenreTU"
GOOGLE_PLACE_ID = "ChIJNaM0g_MBBDQRk_5maXNSmSQ"
ELASTICSEARCH_URL = "https://192.168.80.130:9200"
ELASTICSEARCH_USERNAME = "elastic"
ELASTICSEARCH_PASSWORD = "kp-+K3SgKFJs_+UDojK9"
PLATFORM_REVIEWS_INDEX = "gta_trip_admin_business_v3"
PLATFORM_REVIEWS_URL = "http://192.168.100.5:9012/review/list"
REQUEST_TIMEOUT_SECONDS = 3.0


class WebReviewService:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def google(self, locale: str = "zh-CN") -> dict[str, Any]:
        headers = {
            "X-Goog-Api-Key": GOOGLE_API_KEY,
            "X-Goog-FieldMask": (
                "displayName,rating,userRatingCount,googleMapsUri,"
                "reviews.authorAttribution,reviews.rating,reviews.text,"
                "reviews.relativePublishTimeDescription,reviews.publishTime,"
                "reviews.googleMapsUri"
            ),
        }
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=self._transport,
        ) as client:
            response = await client.get(
                f"https://places.googleapis.com/v1/places/{GOOGLE_PLACE_ID}",
                headers=headers,
                params={"languageCode": locale or "zh-CN"},
            )
            response.raise_for_status()
            payload = response.json()
        reviews = []
        for item in payload.get("reviews", []):
            if not isinstance(item, dict):
                continue
            author = item.get("authorAttribution") or {}
            text = item.get("text") or {}
            reviews.append(
                {
                    "author_name": str(author.get("displayName", "")),
                    "author_uri": str(author.get("uri", "")),
                    "author_photo_uri": str(author.get("photoUri", "")),
                    "rating": float(item.get("rating") or 0),
                    "text": str(text.get("text", "")),
                    "relative_time": str(item.get("relativePublishTimeDescription", "")),
                    "publish_time": str(item.get("publishTime", "")),
                    "google_maps_uri": str(item.get("googleMapsUri", "")),
                }
            )
        display_name = payload.get("displayName") or {}
        return {
            "source": "google",
            "display_name": str(display_name.get("text", "")),
            "rating": float(payload.get("rating") or 0),
            "total": int(payload.get("userRatingCount") or 0),
            "page": 1,
            "page_size": 5,
            "has_more": False,
            "google_maps_uri": str(payload.get("googleMapsUri", "")),
            "reviews": reviews,
        }

    async def platform(self, *, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(50, int(page_size)))
        rating_body = {
            "size": 1000,
            "query": {"range": {"engagement.rating_count": {"gt": 0}}},
            "sort": [{"engagement.rating_count": "desc"}],
            "_source": [
                "title",
                "display_url",
                "engagement.rating_score",
                "engagement.rating_count",
            ],
        }
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=False,
            transport=self._transport,
        ) as client:
            rating_response = await client.post(
                f"{ELASTICSEARCH_URL}/{PLATFORM_REVIEWS_INDEX}/_search",
                auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
                json=rating_body,
            )
            rating_response.raise_for_status()
            hits = rating_response.json().get("hits", {}).get("hits", [])
            reviews_response = await client.get(
                PLATFORM_REVIEWS_URL,
                params={
                    "pageNum": page,
                    "pageSize": page_size,
                    "bizType": "trip",
                    "status": 1,
                },
            )
            reviews_response.raise_for_status()
            reviews_payload = reviews_response.json()
        rows = reviews_payload.get("rows", [])
        raw_total = reviews_payload.get("total", len(rows))
        if isinstance(raw_total, dict):
            raw_total = raw_total.get("value", len(rows))
        total = max(0, int(raw_total or 0))
        total_count = 0
        weighted_score = 0.0
        for hit in hits:
            engagement = (hit.get("_source") or {}).get("engagement") or {}
            count = int(engagement.get("rating_count") or 0)
            score = float(engagement.get("rating_score") or 0)
            if count > 0 and score > 0:
                total_count += count
                weighted_score += score * count
        rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("status", "")).strip() == "1"
            and str(row.get("description", "")).strip()
        ]
        review_ids = [f"trip:{row.get('bizId')}" for row in rows if row.get("bizId")]
        product_map: dict[str, dict[str, Any]] = {}
        if review_ids:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=False,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{ELASTICSEARCH_URL}/{PLATFORM_REVIEWS_INDEX}/_mget",
                    auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
                    json={"ids": review_ids},
                )
                response.raise_for_status()
                for document in response.json().get("docs", []):
                    if document.get("found"):
                        product_map[str(document.get("_id", ""))] = document.get("_source") or {}
        reviews = []
        for row in rows:
            author = row.get("author") or row.get("robotAuthor") or {}
            product = product_map.get(f"trip:{row.get('bizId')}", {})
            images = row.get("images") if isinstance(row.get("images"), list) else []
            reviews.append(
                {
                    "author_name": str(author.get("nickName") or author.get("nickname") or ""),
                    "author_photo_uri": str(author.get("avatar") or ""),
                    "rating": float(row.get("star") or 0),
                    "text": str(row.get("description") or ""),
                    "publish_time": str(row.get("createdAt") or ""),
                    "images": [
                        str(image.get("imageUrl", ""))
                        for image in images
                        if isinstance(image, dict) and image.get("imageUrl")
                    ],
                    "product_title": str(product.get("title") or row.get("tagTitle") or ""),
                    "display_url": str(product.get("display_url") or ""),
                }
            )
        return {
            "source": "platform",
            "rating": weighted_score / total_count if total_count else 0.0,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": page * page_size < total,
            "reviews": reviews,
        }
