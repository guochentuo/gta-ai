from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from tests.web.app import create_web_router


@pytest.mark.asyncio
async def test_index_serves_static_chat_interface() -> None:
    app = FastAPI()
    app.include_router(create_web_router())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "GreenTourAI" in response.text
    assert "http://${ROUTER_HOST}:7100" in response.text
    assert 'MEDIA_CDN_PREFIX = "https://hk-cdn.greentourasia.com/"' in response.text
    assert "withKnowledgeImages" in response.text
    assert 'style="display:block;width:100%;max-width:720px;' in response.text
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in response.text
    assert (
        "const completeImages = [...html.matchAll(knowledgeImagePattern)].slice(0, 3);"
        in response.text
    )
    assert "if (canRenderEarly || completeImages.length === 3" in response.text
    assert "aspect-ratio:4/3" in response.text
    assert "message.knowledgeImages || []" in response.text
    assert "固定三格骨架" in response.text
    assert 'aria-label="图片加载中"' in response.text
    assert "@keyframes gta-image-skeleton" in response.text
    assert "x-oss-process=image/resize,m_fill,w_600,h_450/quality,q_85" in response.text
    assert "图片Markdown是内部传输格式" in response.text
    assert "剩余图片语法全部属于内部传输文本" in response.text
    assert "function stripImageTransportText" in response.text
    assert response.text.count("target.pending = false;") == 3
    assert "function requestWelcome()" in response.text
    assert 'new URLSearchParams(window.location.search).get("keyword")' in response.text
    assert "renderAssistantContent" in response.text
    assert "X-GTA-Retrieval-Images" not in response.text
    assert "knowledgeImageCount" in response.text
    assert "const canRenderEarly" in response.text
    assert 'const gtaImageMarker = "[[IMAGE_GROUP_1]]"' in response.text
    assert 'payload.type === "image_group"' in response.text
    assert "const firstVisibleContent" in response.text
    assert "首个正文 token 立即替换加载动画" in response.text
    assert "chat.messages.slice(-12)" not in response.text
    assert 'return [{ role: "user", content: message.text || "" }];' in response.text
    assert "没有模型标记时绝不自动插图" in response.text
    assert "markerIsTooLate" not in response.text
    assert "useAutomaticImagePosition" not in response.text
    assert "避免页面残留孤立的 > 或 -" in response.text
    assert "stream: true" in response.text
    assert 'if (data === "[DONE]")' in response.text
    assert "await reader.cancel();" in response.text
    assert 'console.info("[GTA AI Web] 用户发送"' in response.text
    assert 'console.info("[GTA AI Web] 模型返回"' in response.text
    assert 'fetch("/debug/chat-log"' in response.text


@pytest.mark.asyncio
async def test_chat_debug_log_is_printed_by_web_service(capsys: pytest.CaptureFixture[str]) -> None:
    app = FastAPI()
    app.include_router(create_web_router())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/debug/chat-log",
            json={"event": "模型返回", "content": "你好", "reasoning": "测试"},
        )

    assert response.status_code == 204
    output = capsys.readouterr().out
    assert "[GTA AI Web] 模型返回" in output
    assert '"content":"你好"' in output
