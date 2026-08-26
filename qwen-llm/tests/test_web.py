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
    assert "GTA AI" in response.text
    assert ":7100/v1/chat/completions" in response.text
    assert "stream: true" in response.text
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
