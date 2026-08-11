import httpx
import pytest

from gta_ai.clients import LocalModelClient
from gta_ai.config import Settings
from gta_ai.schemas import HealthState, LocalGenerationRequest, ModelMessage


@pytest.mark.asyncio
async def test_probe_and_generate_use_openai_compatible_local_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "Qwen/Qwen3.6-27B-FP8"}]},
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "Qwen/Qwen3.6-27B-FP8",
                    "choices": [
                        {
                            "message": {"content": "分析完成"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(404)

    client = LocalModelClient(
        Settings.for_tests(),
        transport=httpx.MockTransport(handler),
    )

    health = await client.probe()
    result = await client.generate(
        LocalGenerationRequest(messages=[ModelMessage(role="user", content="分析数据")])
    )

    assert health.state is HealthState.OK
    assert result.content == "分析完成"
    assert result.finish_reason == "stop"
