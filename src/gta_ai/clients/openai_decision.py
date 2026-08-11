from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from gta_ai.config import Settings
from gta_ai.schemas import DecisionPacket, FinalDecision


class OpenAIConfigurationError(RuntimeError):
    pass


class OpenAIDecisionClient:
    """External final-decision client using the OpenAI Responses API."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._settings.openai_api_key is None:
            raise OpenAIConfigurationError("GTA_AI_OPENAI_API_KEY is not configured")
        self._client = AsyncOpenAI(
            api_key=self._settings.openai_api_key.get_secret_value(),
            timeout=self._settings.openai_timeout_seconds,
        )
        return self._client

    async def decide(self, packet: DecisionPacket) -> FinalDecision:
        client = self._get_client()
        response = await client.responses.parse(
            model=self._settings.openai_model,
            reasoning={"effort": self._settings.openai_reasoning_effort},
            input=[
                {
                    "role": "developer",
                    "content": (
                        "Act as the final decision reviewer. Use only the supplied decision "
                        "packet, preserve numerical facts, reject unsupported findings, and "
                        "return the requested structured result."
                    ),
                },
                {"role": "user", "content": packet.model_dump_json()},
            ],
            text_format=FinalDecision,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI returned no parsed final decision")
        return response.output_parsed
