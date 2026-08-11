from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from gta_ai.schemas.base import StrictModel


class ChatTextPart(StrictModel):
    type: Literal["text"]
    text: str = Field(min_length=1, max_length=100_000)


class ChatImageUrl(StrictModel):
    url: str = Field(min_length=1, max_length=15_000_000)

    @field_validator("url")
    @classmethod
    def supported_image_url(cls, value: str) -> str:
        if value.startswith(("data:image/", "http://", "https://")):
            return value
        raise ValueError("image URL must use data:image, http, or https")


class ChatImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: ChatImageUrl


ChatContentPart = Annotated[ChatTextPart | ChatImagePart, Field(discriminator="type")]


class BrowserChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ChatContentPart]

    @model_validator(mode="after")
    def content_is_not_empty(self) -> BrowserChatMessage:
        if isinstance(self.content, str):
            if not self.content.strip():
                raise ValueError("message content must not be empty")
        elif not self.content:
            raise ValueError("message content must not be empty")
        return self


class BrowserChatRequest(StrictModel):
    messages: list[BrowserChatMessage] = Field(min_length=1, max_length=100)
    enable_thinking: bool = False
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0, le=2)
