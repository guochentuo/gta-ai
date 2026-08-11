from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model for payloads that must reject unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)
