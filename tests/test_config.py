import pytest
from pydantic import SecretStr, ValidationError

from gta_ai.config import Settings


def test_default_settings_are_local_and_do_not_require_secrets() -> None:
    settings = Settings.for_tests()

    assert settings.host == "127.0.0.1"
    assert settings.local_llm_model == "Qwen/Qwen3.6-27B-FP8"
    assert settings.openai_model == "gpt-5.6-sol"
    assert settings.openai_is_configured is False


def test_secret_is_masked_and_detected() -> None:
    settings = Settings.for_tests(openai_api_key="test-secret")

    assert isinstance(settings.openai_api_key, SecretStr)
    assert str(settings.openai_api_key) == "**********"
    assert settings.openai_is_configured is True


def test_invalid_port_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.for_tests(port=70000)
