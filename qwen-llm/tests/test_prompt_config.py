from pathlib import Path

import pytest
from router.prompt_config import load_prompt_catalog


def test_prompt_catalog_loads_all_model_prompt_sections() -> None:
    catalog = load_prompt_catalog()

    assert "GreenTourAsia" in catalog.persona
    assert "不得先生成中文再翻译" in catalog.language_match
    assert "summary" in catalog.summarization
    assert "do not translate literally" in catalog.welcome_localization
    assert "action_label" in catalog.handoff_localization
    assert "[[IMAGE_GROUP_1]]" in catalog.image_marker_context


def test_empty_prompt_configuration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompts.toml"
    path.write_text('[persona]\nsystem = ""\n', encoding="utf-8")

    with pytest.raises((KeyError, ValueError)):
        load_prompt_catalog(path)
