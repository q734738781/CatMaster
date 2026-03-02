from __future__ import annotations

import catmaster.runtime.tool_output_config as tool_output_config


def test_load_yaml_without_pyyaml_returns_empty_dict(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "tool_output.yaml"
    config_path.write_text(
        "offload:\n"
        "  preview_chars: 5000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tool_output_config, "yaml", None)

    raw = tool_output_config._load_config_file(config_path)
    assert raw == {}


def test_get_tool_output_config_yaml_without_pyyaml_falls_back_to_defaults(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "tool_output.yaml"
    config_path.write_text(
        "offload:\n"
        "  offload_chars: 42\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tool_output_config, "yaml", None)
    tool_output_config.get_tool_output_config.cache_clear()
    try:
        cfg = tool_output_config.get_tool_output_config(str(config_path))
        assert cfg.offload_chars == tool_output_config.ToolOutputConfig.offload_chars
        assert cfg.offload_dir_rel == tool_output_config.ToolOutputConfig.offload_dir_rel
    finally:
        tool_output_config.get_tool_output_config.cache_clear()


def test_tool_output_config_loads_preview_settings(tmp_path) -> None:
    config_path = tmp_path / "tool_output.yaml"
    config_path.write_text(
        "offload:\n"
        "  preview_chars: 3500\n",
        encoding="utf-8",
    )
    tool_output_config.get_tool_output_config.cache_clear()
    try:
        cfg = tool_output_config.get_tool_output_config(str(config_path))
        assert cfg.preview_chars == 3500
        assert cfg.offload_chars == tool_output_config.ToolOutputConfig.offload_chars
    finally:
        tool_output_config.get_tool_output_config.cache_clear()
