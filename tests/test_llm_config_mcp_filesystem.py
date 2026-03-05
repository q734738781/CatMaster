from __future__ import annotations

import pytest

from catmaster.llm.config import LLMProfile, MCPConfig, MCPFilesystemConfig


def test_mcp_filesystem_config_from_dict_enabled_stdio() -> None:
    cfg = MCPFilesystemConfig.from_dict(
        {
            "enabled": True,
            "transport": "stdio",
            "mode": "stateful",
            "server_name": "filesystem",
            "command": "npx",
            "args_prefix": ["-y", "@modelcontextprotocol/server-filesystem"],
            "model_root_token": ".",
            "hide_list_allowed_directories": True,
            "expose_roles": {
                "proposal": "readonly",
                "director": "readonly",
                "task_runner": "full",
            },
        }
    )

    assert cfg.enabled is True
    assert cfg.transport == "stdio"
    assert cfg.mode == "stateful"
    assert cfg.server_name == "filesystem"
    assert cfg.args_prefix[:2] == ["-y", "@modelcontextprotocol/server-filesystem"]
    assert cfg.expose_roles["task_runner"] == "full"
    assert cfg.expose_roles["memory_patch"] == "readonly"


def test_mcp_filesystem_config_invalid_expose_role_mode() -> None:
    with pytest.raises(ValueError):
        MCPFilesystemConfig.from_dict(
            {
                "enabled": True,
                "expose_roles": {"task_runner": "invalid_mode"},
            }
        )


def test_llm_profile_from_env_has_mcp_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATMASTER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CATMASTER_LLM_MODEL", "gpt-5-nano")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    profile = LLMProfile.from_env()
    assert isinstance(profile.mcp, MCPConfig)
    assert profile.mcp.filesystem.enabled is False
