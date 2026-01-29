from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Literal
import logging
import os

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

Provider = Literal["openai", "openrouter", "deepseek", "gemini", "oai_compatible", "langchain"]
DriverKind = Literal["openai_responses", "openai_chat_completions", "langchain_bind_tools"]

_DEFAULT_CONFIG_PATH = Path("configs/llm.yaml")
_logger = logging.getLogger(__name__)


@dataclass
class ToolCallingConfig:
    driver: DriverKind = "openai_responses"
    parallel_tool_calls: bool = False
    supports_builtin_tools: bool = False
    strict_json_schema: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallingConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            driver=data.get("driver", cls.driver),  # type: ignore[arg-type]
            parallel_tool_calls=bool(data.get("parallel_tool_calls", cls.parallel_tool_calls)),
            supports_builtin_tools=bool(data.get("supports_builtin_tools", cls.supports_builtin_tools)),
            strict_json_schema=bool(data.get("strict_json_schema", cls.strict_json_schema)),
        )


@dataclass
class LLMConfig:
    provider: Provider = "openai"
    model: str = "gpt-5.2"

    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None

    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    default_headers: Dict[str, str] = field(default_factory=dict)

    langchain_class: Optional[str] = None
    langchain_kwargs: Dict[str, Any] = field(default_factory=dict)

    tool_calling: ToolCallingConfig = field(default_factory=ToolCallingConfig)

    timeout_s: Optional[float] = None
    max_retries: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        if not isinstance(data, dict):
            return cls()
        tool_calling = ToolCallingConfig.from_dict(data.get("tool_calling", {}))
        default_headers = data.get("default_headers") or {}
        langchain_kwargs = data.get("langchain_kwargs") or {}
        extra = data.get("extra") or {}
        provider = _to_str_or_none(data.get("provider"))
        if provider:
            provider = provider.lower()
        return cls(
            provider=provider or None,  # type: ignore[arg-type]
            model=_to_str_or_none(data.get("model")) or "",
            temperature=_to_float(data.get("temperature")),
            top_p=_to_float(data.get("top_p")),
            max_tokens=_to_int(data.get("max_tokens")),
            max_output_tokens=_to_int(data.get("max_output_tokens")),
            reasoning_effort=_to_str_or_none(data.get("reasoning_effort")),
            frequency_penalty=_to_float(data.get("frequency_penalty")),
            presence_penalty=_to_float(data.get("presence_penalty")),
            api_key_env=_to_str_or_none(data.get("api_key_env")),
            api_key=_to_str_or_none(data.get("api_key")),
            base_url=_to_str_or_none(data.get("base_url")),
            default_headers=dict(default_headers) if isinstance(default_headers, dict) else {},
            langchain_class=_to_str_or_none(data.get("langchain_class")),
            langchain_kwargs=dict(langchain_kwargs) if isinstance(langchain_kwargs, dict) else {},
            tool_calling=tool_calling,
            timeout_s=_to_float(data.get("timeout_s")),
            max_retries=_to_int(data.get("max_retries")),
            extra=dict(extra) if isinstance(extra, dict) else {},
        )

    def apply_env_fallbacks(self) -> None:
        provider = self.provider or "openai"
        env_provider = os.getenv("CATMASTER_LLM_PROVIDER", "").strip().lower()
        if not self.provider:
            self.provider = env_provider or "openai"  # type: ignore[assignment]
            provider = self.provider
        if not self.model:
            model = os.getenv("CATMASTER_LLM_MODEL", "").strip()
            self.model = model or "gpt-5.2"
        if self.api_key_env is None:
            self.api_key_env = _default_api_key_env(provider)
        if self.base_url is None:
            env_base = os.getenv("CATMASTER_BASE_URL", "").strip()
            if env_base:
                self.base_url = env_base
            elif provider == "openrouter":
                base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()
                self.base_url = base_url or "https://openrouter.ai/api/v1"
        if provider == "openrouter" and not self.default_headers:
            referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
            title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
            headers: Dict[str, str] = {}
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
            if headers:
                self.default_headers = headers
        if self.temperature is None:
            temp = _to_float(os.getenv("CATMASTER_TEMPERATURE", ""))
            self.temperature = temp if temp is not None else 0.0
        if self.reasoning_effort is None:
            effort = os.getenv("CATMASTER_REASONING_EFFORT", "").strip()
            self.reasoning_effort = effort or None
        driver_env = os.getenv("CATMASTER_TOOL_DRIVER", "").strip()
        if driver_env:
            self.tool_calling.driver = driver_env  # type: ignore[assignment]
        elif provider == "openrouter" and not self.tool_calling.driver:
            self.tool_calling.driver = "openai_chat_completions"
        if provider == "openai" and self.tool_calling.driver == "openai_responses":
            self.tool_calling.supports_builtin_tools = True


@dataclass
class LLMProfile:
    """Orchestrator LLM profile: main + optional summary config."""

    main: LLMConfig = field(default_factory=LLMConfig)
    summary: Optional[LLMConfig] = None

    @staticmethod
    def from_env() -> "LLMProfile":
        provider = os.getenv("CATMASTER_LLM_PROVIDER", "openai").strip().lower()
        model = os.getenv("CATMASTER_LLM_MODEL", "gpt-5.2").strip()
        if provider == "openrouter":
            api_key_env = "OPENROUTER_API_KEY"
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
            driver = os.getenv("CATMASTER_TOOL_DRIVER", "openai_chat_completions").strip()
        else:
            api_key_env = os.getenv("CATMASTER_API_KEY_ENV", "OPENAI_API_KEY").strip()
            base_url = os.getenv("CATMASTER_BASE_URL", "").strip() or None
            driver = os.getenv("CATMASTER_TOOL_DRIVER", "openai_responses").strip()
        temperature = _to_float(os.getenv("CATMASTER_TEMPERATURE", ""))
        reasoning_effort = os.getenv("CATMASTER_REASONING_EFFORT", "").strip() or None
        main = LLMConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            temperature=temperature if temperature is not None else 0.0,
            reasoning_effort=reasoning_effort,
            api_key_env=api_key_env,
            base_url=base_url,
            tool_calling=ToolCallingConfig(
                driver=driver,  # type: ignore[arg-type]
                supports_builtin_tools=(provider == "openai" and driver == "openai_responses"),
            ),
        )
        main.apply_env_fallbacks()
        return LLMProfile(main=main, summary=None)

    @staticmethod
    def from_env_or_file(path: Optional[str] = None) -> "LLMProfile":
        config_path = Path(path) if path else Path(os.getenv("CATMASTER_LLM_CONFIG", str(_DEFAULT_CONFIG_PATH)))
        if config_path.exists():
            if yaml is None:
                _logger.warning("PyYAML not available; ignoring LLM config %s", config_path)
            else:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"LLM config must be a mapping: {config_path}")
                main_raw = raw.get("main") if "main" in raw else raw
                summary_raw = raw.get("summary")
                if isinstance(main_raw, dict):
                    provider = (main_raw.get("provider") or os.getenv("CATMASTER_LLM_PROVIDER", "openai")).strip().lower()
                    if provider == "openrouter":
                        tool_calling = main_raw.get("tool_calling")
                        if not isinstance(tool_calling, dict):
                            tool_calling = {}
                            main_raw["tool_calling"] = tool_calling
                        tool_calling.setdefault("driver", "openai_chat_completions")
                profile = LLMProfile(
                    main=LLMConfig.from_dict(main_raw if isinstance(main_raw, dict) else {}),
                    summary=LLMConfig.from_dict(summary_raw) if isinstance(summary_raw, dict) else None,
                )
                profile.main.apply_env_fallbacks()
                if profile.summary is not None:
                    profile.summary.apply_env_fallbacks()
                return profile
        return LLMProfile.from_env()


def _default_api_key_env(provider: str) -> str:
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    return "OPENAI_API_KEY"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _to_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "LLMConfig",
    "LLMProfile",
    "ToolCallingConfig",
    "Provider",
    "DriverKind",
]
