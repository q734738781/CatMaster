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
ToolCallingRole = Literal["proposal", "director", "task_runner"]
AgentRole = Literal["proposal", "director", "task_runner", "memory_patch", "summary"]

_DEFAULT_CONFIG_PATH = Path("configs/llm.yaml")
_logger = logging.getLogger(__name__)
TOOL_CALLING_AGENT_ROLES: tuple[ToolCallingRole, ...] = ("proposal", "director", "task_runner")
AGENT_ROLES: tuple[AgentRole, ...] = ("proposal", "director", "task_runner", "memory_patch", "summary")


@dataclass
class ToolCallingConfig:
    profile: Optional[str] = None
    driver: DriverKind = "openai_responses"
    parallel_tool_calls: bool = False
    supports_builtin_tools: bool = False
    strict_json_schema: bool = False
    request_options: Dict[str, Any] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallingConfig":
        if not isinstance(data, dict):
            return cls()
        request_options = data.get("request_options")
        extra_body = data.get("extra_body")
        return cls(
            profile=_to_str_or_none(data.get("profile")),
            driver=data.get("driver", cls.driver),  # type: ignore[arg-type]
            parallel_tool_calls=_to_bool(
                data.get("parallel_tool_calls"),
                default=cls.parallel_tool_calls,
                source="tool_calling.parallel_tool_calls",
            ),
            supports_builtin_tools=_to_bool(
                data.get("supports_builtin_tools"),
                default=cls.supports_builtin_tools,
                source="tool_calling.supports_builtin_tools",
            ),
            strict_json_schema=_to_bool(
                data.get("strict_json_schema"),
                default=cls.strict_json_schema,
                source="tool_calling.strict_json_schema",
            ),
            request_options=dict(request_options) if isinstance(request_options, dict) else {},
            extra_body=dict(extra_body) if isinstance(extra_body, dict) else {},
        )


@dataclass
class ProposalPolicyConfig:
    browse_tools_enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposalPolicyConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            browse_tools_enabled=_to_bool(
                data.get("browse_tools_enabled"),
                default=cls.browse_tools_enabled,
                source="agent_policies.proposal.browse_tools_enabled",
            )
        )


@dataclass
class AgentPoliciesConfig:
    proposal: ProposalPolicyConfig = field(default_factory=ProposalPolicyConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPoliciesConfig":
        if not isinstance(data, dict):
            return cls()
        proposal_raw = data.get("proposal")
        return cls(
            proposal=ProposalPolicyConfig.from_dict(proposal_raw if isinstance(proposal_raw, dict) else {}),
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
    """Role-routed LLM profile: named model configs + explicit role bindings."""

    models: Dict[str, LLMConfig] = field(default_factory=dict)
    agents: Dict[str, str] = field(default_factory=dict)
    tool_calling_profiles: Dict[str, ToolCallingConfig] = field(default_factory=dict)
    agent_policies: AgentPoliciesConfig = field(default_factory=AgentPoliciesConfig)

    def label_for_role(self, role: str) -> str:
        label = self.agents.get(role)
        if not label:
            raise ValueError(f"Missing model label binding for role: {role}")
        if label not in self.models:
            raise ValueError(f"Role {role} references unknown model label: {label}")
        return label

    def config_for_role(self, role: str) -> LLMConfig:
        return self.models[self.label_for_role(role)]

    @property
    def main(self) -> LLMConfig:
        # Task runner is the canonical "main" execution model in runtime metadata.
        return self.config_for_role("task_runner")

    @property
    def summary(self) -> LLMConfig:
        return self.config_for_role("summary")

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
        label = main.model
        return LLMProfile(
            models={label: main},
            agents={role: label for role in AGENT_ROLES},
            tool_calling_profiles={},
            agent_policies=AgentPoliciesConfig(),
        )

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
                models_raw = raw.get("models")
                agents_raw = raw.get("agents")
                profiles_raw = raw.get("tool_calling_profiles")
                agent_policies_raw = raw.get("agent_policies")
                if not isinstance(models_raw, dict):
                    raise ValueError(f"LLM config requires top-level 'models' mapping: {config_path}")
                if not models_raw:
                    raise ValueError(f"LLM config 'models' cannot be empty: {config_path}")
                if not isinstance(agents_raw, dict):
                    raise ValueError(f"LLM config requires top-level 'agents' mapping: {config_path}")
                if not isinstance(profiles_raw, dict):
                    raise ValueError(f"LLM config requires top-level 'tool_calling_profiles' mapping: {config_path}")
                if not profiles_raw:
                    raise ValueError(f"LLM config 'tool_calling_profiles' cannot be empty: {config_path}")

                unknown_roles = sorted(set(agents_raw.keys()) - set(AGENT_ROLES))
                if unknown_roles:
                    joined = ", ".join(unknown_roles)
                    raise ValueError(f"Unknown role(s) in llm config agents: {joined}")
                missing_roles = [role for role in AGENT_ROLES if role not in agents_raw]
                if missing_roles:
                    joined = ", ".join(missing_roles)
                    raise ValueError(f"Missing required role binding(s) in llm config agents: {joined}")

                tool_calling_profiles: Dict[str, ToolCallingConfig] = {}
                for name_raw, item in profiles_raw.items():
                    name = str(name_raw).strip()
                    if not name:
                        raise ValueError("tool_calling_profiles labels must be non-empty strings")
                    if not isinstance(item, dict):
                        raise ValueError(f"tool_calling_profiles[{name!r}] must be a mapping")
                    cfg = ToolCallingConfig.from_dict(item)
                    driver = str(cfg.driver or "").strip()
                    if not driver:
                        raise ValueError(f"tool_calling_profiles[{name!r}] requires non-empty driver")
                    cfg.profile = name
                    tool_calling_profiles[name] = cfg

                models: Dict[str, LLMConfig] = {}
                for label_raw, item in models_raw.items():
                    label = str(label_raw).strip()
                    if not label:
                        raise ValueError("LLM config model labels must be non-empty strings")
                    if not isinstance(item, dict):
                        raise ValueError(f"LLM config model {label!r} must be a mapping")
                    merged_model = dict(item)
                    tool_calling_raw = merged_model.get("tool_calling")
                    if not isinstance(tool_calling_raw, dict):
                        raise ValueError(f"LLM config model {label!r} requires tool_calling mapping")
                    profile_name = str(tool_calling_raw.get("profile") or "").strip()
                    if not profile_name:
                        raise ValueError(f"LLM config model {label!r} requires tool_calling.profile")
                    template = tool_calling_profiles.get(profile_name)
                    if template is None:
                        raise ValueError(f"LLM config model {label!r} references unknown tool_calling profile: {profile_name!r}")
                    merged_tool_calling = _merge_tool_calling_config(template, tool_calling_raw)
                    merged_model["tool_calling"] = _tool_calling_config_to_dict(merged_tool_calling)
                    cfg = LLMConfig.from_dict(merged_model)
                    cfg.apply_env_fallbacks()
                    cfg.tool_calling.profile = profile_name
                    models[label] = cfg

                agents: Dict[str, str] = {}
                for role in AGENT_ROLES:
                    bound = str(agents_raw.get(role, "")).strip()
                    if not bound:
                        raise ValueError(f"Role {role!r} must bind to a non-empty model label")
                    if bound not in models:
                        raise ValueError(f"Role {role!r} references unknown model label: {bound!r}")
                    agents[role] = bound

                for role in TOOL_CALLING_AGENT_ROLES:
                    cfg = models[agents[role]]
                    driver = str(getattr(cfg.tool_calling, "driver", "") or "").strip()
                    if not driver:
                        raise ValueError(f"Role {role!r} requires tool_calling.driver in model {agents[role]!r}")

                return LLMProfile(
                    models=models,
                    agents=agents,
                    tool_calling_profiles=tool_calling_profiles,
                    agent_policies=AgentPoliciesConfig.from_dict(agent_policies_raw if isinstance(agent_policies_raw, dict) else {}),
                )
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


def _to_bool(value: Any, *, default: bool, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    _logger.warning("Ignoring invalid %s=%r (allowed: true/false)", source, value)
    return default


def _merge_tool_calling_config(template: ToolCallingConfig, model_tool_calling_raw: Dict[str, Any]) -> ToolCallingConfig:
    if not isinstance(model_tool_calling_raw, dict):
        return template
    request_options = dict(template.request_options)
    request_options_raw = model_tool_calling_raw.get("request_options")
    if isinstance(request_options_raw, dict):
        request_options.update(request_options_raw)
    extra_body = dict(template.extra_body)
    extra_body_raw = model_tool_calling_raw.get("extra_body")
    if isinstance(extra_body_raw, dict):
        extra_body.update(extra_body_raw)
    return ToolCallingConfig(
        profile=template.profile,
        driver=model_tool_calling_raw.get("driver", template.driver),  # type: ignore[arg-type]
        parallel_tool_calls=_to_bool(
            model_tool_calling_raw.get("parallel_tool_calls"),
            default=template.parallel_tool_calls,
            source="models.*.tool_calling.parallel_tool_calls",
        ),
        supports_builtin_tools=_to_bool(
            model_tool_calling_raw.get("supports_builtin_tools"),
            default=template.supports_builtin_tools,
            source="models.*.tool_calling.supports_builtin_tools",
        ),
        strict_json_schema=_to_bool(
            model_tool_calling_raw.get("strict_json_schema"),
            default=template.strict_json_schema,
            source="models.*.tool_calling.strict_json_schema",
        ),
        request_options=request_options,
        extra_body=extra_body,
    )


def _tool_calling_config_to_dict(cfg: ToolCallingConfig) -> Dict[str, Any]:
    return {
        "profile": cfg.profile,
        "driver": cfg.driver,
        "parallel_tool_calls": cfg.parallel_tool_calls,
        "supports_builtin_tools": cfg.supports_builtin_tools,
        "strict_json_schema": cfg.strict_json_schema,
        "request_options": dict(cfg.request_options),
        "extra_body": dict(cfg.extra_body),
    }


__all__ = [
    "LLMConfig",
    "LLMProfile",
    "ToolCallingConfig",
    "ProposalPolicyConfig",
    "AgentPoliciesConfig",
    "Provider",
    "DriverKind",
    "ToolCallingRole",
    "AgentRole",
    "TOOL_CALLING_AGENT_ROLES",
    "AGENT_ROLES",
]
