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
AgentRole = Literal[
    "proposal",
    "director",
    "task_runner",
    "research_lead",
    "research_state_updater",
    "write_director",
    "section_writer",
    "write_reviewer",
    "academic_polisher",
    "tex_compile_fixer",
    "memory_patch",
    "summary",
    "tool_selector",
    "image_analyzer",
    "literature_synthesizer",
    "literature_deep_research",
]

_DEFAULT_CONFIG_PATH = Path("configs/llm.yaml")
_logger = logging.getLogger(__name__)
REQUIRED_AGENT_ROLES: tuple[str, ...] = ("proposal", "director", "task_runner", "memory_patch", "summary")
OPTIONAL_AGENT_ROLE_FALLBACKS: dict[str, str] = {
    "research_lead": "director",
    "research_state_updater": "research_lead",
    "write_director": "research_lead",
    "section_writer": "task_runner",
    "write_reviewer": "summary",
    "academic_polisher": "summary",
    "tex_compile_fixer": "academic_polisher",
    "tool_selector": "task_runner",
    "image_analyzer": "task_runner",
    "literature_synthesizer": "director",
    "literature_deep_research": "literature_synthesizer",
}
AGENT_ROLES: tuple[AgentRole, ...] = (
    "proposal",
    "director",
    "task_runner",
    "research_lead",
    "research_state_updater",
    "write_director",
    "section_writer",
    "write_reviewer",
    "academic_polisher",
    "tex_compile_fixer",
    "memory_patch",
    "summary",
    "tool_selector",
    "image_analyzer",
    "literature_synthesizer",
    "literature_deep_research",
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
class AgentRuntimeConfig:
    recursion_limit: int = 300
    max_tool_calls: int = 60
    print_state_messages: bool = False
    print_http_raw_post: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRuntimeConfig":
        if not isinstance(data, dict):
            return cls()
        legacy_keys = [key for key in ("termination_mode", "strict_control_contract") if key in data]
        if legacy_keys:
            joined = ", ".join(legacy_keys)
            raise ValueError(
                f"agent_runtime no longer supports: {joined}. "
                "Remove these keys and use structured-response runtime defaults."
            )
        raw_limit = _to_int(data.get("recursion_limit"))
        if raw_limit is None:
            recursion_limit = cls.recursion_limit
        elif raw_limit <= 0:
            recursion_limit = 1_000_000
        else:
            recursion_limit = raw_limit
        raw_max_tool_calls = _to_int(data.get("max_tool_calls"))
        if raw_max_tool_calls is None:
            max_tool_calls = cls.max_tool_calls
        elif raw_max_tool_calls <= 0:
            max_tool_calls = cls.max_tool_calls
        else:
            max_tool_calls = raw_max_tool_calls
        return cls(
            recursion_limit=recursion_limit,
            max_tool_calls=max_tool_calls,
            print_state_messages=_to_bool(
                data.get("print_state_messages"),
                default=cls.print_state_messages,
                source="agent_runtime.print_state_messages",
            ),
            print_http_raw_post=_to_bool(
                data.get("print_http_raw_post"),
                default=cls.print_http_raw_post,
                source="agent_runtime.print_http_raw_post",
            ),
        )


_LITERATURE_ALLOWED_DEPTHS: tuple[str, ...] = ("none", "quick", "standard", "focused", "deep_report")


@dataclass
class LiteratureDepthBudgetConfig:
    agent_step_budget: int = 4
    search_limit: int = 6
    recommendation_limit: int = 0
    recommendation_seed_count: int = 0
    public_web_limit: int = 0
    use_public_web: bool = False

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        defaults: "LiteratureDepthBudgetConfig | None" = None,
    ) -> "LiteratureDepthBudgetConfig":
        base = defaults or cls()
        if not isinstance(data, dict):
            return cls(
                agent_step_budget=base.agent_step_budget,
                search_limit=base.search_limit,
                recommendation_limit=base.recommendation_limit,
                recommendation_seed_count=base.recommendation_seed_count,
                public_web_limit=base.public_web_limit,
                use_public_web=base.use_public_web,
            )
        return cls(
            agent_step_budget=max(1, _to_int(data.get("agent_step_budget")) or base.agent_step_budget),
            search_limit=max(0, _to_int(data.get("search_limit")) or base.search_limit),
            recommendation_limit=max(0, _to_int(data.get("recommendation_limit")) or base.recommendation_limit),
            recommendation_seed_count=max(
                0,
                _to_int(data.get("recommendation_seed_count")) or base.recommendation_seed_count,
            ),
            public_web_limit=max(0, _to_int(data.get("public_web_limit")) or base.public_web_limit),
            use_public_web=_to_bool(
                data.get("use_public_web"),
                default=base.use_public_web,
                source="literature.budgets.*.use_public_web",
            ),
        )


@dataclass
class LiteratureRuntimeConfig:
    auto_default_depth: str = "quick"
    role_auto_max: Dict[str, str] = field(
        default_factory=lambda: {
            "proposal": "standard",
            "director": "focused",
            "fast_director": "focused",
            "task_runner": "none",
            "research_lead": "deep_report",
            "section_writer": "standard",
            "write_director": "none",
            "write_reviewer": "none",
        }
    )
    public_web_on_search_failure: bool = True
    summary_key_paper_count: int = 5
    semantic_scholar_retry_429_attempts: int = 5
    semantic_scholar_retry_429_wait_seconds: float = 15.0
    budgets: Dict[str, LiteratureDepthBudgetConfig] = field(
        default_factory=lambda: {
            "none": LiteratureDepthBudgetConfig(
                agent_step_budget=1,
                search_limit=0,
                recommendation_limit=0,
                recommendation_seed_count=0,
                public_web_limit=0,
                use_public_web=False,
            ),
            "quick": LiteratureDepthBudgetConfig(
                agent_step_budget=4,
                search_limit=6,
                recommendation_limit=0,
                recommendation_seed_count=0,
                public_web_limit=5,
                use_public_web=False,
            ),
            "standard": LiteratureDepthBudgetConfig(
                agent_step_budget=6,
                search_limit=10,
                recommendation_limit=4,
                recommendation_seed_count=3,
                public_web_limit=3,
                use_public_web=True,
            ),
            "focused": LiteratureDepthBudgetConfig(
                agent_step_budget=8,
                search_limit=12,
                recommendation_limit=6,
                recommendation_seed_count=3,
                public_web_limit=5,
                use_public_web=True,
            ),
            "deep_report": LiteratureDepthBudgetConfig(
                agent_step_budget=16,
                search_limit=14,
                recommendation_limit=6,
                recommendation_seed_count=3,
                public_web_limit=5,
                use_public_web=True,
            ),
        }
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteratureRuntimeConfig":
        defaults = cls()
        if not isinstance(data, dict):
            return defaults

        auto_default_depth = str(data.get("auto_default_depth", defaults.auto_default_depth)).strip().lower()
        if auto_default_depth not in _LITERATURE_ALLOWED_DEPTHS:
            raise ValueError(
                "literature.auto_default_depth must be one of: "
                + ", ".join(_LITERATURE_ALLOWED_DEPTHS)
            )

        role_auto_max = dict(defaults.role_auto_max)
        role_auto_max_raw = data.get("role_auto_max")
        if isinstance(role_auto_max_raw, dict):
            for role, depth in role_auto_max_raw.items():
                role_text = str(role or "").strip()
                depth_text = str(depth or "").strip().lower()
                if not role_text:
                    continue
                if depth_text not in _LITERATURE_ALLOWED_DEPTHS:
                    raise ValueError(
                        f"literature.role_auto_max.{role_text} must be one of: "
                        + ", ".join(_LITERATURE_ALLOWED_DEPTHS)
                    )
                role_auto_max[role_text] = depth_text

        budgets = {
            depth: LiteratureDepthBudgetConfig.from_dict({}, defaults=budget)
            for depth, budget in defaults.budgets.items()
        }
        budgets_raw = data.get("budgets")
        if isinstance(budgets_raw, dict):
            for depth, budget_raw in budgets_raw.items():
                depth_text = str(depth or "").strip().lower()
                if depth_text not in budgets:
                    raise ValueError(
                        f"literature.budgets.{depth_text} must be one of: "
                        + ", ".join(_LITERATURE_ALLOWED_DEPTHS)
                    )
                budgets[depth_text] = LiteratureDepthBudgetConfig.from_dict(
                    budget_raw if isinstance(budget_raw, dict) else {},
                    defaults=budgets[depth_text],
                )

        return cls(
            auto_default_depth=auto_default_depth,
            role_auto_max=role_auto_max,
            public_web_on_search_failure=_to_bool(
                data.get("public_web_on_search_failure"),
                default=defaults.public_web_on_search_failure,
                source="literature.public_web_on_search_failure",
            ),
            summary_key_paper_count=max(
                1,
                _to_int(data.get("summary_key_paper_count")) or defaults.summary_key_paper_count,
            ),
            semantic_scholar_retry_429_attempts=max(
                0,
                _to_int(data.get("semantic_scholar_retry_429_attempts"))
                if _to_int(data.get("semantic_scholar_retry_429_attempts")) is not None
                else defaults.semantic_scholar_retry_429_attempts,
            ),
            semantic_scholar_retry_429_wait_seconds=max(
                0.0,
                _to_float(data.get("semantic_scholar_retry_429_wait_seconds"))
                if _to_float(data.get("semantic_scholar_retry_429_wait_seconds")) is not None
                else defaults.semantic_scholar_retry_429_wait_seconds,
            ),
            budgets=budgets,
        )

    def budget_for_depth(self, depth: str) -> LiteratureDepthBudgetConfig:
        return self.budgets.get(str(depth or "").strip().lower(), self.budgets[self.auto_default_depth])


@dataclass
class ImageGenerationConfig:
    model_label: str | None = None
    image_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageGenerationConfig":
        if not isinstance(data, dict):
            return cls()
        model_label = _to_str_or_none(data.get("model_label"))
        image_config = data.get("image_config")
        return cls(
            model_label=model_label,
            image_config=dict(image_config) if isinstance(image_config, dict) else {},
        )


@dataclass
class WritingRuntimeConfig:
    author_name: str = "CatMaster"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WritingRuntimeConfig":
        if not isinstance(data, dict):
            return cls()
        author_name = str(data.get("author_name", cls.author_name)).strip() or cls.author_name
        return cls(author_name=author_name)


@dataclass
class LLMConfig:
    provider: Provider = "openai"
    model: str = "gpt-5.2"

    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    reasoning: Dict[str, Any] = field(default_factory=dict)

    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    default_headers: Dict[str, str] = field(default_factory=dict)
    provider_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    print_http_raw_post: bool = False

    langchain_class: Optional[str] = None
    langchain_kwargs: Dict[str, Any] = field(default_factory=dict)

    timeout_s: Optional[float] = None
    max_retries: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        if not isinstance(data, dict):
            return cls()

        _reject_legacy_model_fields(data, model_label="<inline>")

        default_headers = data.get("default_headers") or {}
        provider_options = data.get("provider_options") or {}
        reasoning = data.get("reasoning") or {}
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
            reasoning=dict(reasoning) if isinstance(reasoning, dict) else {},
            frequency_penalty=_to_float(data.get("frequency_penalty")),
            presence_penalty=_to_float(data.get("presence_penalty")),
            api_key_env=_to_str_or_none(data.get("api_key_env")),
            api_key=_to_str_or_none(data.get("api_key")),
            base_url=_to_str_or_none(data.get("base_url")),
            default_headers=dict(default_headers) if isinstance(default_headers, dict) else {},
            provider_options=_normalize_provider_options(provider_options),
            print_http_raw_post=_to_bool(
                data.get("print_http_raw_post"),
                default=cls.print_http_raw_post,
                source="models.*.print_http_raw_post",
            ),
            langchain_class=_to_str_or_none(data.get("langchain_class")),
            langchain_kwargs=dict(langchain_kwargs) if isinstance(langchain_kwargs, dict) else {},
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
        if not isinstance(self.reasoning, dict):
            self.reasoning = {}
        if not self.reasoning.get("effort"):
            effort = os.getenv("CATMASTER_REASONING_EFFORT", "").strip()
            if effort:
                self.reasoning["effort"] = effort


@dataclass
class LLMProfile:
    """Role-routed LLM profile: named model configs + explicit role bindings."""

    models: Dict[str, LLMConfig] = field(default_factory=dict)
    agents: Dict[str, str] = field(default_factory=dict)
    agent_policies: AgentPoliciesConfig = field(default_factory=AgentPoliciesConfig)
    agent_runtime: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
    literature: LiteratureRuntimeConfig = field(default_factory=LiteratureRuntimeConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    writing: WritingRuntimeConfig = field(default_factory=WritingRuntimeConfig)

    def label_for_role(self, role: str) -> str:
        label = self.agents.get(role)
        if not label:
            fallback_role = OPTIONAL_AGENT_ROLE_FALLBACKS.get(role)
            if fallback_role:
                label = self.agents.get(fallback_role)
        if not label:
            raise ValueError(f"Missing model label binding for role: {role}")
        if label not in self.models:
            raise ValueError(f"Role {role} references unknown model label: {label}")
        return label

    def config_for_role(self, role: str) -> LLMConfig:
        return self.models[self.label_for_role(role)]

    @property
    def main(self) -> LLMConfig:
        return self.config_for_role("task_runner")

    @property
    def summary(self) -> LLMConfig:
        return self.config_for_role("summary")

    @property
    def tool_selector(self) -> LLMConfig:
        return self.config_for_role("tool_selector")

    @property
    def image_analyzer(self) -> LLMConfig:
        return self.config_for_role("image_analyzer")

    def config_for_image_generation(self) -> LLMConfig:
        label = str(self.image_generation.model_label or "").strip()
        if not label:
            return self.image_analyzer
        if label not in self.models:
            raise ValueError(f"image_generation.model_label references unknown model label: {label!r}")
        return self.models[label]

    @property
    def literature_synthesizer(self) -> LLMConfig:
        return self.config_for_role("literature_synthesizer")

    @property
    def literature_deep_research(self) -> LLMConfig:
        return self.config_for_role("literature_deep_research")

    @staticmethod
    def from_env() -> "LLMProfile":
        provider = os.getenv("CATMASTER_LLM_PROVIDER", "openai").strip().lower()
        model = os.getenv("CATMASTER_LLM_MODEL", "gpt-5.2").strip()
        if provider == "openrouter":
            api_key_env = "OPENROUTER_API_KEY"
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        else:
            api_key_env = os.getenv("CATMASTER_API_KEY_ENV", "OPENAI_API_KEY").strip()
            base_url = os.getenv("CATMASTER_BASE_URL", "").strip() or None
        temperature = _to_float(os.getenv("CATMASTER_TEMPERATURE", ""))
        reasoning_effort = os.getenv("CATMASTER_REASONING_EFFORT", "").strip() or None
        reasoning: Dict[str, Any] = {}
        if reasoning_effort:
            reasoning["effort"] = reasoning_effort

        main = LLMConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            temperature=temperature if temperature is not None else 0.0,
            reasoning=reasoning,
            api_key_env=api_key_env,
            base_url=base_url,
        )
        main.apply_env_fallbacks()
        label = main.model
        env_limit = _to_int(os.getenv("CATMASTER_RECURSION_LIMIT", ""))
        if env_limit is None:
            recursion_limit = AgentRuntimeConfig.recursion_limit
        elif env_limit <= 0:
            recursion_limit = 1_000_000
        else:
            recursion_limit = env_limit
        env_max_tool_calls = _to_int(os.getenv("CATMASTER_MAX_TOOL_CALLS", ""))
        if env_max_tool_calls is None:
            max_tool_calls = AgentRuntimeConfig.max_tool_calls
        elif env_max_tool_calls <= 0:
            max_tool_calls = AgentRuntimeConfig.max_tool_calls
        else:
            max_tool_calls = env_max_tool_calls
        raw_http_env = os.getenv("CATMASTER_PRINT_HTTP_RAW_POST")
        if raw_http_env is None or not str(raw_http_env).strip():
            print_http_raw_post = False
        else:
            print_http_raw_post = _to_bool(
                raw_http_env,
                default=False,
                source="CATMASTER_PRINT_HTTP_RAW_POST",
            )
        main.print_http_raw_post = print_http_raw_post
        return LLMProfile(
            models={label: main},
            agents={role: label for role in AGENT_ROLES},
            agent_policies=AgentPoliciesConfig(),
            agent_runtime=AgentRuntimeConfig(
                recursion_limit=recursion_limit,
                max_tool_calls=max_tool_calls,
                print_state_messages=False,
                print_http_raw_post=print_http_raw_post,
            ),
            literature=LiteratureRuntimeConfig(),
            image_generation=ImageGenerationConfig(),
            writing=WritingRuntimeConfig(),
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

                if "tool_calling_profiles" in raw:
                    raise ValueError(
                        f"LLM config no longer supports top-level 'tool_calling_profiles' (file: {config_path}). "
                        "Use models.*.provider_options instead."
                    )

                models_raw = raw.get("models")
                agents_raw = raw.get("agents")
                agent_policies_raw = raw.get("agent_policies")
                agent_runtime_raw = raw.get("agent_runtime")
                literature_raw = raw.get("literature")
                image_generation_raw = raw.get("image_generation")
                writing_raw = raw.get("writing")
                if not isinstance(models_raw, dict):
                    raise ValueError(f"LLM config requires top-level 'models' mapping: {config_path}")
                if not models_raw:
                    raise ValueError(f"LLM config 'models' cannot be empty: {config_path}")
                if not isinstance(agents_raw, dict):
                    raise ValueError(f"LLM config requires top-level 'agents' mapping: {config_path}")

                unknown_roles = sorted(set(agents_raw.keys()) - set(AGENT_ROLES))
                if unknown_roles:
                    joined = ", ".join(unknown_roles)
                    raise ValueError(f"Unknown role(s) in llm config agents: {joined}")
                missing_roles = [role for role in REQUIRED_AGENT_ROLES if role not in agents_raw]
                if missing_roles:
                    joined = ", ".join(missing_roles)
                    raise ValueError(f"Missing required role binding(s) in llm config agents: {joined}")

                models: Dict[str, LLMConfig] = {}
                for label_raw, item in models_raw.items():
                    label = str(label_raw).strip()
                    if not label:
                        raise ValueError("LLM config model labels must be non-empty strings")
                    if not isinstance(item, dict):
                        raise ValueError(f"LLM config model {label!r} must be a mapping")
                    _reject_legacy_model_fields(item, model_label=label)
                    cfg = LLMConfig.from_dict(item)
                    cfg.apply_env_fallbacks()
                    models[label] = cfg

                agents: Dict[str, str] = {}
                for role in REQUIRED_AGENT_ROLES:
                    bound = str(agents_raw.get(role, "")).strip()
                    if not bound:
                        raise ValueError(f"Role {role!r} must bind to a non-empty model label")
                    if bound not in models:
                        raise ValueError(f"Role {role!r} references unknown model label: {bound!r}")
                    agents[role] = bound
                for role, fallback_role in OPTIONAL_AGENT_ROLE_FALLBACKS.items():
                    bound = str(agents_raw.get(role, "")).strip()
                    if not bound:
                        bound = str(agents.get(fallback_role, "")).strip()
                    if not bound:
                        raise ValueError(
                            f"Role {role!r} must bind to a non-empty model label "
                            f"(or fallback role {fallback_role!r} must be configured)"
                        )
                    if bound not in models:
                        raise ValueError(f"Role {role!r} references unknown model label: {bound!r}")
                    agents[role] = bound

                image_generation_cfg = ImageGenerationConfig.from_dict(
                    image_generation_raw if isinstance(image_generation_raw, dict) else {}
                )
                image_generation_label = str(image_generation_cfg.model_label or "").strip()
                if image_generation_label and image_generation_label not in models:
                    raise ValueError(
                        "image_generation.model_label references unknown model label: "
                        f"{image_generation_label!r}"
                    )

                agent_runtime_cfg = AgentRuntimeConfig.from_dict(
                    agent_runtime_raw if isinstance(agent_runtime_raw, dict) else {}
                )
                if agent_runtime_cfg.print_http_raw_post:
                    for cfg in models.values():
                        cfg.print_http_raw_post = True

                return LLMProfile(
                    models=models,
                    agents=agents,
                    agent_policies=AgentPoliciesConfig.from_dict(agent_policies_raw if isinstance(agent_policies_raw, dict) else {}),
                    agent_runtime=agent_runtime_cfg,
                    literature=LiteratureRuntimeConfig.from_dict(
                        literature_raw if isinstance(literature_raw, dict) else {}
                    ),
                    image_generation=image_generation_cfg,
                    writing=WritingRuntimeConfig.from_dict(writing_raw if isinstance(writing_raw, dict) else {}),
                )
        return LLMProfile.from_env()


def _reject_legacy_model_fields(data: Dict[str, Any], *, model_label: str) -> None:
    if "tool_calling" in data:
        raise ValueError(
            f"LLM config model {model_label!r} no longer supports 'tool_calling'. "
            "Use 'provider_options' and 'reasoning' instead."
        )
    if "extra_body" in data:
        raise ValueError(
            f"LLM config model {model_label!r} no longer supports top-level 'extra_body'. "
            "Use 'provider_options.<provider>.extra_body' instead."
        )
    if "reasoning_effort" in data:
        raise ValueError(
            f"LLM config model {model_label!r} no longer supports 'reasoning_effort'. "
            "Use 'reasoning: { effort: ... }' instead."
        )


def _normalize_provider_options(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw_key, raw_options in value.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if isinstance(raw_options, dict):
            out[key] = dict(raw_options)
    return out


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


__all__ = [
    "LLMConfig",
    "LLMProfile",
    "ProposalPolicyConfig",
    "AgentPoliciesConfig",
    "AgentRuntimeConfig",
    "LiteratureDepthBudgetConfig",
    "LiteratureRuntimeConfig",
    "Provider",
    "AgentRole",
    "AGENT_ROLES",
]
