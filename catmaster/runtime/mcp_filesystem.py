from __future__ import annotations

"""Runtime MCP filesystem integration with path-contract enforcement."""

from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from catmaster.llm.config import MCPFilesystemConfig
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_scoped_path, system_root, workspace_root

_READONLY_TOOL_NAMES = {
    "read_text_file",
    "read_multiple_files",
    "list_directory",
    "search_files",
    "directory_tree",
}
_WRITE_TOOL_NAMES = {
    "write_file",
    "edit_file",
    "create_directory",
    "move_file",
}
_ALWAYS_HIDE_TOOL_NAMES = {"read_media_file"}
_OFFLOAD_CANDIDATE_TOOLS = {
    "search_files",
    "list_directory",
    "directory_tree",
    "read_multiple_files",
}
_PATH_ARG_KEYS = {"path", "source", "destination"}
_PATH_LIST_ARG_KEYS = {"paths"}
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")
_DEFAULT_SKILL_MOUNT_NAMES = ("skills", "writing_skills")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _snippet(text: str, limit: int = 240) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    token = token.strip("._")
    return token or "mcp_tool"


@dataclass
class MCPFilesystemRuntime:
    """Run-scoped MCP filesystem runtime and tool surface provider."""

    config: MCPFilesystemConfig
    run_context: RunContext
    reporter: Any | None = None

    def __post_init__(self) -> None:
        self.files_root = workspace_root(self.run_context.workspace).resolve()
        repo_root = Path(__file__).resolve().parents[2]
        self.skill_mounts: dict[str, Path] = {}
        for root_name in _DEFAULT_SKILL_MOUNT_NAMES:
            candidate = (repo_root / root_name).resolve()
            if candidate.is_dir():
                self.skill_mounts[f"@{root_name}"] = candidate
        self.skills_root: Path | None = self.skill_mounts.get("@skills")
        self.server_name = str(self.config.server_name or "filesystem").strip() or "filesystem"
        self.model_root_token = str(self.config.model_root_token or ".").strip() or "."

        self._client: Any = None
        self._session_cm: Any = None
        self._session: Any = None
        self._tools: list[Any] = []
        self._tool_names: list[str] = []
        self._call_tool_result_cls: Any = None
        self._text_content_cls: Any = None
        self._files_root_abs = str(self.files_root)
        self._files_root_posix = self.files_root.as_posix()
        self._skill_mount_abs = {
            token: str(path)
            for token, path in self.skill_mounts.items()
        }
        self._skill_mount_posix = {
            token: path.as_posix()
            for token, path in self.skill_mounts.items()
        }

    async def __aenter__(self) -> "MCPFilesystemRuntime":
        self.config.validate()
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools
            from mcp.types import CallToolResult, TextContent
        except Exception as exc:  # pragma: no cover - runtime dependency failure
            raise RuntimeError(
                "MCP filesystem is enabled but MCP dependencies are unavailable. "
                "Install/verify `langchain-mcp-adapters` and MCP SDK."
            ) from exc

        self._call_tool_result_cls = CallToolResult
        self._text_content_cls = TextContent

        connection = self._build_connection()
        interceptors = [self._request_path_interceptor, self._response_rewrite_interceptor]

        if self.config.mode == "stateful":
            self._client = MultiServerMCPClient({self.server_name: connection})
            self._session_cm = self._client.session(self.server_name)
            try:
                self._session = await self._session_cm.__aenter__()
                self._tools = await load_mcp_tools(
                    self._session,
                    server_name=self.server_name,
                    tool_interceptors=interceptors,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Failed to launch MCP filesystem server command. "
                    f"Missing executable: {exc.filename or self.config.command}"
                ) from exc
        else:
            self._client = MultiServerMCPClient(
                {self.server_name: connection},
                tool_interceptors=interceptors,
            )
            self._tools = await self._client.get_tools(server_name=self.server_name)

        self._tool_names = [str(getattr(tool, "name", "") or "") for tool in self._tools]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(exc_type, exc, tb)
            finally:
                self._session_cm = None
                self._session = None
        self._tools = []
        self._tool_names = []

    async def build_tools(self) -> list[Any]:
        return list(self._tools)

    def role_filtered_tools(self, *, role: str) -> list[Any]:
        mode = str(self.config.expose_roles.get(role, "none") or "none").strip().lower()
        if mode == "none":
            return []

        allow_write = mode == "full"
        out: list[Any] = []
        for tool in self._tools:
            name = str(getattr(tool, "name", "") or "")
            if not name:
                continue
            if name in _ALWAYS_HIDE_TOOL_NAMES:
                continue
            if self.config.hide_list_allowed_directories and name == "list_allowed_directories":
                continue
            if allow_write:
                out.append(tool)
                continue
            if name in _READONLY_TOOL_NAMES:
                out.append(tool)
        return out

    def render_capability_guide(self, *, mode: str = "full") -> str:
        task_tools = self.role_filtered_tools(role="task_runner")
        names = [str(getattr(tool, "name", "") or "") for tool in task_tools]
        readonly = [name for name in names if name in _READONLY_TOOL_NAMES]
        writable = [name for name in names if name in _WRITE_TOOL_NAMES]

        if str(mode).strip().lower() == "short":
            parts: list[str] = []
            if readonly:
                parts.append(f"Filesystem read/discovery: {', '.join(readonly)}")
            if writable:
                parts.append(f"Filesystem write/edit: {', '.join(writable)}")
            return "\n".join(parts) if parts else "Filesystem MCP tools are disabled for this role."

        lines = ["Filesystem tools:"]
        if readonly:
            lines.append(f"- read/discovery: {', '.join(readonly)}")
        if writable:
            lines.append(f"- write/edit: {', '.join(writable)}")
        if not readonly and not writable:
            lines.append("- (disabled for this role)")
        lines.extend(
            [
                f"- all paths are relative to project files root `{self.model_root_token}`",
                f"- reference absolute files root (orientation only): `{self._files_root_posix}`",
                "- prefer relative paths in filesystem tool arguments; absolute paths are fallback-only",
                "- if absolute paths are used, they must stay under project files root",
                "- metadata paths are forbidden in filesystem tool arguments",
            ]
        )
        if self.skill_mounts:
            mounts = ", ".join(f"`{token}`" for token in sorted(self.skill_mounts))
            lines.insert(len(lines) - 3, f"- skills are mounted read-only under {mounts}")
        else:
            lines.insert(len(lines) - 3, "- skills mounts are unavailable in this invocation")
        return "\n".join(lines)

    def _build_connection(self) -> dict[str, Any]:
        transport = str(self.config.transport or "stdio").strip().lower()
        if transport == "stdio":
            args = [str(item) for item in self.config.args_prefix]
            args.append(str(self.files_root))
            for root in self.skill_mounts.values():
                try:
                    root.relative_to(self.files_root)
                except Exception:
                    args.append(str(root))
            return {
                "transport": "stdio",
                "command": str(self.config.command),
                "args": args,
                "env": self._stdio_env(),
            }
        payload: dict[str, Any] = {
            "transport": transport if transport != "streamable-http" else "streamable_http",
            "url": str(self.config.url or ""),
        }
        if self.config.headers:
            payload["headers"] = dict(self.config.headers)
        return payload

    def _stdio_env(self) -> dict[str, str]:
        env = {str(k): str(v) for k, v in os.environ.items()}
        npm_cache = (system_root(self.run_context.workspace) / "_npm_cache" / self.server_name).resolve()
        npm_cache.mkdir(parents=True, exist_ok=True)
        env["NPM_CONFIG_CACHE"] = str(npm_cache)
        env["npm_config_cache"] = str(npm_cache)
        return env

    def _path_error(self, *, tool_name: str, message: str, args: Mapping[str, Any]) -> CatMasterToolExecutionError:
        return CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=message,
            artifact={"error": message, "tool_args": _json_safe(args)},
            retryable=False,
            error_code="invalid_path",
        )

    def _is_write_tool(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in _WRITE_TOOL_NAMES

    def _resolve_skill_token_path(
        self,
        path_text: str,
        *,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> str:
        mount_token = next((token for token in self.skill_mounts if path_text == token or path_text.startswith(token + "/")), None)
        if mount_token is None:
            raise self._path_error(
                tool_name=tool_name,
                message=f"{tool_name}: invalid skill-mount path {path_text!r}.",
                args=args,
            )
        root = self.skill_mounts.get(mount_token)
        if root is None:
            raise self._path_error(
                tool_name=tool_name,
                message=f"{tool_name}: skills mount is unavailable (missing {mount_token} root).",
                args=args,
            )
        suffix = path_text[len(mount_token) :]
        relative = suffix.lstrip("/\\")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except Exception as exc:
            raise self._path_error(
                tool_name=tool_name,
                message=f"{tool_name}: invalid skills path {path_text!r}.",
                args=args,
            ) from exc
        if self._is_write_tool(tool_name):
            raise self._path_error(
                tool_name=tool_name,
                message=(
                    f"{tool_name}: {mount_token} is read-only; "
                    "write/edit/move/create operations are not allowed under skill mount."
                ),
                args=args,
            )
        return str(target)

    def _absolute_skill_mount_token(self, path_text: str) -> str | None:
        resolved = Path(path_text).resolve()
        for token, root in self.skill_mounts.items():
            try:
                resolved.relative_to(root)
                return token
            except Exception:
                continue
        return None

    def _resolve_model_relpath_to_abs(self, value: str, *, tool_name: str, args: Mapping[str, Any]) -> str:
        path_text = str(value or "").strip()
        if not path_text:
            raise self._path_error(
                tool_name=tool_name,
                message=f"{tool_name}: empty path is not allowed.",
                args=args,
            )
        if any(path_text == token or path_text.startswith(token + "/") for token in self.skill_mounts):
            return self._resolve_skill_token_path(path_text, tool_name=tool_name, args=args)
        if path_text.startswith("~"):
            raise self._path_error(
                tool_name=tool_name,
                message=(
                    f"{tool_name}: home-relative paths are not supported ({path_text!r}). "
                    "Use project-relative paths, or absolute paths under project files root."
                ),
                args=args,
            )
        if _WINDOWS_DRIVE_PREFIX.match(path_text):
            raise self._path_error(
                tool_name=tool_name,
                message=(
                    f"{tool_name}: unsupported drive-style path {path_text!r}. "
                    "Use project-relative paths, or absolute paths under project files root."
                ),
                args=args,
            )
        is_absolute = Path(path_text).is_absolute()
        skill_mount_token = self._absolute_skill_mount_token(path_text) if is_absolute else None
        if is_absolute and skill_mount_token is not None:
            if self._is_write_tool(tool_name):
                raise self._path_error(
                    tool_name=tool_name,
                    message=(
                        f"{tool_name}: {skill_mount_token} is read-only; "
                        "write/edit/move/create operations are not allowed under skill mount."
                    ),
                    args=args,
                )
            return str(Path(path_text).resolve())
        try:
            resolved = resolve_scoped_path(
                path_text,
                "files",
                workspace=self.run_context.workspace,
                must_exist=False,
            )
        except Exception as exc:
            if is_absolute:
                raise self._path_error(
                    tool_name=tool_name,
                    message=(
                        f"{tool_name}: absolute path must stay under project files root "
                        f"{self.model_root_token} (ref: {self._files_root_posix}), got {path_text!r}."
                    ),
                    args=args,
                ) from exc
            raise self._path_error(
                tool_name=tool_name,
                message=f"{tool_name}: invalid path {path_text!r}: {exc}",
                args=args,
            ) from exc
        return str(resolved)

    def _rewrite_request_args(self, *, tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        rewritten: dict[str, Any] = {}
        for key, value in args.items():
            if key in _PATH_ARG_KEYS and isinstance(value, str):
                rewritten[key] = self._resolve_model_relpath_to_abs(value, tool_name=tool_name, args=args)
                continue
            if key in _PATH_LIST_ARG_KEYS and isinstance(value, list):
                rewritten[key] = [
                    self._resolve_model_relpath_to_abs(item, tool_name=tool_name, args=args)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
                continue
            rewritten[key] = value
        return rewritten

    def _relativize_path_string(self, text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if any(raw == token or raw.startswith(token + "/") for token in self.skill_mounts):
            return raw
        if _WINDOWS_DRIVE_PREFIX.match(raw):
            return None
        p = Path(raw)
        if not p.is_absolute():
            return None
        for token, root in self.skill_mounts.items():
            try:
                rel_skill = p.resolve(strict=False).relative_to(root)
                rel_skill_text = rel_skill.as_posix()
                return token if rel_skill_text in {"", "."} else f"{token}/{rel_skill_text}"
            except Exception:
                continue
        try:
            rel = p.resolve(strict=False).relative_to(self.files_root)
        except Exception:
            return None
        if str(rel) in {".", ""}:
            return self.model_root_token
        return rel.as_posix()

    def _relativize_text(self, text: str) -> str:
        if not text:
            return text

        exact = self._relativize_path_string(text)
        if exact is not None:
            return exact

        updated = str(text)
        for token in self.skill_mounts:
            for root_variant in (self._skill_mount_abs.get(token, ""), self._skill_mount_posix.get(token, "")):
                if not root_variant:
                    continue
                updated = updated.replace(root_variant + "/", token + "/")
                updated = updated.replace(root_variant + "\\", token + "/")
                updated = updated.replace(root_variant, token)
        for root_variant in (self._files_root_abs, self._files_root_posix):
            if not root_variant:
                continue
            updated = updated.replace(root_variant + "/", self.model_root_token + "/")
            updated = updated.replace(root_variant + "\\", self.model_root_token + "/")
            updated = updated.replace(root_variant, self.model_root_token)
        return updated

    def _relativize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            exact = self._relativize_path_string(value)
            if exact is not None:
                return exact
            return self._relativize_text(value)
        if isinstance(value, Mapping):
            return {str(k): self._relativize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._relativize_value(v) for v in value]
        return value

    def _content_text(self, blocks: Sequence[Any]) -> str:
        lines: list[str] = []
        for block in blocks:
            if isinstance(block, Mapping):
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    lines.append(str(block.get("text") or ""))
                continue
            if hasattr(block, "type") and getattr(block, "type", "") == "text":
                lines.append(str(getattr(block, "text", "") or ""))
        return "\n".join(line for line in lines if line)

    def _make_text_block(self, text: str) -> Any:
        if self._text_content_cls is not None:
            return self._text_content_cls(type="text", text=text)
        return {"type": "text", "text": text}

    def _extract_path_like_strings(self, value: Any) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, Mapping):
                for item in node.values():
                    _walk(item)
                return
            if isinstance(node, list):
                for item in node:
                    _walk(item)
                return
            if not isinstance(node, str):
                return
            text = str(node).strip()
            if not text or "\n" in text:
                return
            if len(text) > 320:
                return
            if "/" not in text and text != self.model_root_token:
                return
            if text in seen:
                return
            seen.add(text)
            out.append(text)

        _walk(value)
        return out

    def _manifest_relpath(self, tool_name: str) -> str:
        token = _safe_token(tool_name)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        nonce = uuid4().hex[:8]
        rel = Path(self.config.offload.output_dir_rel) / f"{token}_{ts}_{nonce}.json"
        return rel.as_posix()

    def _write_manifest(self, *, tool_name: str, payload: Mapping[str, Any]) -> str:
        rel = self._manifest_relpath(tool_name)
        full = (self.files_root / rel).resolve()
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rel

    def _offload_summary(self, *, tool_name: str, request_args: Mapping[str, Any], content_text: str, structured: Any) -> str:
        if tool_name == "search_files":
            count = 0
            if isinstance(structured, Mapping):
                for key in ("count", "total_count", "returned"):
                    if isinstance(structured.get(key), int):
                        count = int(structured.get(key))
                        break
            if count <= 0:
                count = max(0, len(self._extract_path_like_strings(structured)))
            return f"Found {count} matching paths."
        if tool_name in {"list_directory"}:
            base = str(request_args.get("path") or self.model_root_token)
            count = len(self._extract_path_like_strings(structured))
            return f"Listed {count} entries under {base}."
        if tool_name == "directory_tree":
            return "Directory tree is large; full tree was offloaded."
        if tool_name == "read_multiple_files":
            count = 0
            if isinstance(structured, Mapping):
                for key in ("count", "files_count", "returned"):
                    if isinstance(structured.get(key), int):
                        count = int(structured.get(key))
                        break
            return f"read_multiple_files returned a large payload ({count} files)."
        return f"{tool_name} output was offloaded."

    def _preview_lines(self, *, tool_name: str, structured: Any, content_text: str) -> list[str]:
        if tool_name in {"search_files", "list_directory"}:
            paths = self._extract_path_like_strings(structured)
            if paths:
                return paths[: self.config.offload.max_paths_inline]
        lines = [line.strip() for line in content_text.splitlines() if line.strip()]
        return lines[: self.config.offload.max_lines_inline]

    def _should_offload(
        self,
        *,
        tool_name: str,
        structured: Any,
        content_text: str,
    ) -> bool:
        if tool_name not in _OFFLOAD_CANDIDATE_TOOLS:
            return False
        text_len = len(content_text or "")
        if tool_name in {"search_files", "list_directory"}:
            path_count = len(self._extract_path_like_strings(structured))
            if path_count > self.config.offload.max_paths_inline:
                return True
            return text_len > self.config.offload.max_chars_inline
        if tool_name == "directory_tree":
            return text_len > self.config.offload.max_tree_chars_inline
        if tool_name == "read_multiple_files":
            return text_len > self.config.offload.max_read_multiple_chars_inline
        return text_len > self.config.offload.max_chars_inline

    def _rewrite_call_tool_result(self, *, request: Any, result: Any) -> Any:
        tool_name = str(getattr(request, "name", "") or "filesystem_tool")
        raw_content = list(getattr(result, "content", []) or [])
        rewritten_content: list[Any] = []
        for block in raw_content:
            if isinstance(block, Mapping):
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    updated = dict(block)
                    updated["text"] = self._relativize_text(str(block.get("text", "") or ""))
                    rewritten_content.append(updated)
                    continue
                rewritten_content.append(self._relativize_value(block))
                continue

            block_type = str(getattr(block, "type", "") or "")
            if block_type == "text":
                text = self._relativize_text(str(getattr(block, "text", "") or ""))
                if hasattr(block, "model_copy"):
                    rewritten_content.append(block.model_copy(update={"text": text}))
                else:
                    rewritten_content.append(self._make_text_block(text))
                continue
            rewritten_content.append(block)

        rewritten_structured = self._relativize_value(getattr(result, "structuredContent", None))
        request_args_rel = self._relativize_value(getattr(request, "args", {}))

        if tool_name == "list_allowed_directories" and self.config.hide_list_allowed_directories:
            allowed = [self.model_root_token]
            allowed.extend(sorted(self.skill_mounts))
            rewritten_content = [self._make_text_block(f"Allowed roots: {', '.join(allowed)}")]
            rewritten_structured = {"allowed_directories": allowed}

        content_text = self._content_text(rewritten_content)
        if self._should_offload(
            tool_name=tool_name,
            structured=rewritten_structured,
            content_text=content_text,
        ):
            summary = self._offload_summary(
                tool_name=tool_name,
                request_args=request_args_rel if isinstance(request_args_rel, Mapping) else {},
                content_text=content_text,
                structured=rewritten_structured,
            )
            preview = self._preview_lines(
                tool_name=tool_name,
                structured=rewritten_structured,
                content_text=content_text,
            )
            manifest_ref = self._write_manifest(
                tool_name=tool_name,
                payload={
                    "tool_name": tool_name,
                    "request": _json_safe(request_args_rel),
                    "summary": summary,
                    "content_text": content_text,
                    "structured_content": _json_safe(rewritten_structured),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                },
            )
            rendered = [summary, f"Manifest: {manifest_ref}"]
            if preview:
                rendered.append("")
                rendered.append("Preview:")
                rendered.extend(f"- {_snippet(line, 240)}" for line in preview)
            rewritten_content = [self._make_text_block("\n".join(rendered))]
            rewritten_structured = {
                "offload_ref": manifest_ref,
                "summary": summary,
                "preview_count": len(preview),
            }

        return result.model_copy(
            update={
                "content": rewritten_content,
                "structuredContent": rewritten_structured,
            }
        )

    async def _request_path_interceptor(self, request: Any, handler: Any) -> Any:
        tool_name = str(getattr(request, "name", "") or "filesystem_tool")
        args = getattr(request, "args", {}) or {}
        if isinstance(args, Mapping):
            rewritten = self._rewrite_request_args(tool_name=tool_name, args=args)
            request = request.override(args=rewritten)
        return await handler(request)

    async def _response_rewrite_interceptor(self, request: Any, handler: Any) -> Any:
        result = await handler(request)
        if self._call_tool_result_cls is None:
            return result
        if not isinstance(result, self._call_tool_result_cls):
            return result
        return self._rewrite_call_tool_result(request=request, result=result)


__all__ = ["MCPFilesystemRuntime"]
