from __future__ import annotations

"""Apply memory edits in Aider SEARCH/REPLACE format."""

import difflib
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_workspace_path


class MemoryApplyAiderEditsInput(BaseModel):
    """Apply Aider SEARCH/REPLACE edit blocks to memory files."""

    edits_text: str = Field(..., description="Aider edit blocks text.")
    allowed_paths: List[str] = Field(
        default_factory=lambda: ["MEMORY/", "notes/"],
        description="Allowed project-files-relative path prefixes.",
    )
    emit_diff: bool = Field(True, description="Whether to return unified diff text for audit.")


def _normalize_allowed_prefixes(raw: List[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for item in raw:
        text = str(item or "").strip().replace("\\", "/")
        if not text:
            continue
        if not text.endswith("/"):
            text = f"{text}/"
        out.append(text)
    return tuple(out)


def _normalize_rel_path(raw_path: str, allowed_prefixes: Tuple[str, ...]) -> str:
    path = str(raw_path or "").strip().replace("\\", "/")
    if not path:
        raise ValueError("empty edit path")
    pure = Path(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"forbidden path: {path}")
    normalized = str(pure).replace("\\", "/")
    if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(f"forbidden path: {normalized}")
    return normalized


def _render_git_style_diff(before: str, after: str, rel_path: str) -> str:
    if before == after:
        return ""
    lines: List[str] = [f"diff --git a/{rel_path} b/{rel_path}\n"]
    lines.extend(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )
    return "".join(lines)


def _fail(
    *,
    message: str,
    error_code: str,
    error_detail: str | None = None,
    failed_path: str = "",
    failed_block_index: int = 0,
) -> Dict[str, object]:
    return create_tool_output(
        "memory_apply_aider_edits",
        success=False,
        error=message,
        data={
            "error_code": str(error_code or "").strip(),
            "error_detail": str(error_detail or message).strip(),
            "failed_path": str(failed_path or "").strip(),
            "failed_block_index": int(failed_block_index or 0),
        },
    )


def memory_apply_aider_edits(payload: Dict[str, object]) -> Dict[str, object]:
    params = MemoryApplyAiderEditsInput(**payload)
    edits_text = str(params.edits_text or "").strip()
    if not edits_text:
        return _fail(
            message="edits_text is empty",
            error_code="empty_input",
        )

    try:
        from aider.coders.editblock_coder import (  # type: ignore
            DEFAULT_FENCE,
            do_replace,
            find_original_update_blocks,
        )
    except Exception as exc:
        return _fail(
            message=f"aider-chat import failed: {type(exc).__name__}: {exc}",
            error_code="import_failed",
        )

    allowed_prefixes = _normalize_allowed_prefixes(params.allowed_paths)
    if not allowed_prefixes:
        return _fail(
            message="allowed_paths is empty",
            error_code="invalid_config",
        )

    try:
        parsed_blocks = list(find_original_update_blocks(edits_text, DEFAULT_FENCE, None))
    except Exception as exc:
        return _fail(
            message=f"invalid aider edit blocks: {type(exc).__name__}: {exc}",
            error_code="parse_failed",
        )

    if not parsed_blocks:
        return _fail(
            message="no aider edit blocks found",
            error_code="no_blocks",
        )

    staged_content: Dict[str, str] = {}
    existed_before: Dict[str, bool] = {}
    original_content: Dict[str, str] = {}
    touched_order: List[str] = []

    for idx, block in enumerate(parsed_blocks, start=1):
        if not isinstance(block, tuple):
            return _fail(
                message="unsupported aider block payload",
                error_code="unsupported_block",
                failed_block_index=idx,
            )

        # Aider may yield (None, shell_command) for fenced shell command blocks.
        if len(block) == 2 and block[0] is None:
            return _fail(
                message="shell command blocks are not allowed in memory edits",
                error_code="shell_block_forbidden",
                failed_block_index=idx,
            )

        if len(block) != 3:
            return _fail(
                message="unsupported aider block shape",
                error_code="invalid_block_shape",
                failed_block_index=idx,
            )

        raw_path, before_text, after_text = block
        rel_path = ""
        abs_path: Path | None = None
        try:
            rel_path = _normalize_rel_path(str(raw_path or ""), allowed_prefixes)
            abs_path = resolve_workspace_path(rel_path, must_exist=False)
        except Exception as exc:
            return _fail(
                message=f"path validation failed: {exc}",
                error_code="path_forbidden",
                error_detail=str(exc),
                failed_path=str(raw_path or ""),
                failed_block_index=idx,
            )

        if rel_path not in original_content:
            assert abs_path is not None
            file_exists = abs_path.exists()
            current_text = abs_path.read_text(encoding="utf-8") if file_exists else ""
            original_content[rel_path] = current_text
            staged_content[rel_path] = current_text
            existed_before[rel_path] = file_exists
            touched_order.append(rel_path)

        current_text = staged_content[rel_path]
        try:
            updated_text = do_replace(
                rel_path,
                current_text,
                str(before_text or ""),
                str(after_text or ""),
                DEFAULT_FENCE,
            )
        except Exception as exc:
            return _fail(
                message=f"aider replace failed for {rel_path}: {type(exc).__name__}: {exc}",
                error_code="replace_failed",
                error_detail=f"{type(exc).__name__}: {exc}",
                failed_path=rel_path,
                failed_block_index=idx,
            )

        if updated_text is None:
            return _fail(
                message=f"aider replace did not match for {rel_path}",
                error_code="replace_no_match",
                failed_path=rel_path,
                failed_block_index=idx,
            )

        staged_content[rel_path] = updated_text

    applied_files: List[str] = []
    created_files: List[str] = []
    diff_chunks: List[str] = []
    written_paths: List[str] = []
    for rel_path in touched_order:
        before_text = original_content.get(rel_path, "")
        after_text = staged_content.get(rel_path, before_text)
        if before_text == after_text:
            continue
        abs_path = resolve_workspace_path(rel_path, must_exist=False)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            abs_path.write_text(after_text, encoding="utf-8")
            written_paths.append(rel_path)
        except Exception as exc:
            rollback_errors: List[str] = []
            for restore_rel in reversed(written_paths):
                restore_abs = resolve_workspace_path(restore_rel, must_exist=False)
                try:
                    if existed_before.get(restore_rel, False):
                        restore_abs.parent.mkdir(parents=True, exist_ok=True)
                        restore_abs.write_text(original_content.get(restore_rel, ""), encoding="utf-8")
                    else:
                        if restore_abs.exists():
                            restore_abs.unlink()
                except Exception as restore_exc:
                    rollback_errors.append(f"{restore_rel}: {type(restore_exc).__name__}: {restore_exc}")
            detail = f"{type(exc).__name__}: {exc}"
            if rollback_errors:
                detail = f"{detail} | rollback_errors={' ; '.join(rollback_errors)}"
            return _fail(
                message=f"write failed for {rel_path}: {type(exc).__name__}: {exc}",
                error_code="write_failed",
                error_detail=detail,
                failed_path=rel_path,
            )
        applied_files.append(rel_path)
        if not existed_before.get(rel_path, True):
            created_files.append(rel_path)
        if params.emit_diff:
            chunk = _render_git_style_diff(before_text, after_text, rel_path)
            if chunk:
                diff_chunks.append(chunk)

    return create_tool_output(
        "memory_apply_aider_edits",
        success=True,
        data={
            "applied_files": applied_files,
            "created_files": created_files,
            "diff_text": "".join(diff_chunks),
            "edit_format": "aider_search_replace",
            "edit_block_count": len(parsed_blocks),
        },
    )


__all__ = ["memory_apply_aider_edits", "MemoryApplyAiderEditsInput"]
