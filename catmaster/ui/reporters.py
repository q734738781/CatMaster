from __future__ import annotations

import sys
import threading
import time
import textwrap
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional
import termios
import tty
import select
import os

from .bus import UIBus, SENTINEL
from .events import UIEvent


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _truncate(value: Any, max_len: int = 160) -> str:
    text = _clean_text(value)
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _wrap_lines(value: Any, width: int, max_lines: int) -> List[str]:
    text = _clean_text(value)
    if not text:
        return []
    lines: List[str] = []
    for chunk in text.split("\n"):
        wrapped = textwrap.wrap(chunk, width=width) or [""]
        lines.extend(wrapped)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = _truncate(lines[-1], width)
    return lines


def _compact_params(params: Any, max_items: int = 4, max_len: int = 140) -> str:
    if not isinstance(params, dict):
        return _truncate(params, max_len)
    parts = []
    for key in list(params.keys())[:max_items]:
        val = params.get(key)
        if isinstance(val, (str, int, float, bool)):
            sval = str(val)
        elif isinstance(val, list):
            sval = f"list[{len(val)}]"
        elif isinstance(val, dict):
            sval = f"dict[{len(val)}]"
        else:
            sval = type(val).__name__
        parts.append(f"{key}={sval}")
    return _truncate(", ".join(parts), max_len)


def _summarize_event(event: UIEvent) -> str:
    name = event.name
    payload = event.payload or {}
    if name == "RUN_INIT_DONE":
        run_id = event.run_id or payload.get("run_id", "")
        model = payload.get("model_name", "")
        return f"run_id={run_id} model={model}".strip()
    if name == "PLAN_CREATED":
        n_items = payload.get("n_items")
        return f"{n_items} tasks" if n_items is not None else "Plan created"
    if name == "PLAN_REVIEW_REVISED":
        n_items = payload.get("n_items")
        return f"Plan revised ({n_items} tasks)" if n_items is not None else "Plan revised"
    if name == "TASK_START":
        goal = payload.get("goal", "")
        return f"{event.task_id}: {_truncate(goal, 120)}".strip()
    if name == "TASK_END":
        outcome = payload.get("outcome", "")
        return f"{event.task_id}: outcome={outcome}".strip()
    if name == "TASK_DECISION":
        action = payload.get("action", "")
        method = payload.get("method", "")
        params = payload.get("params_compact", "")
        parts = [action]
        if method:
            parts.append(method)
        if params:
            parts.append(params)
        return " | ".join(parts)
    if name == "TOOL_CALL_START":
        tool = payload.get("tool", "")
        params = payload.get("params_compact", "")
        return f"{tool} {params}".strip()
    if name == "TOOL_CALL_END":
        tool = payload.get("tool", "")
        status = payload.get("status", "")
        return f"{tool} status={status}".strip()
    if name == "TASK_SUMMARY":
        outcome = payload.get("outcome", "")
        summary = payload.get("summary_snippet", "")
        return f"{outcome} - {_truncate(summary, 120)}".strip()
    if name == "FINAL_SUMMARY_DONE":
        return "Final report generated"
    if name == "RUN_END":
        status = payload.get("status", "")
        return f"status={status}".strip()
    return name


def _format_plain_line(event: UIEvent) -> str:
    ts = datetime.fromtimestamp(event.ts).strftime("%H:%M:%S")
    summary = _summarize_event(event)
    return f"{ts} {event.name} {summary}".rstrip()


class Reporter:
    ui_debug: bool = False

    def start(self) -> None:
        pass

    def emit(self, event: UIEvent) -> None:
        pass

    def close(self) -> None:
        pass

    def is_live(self) -> bool:
        return False

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def prompt_plan_feedback(self, *, todo: List[str], plan_description: str) -> str:
        return ""

    def prompt_hitl_feedback(self, *, report_text: str, report_path: str) -> str:
        return ""

    def show_final_summary(self, summary: str) -> None:
        return


class NullReporter(Reporter):
    def emit(self, event: UIEvent) -> None:
        return


class PlainConsoleReporter(Reporter):
    def __init__(self, *, ui_debug: bool = False, max_feed: int = 200):
        self.ui_debug = ui_debug
        self._bus = UIBus(maxsize=max_feed)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, event: UIEvent) -> None:
        self._bus.emit(event)

    def close(self) -> None:
        self._bus.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._running.clear()

    def _run(self) -> None:
        while self._running.is_set():
            item = self._bus.get(timeout=0.2)
            if item is SENTINEL:
                break
            if item is None:
                continue
            line = _format_plain_line(item)
            print(line, flush=True)

    def prompt_plan_feedback(self, *, todo: List[str], plan_description: str) -> str:
        print("\n=== Proposed Plan ===")
        print("Plan Description:")
        print(plan_description.strip() if plan_description.strip() else "(none)")
        print("\nTherefore, Here is a proposed plan:")
        for i, item in enumerate(todo, start=1):
            print(f"{i}. {item}")
        print("\nEnter feedback to revise, or type 'yes' to approve:")
        return input("> ").strip()

    def prompt_hitl_feedback(self, *, report_text: str, report_path: str) -> str:
        print("\n=== Intervention Required ===")
        if report_path:
            print(f"Interrupted report: {report_path}")
        print("\n--- Interrupted Report ---")
        print(report_text.strip() if report_text.strip() else "(none)")
        print("\nEnter feedback to continue, or press Enter to keep paused:")
        return input("> ").strip()

    def show_final_summary(self, summary: str) -> None:
        print("\n=== Final Summary ===")
        print(summary or "(none)")
        input("\nPress Enter to exit...")


@dataclass
class UIState:
    start_time: Optional[float] = None
    run_id: str = ""
    run_dir: str = ""
    model_name: str = ""
    phase: str = "INIT"
    activity: str = ""
    status: str = ""
    tasks: Dict[str, Dict[str, str]] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    plan_todo: List[str] = field(default_factory=list)
    plan_description: str = ""
    current_task_id: str = ""
    current_task_goal: str = ""
    current_instruction: str = ""
    last_decision: str = ""
    last_tool: Dict[str, Any] = field(default_factory=dict)
    whiteboard_counts: Dict[str, int] = field(default_factory=dict)
    last_summary: str = ""
    last_llm_kind: str = ""
    last_llm_elapsed_ms: Optional[int] = None
    last_llm_snippet: str = ""
    llm_log_path: str = ""
    event_feed: Deque[str] = field(default_factory=lambda: deque(maxlen=30))
    splash_logo: str = ""
    splash_tagline: str = ""
    splash_subtitle: str = ""


class RichLiveReporter(Reporter):
    def __init__(
        self,
        *,
        ui_debug: bool = False,
        show_splash: bool = True,
        refresh_per_second: int = 2,
        max_feed: int = 30,
    ) -> None:
        self.ui_debug = ui_debug
        self._bus = UIBus(maxsize=200)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._pause = threading.Event()
        self._paused = threading.Event()
        self._show_splash = show_splash
        self._refresh = refresh_per_second
        self._state = UIState(event_feed=deque(maxlen=max_feed))
        self._console = None
        self._cleared = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, event: UIEvent) -> None:
        self._bus.emit(event)

    def close(self) -> None:
        self._bus.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._running.clear()

    def is_live(self) -> bool:
        return True

    def pause(self) -> None:
        if not self._thread or not self._thread.is_alive():
            return
        self._pause.set()
        self._paused.wait(timeout=2.0)

    def resume(self) -> None:
        self._pause.clear()

    def _run(self) -> None:
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.align import Align
            from rich import box
        except Exception as exc:
            print(f"Rich UI unavailable: {exc}", file=sys.stderr)
            return

        if self._console is None:
            self._console = Console()
        console = self._console

        def render_splash() -> Panel:
            width = console.size.width
            logo = self._fit_logo(self._state.splash_logo or "", width)
            subtitle = self._state.splash_subtitle or "Initializing... Please wait..."
            spinner = self._spinner()

            logo_lines = [line.rstrip() for line in logo.splitlines() if line.strip()]
            lines: List[str] = []
            lines.extend([""] * 1)
            lines.extend(logo_lines)
            lines.extend([""] * 1)
            lines.append(f"{spinner} {subtitle}")
            lines.extend([""] * 1)

            text = "\n".join(lines)
            title = self._state.splash_tagline or "CatMaster"
            return Panel(
                Align.center(Text(text), vertical="middle"),
                title=title,
                padding=(1, 2),
            )

        def render_dashboard() -> Layout:
            layout = Layout()
            layout.split(
                Layout(name="header", size=3),
                Layout(name="body", ratio=1),
                Layout(name="footer", size=9),
            )
            layout["body"].split_row(
                Layout(name="left", ratio=2),
                Layout(name="right", ratio=1),
            )
            layout["left"].split(
                Layout(name="tasks", ratio=2),
                Layout(name="current", ratio=3),
            )
            if self.ui_debug:
                layout["right"].split(
                    Layout(name="tool", ratio=2),
                    Layout(name="whiteboard", ratio=1),
                    Layout(name="debug", ratio=1),
                )
            else:
                layout["right"].split(
                    Layout(name="tool", ratio=2),
                    Layout(name="whiteboard", ratio=1),
                )

            elapsed_str = ""
            if self._state.start_time is not None:
                elapsed = int(time.time() - self._state.start_time)
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            run_id = self._state.run_id or "(pending)"
            phase = self._state.phase
            activity = self._state.activity
            spinner = self._spinner() if activity else ""
            header_text = f"{spinner} {phase}".strip()
            if activity:
                header_text += f" | {activity}"
            header_text += f" | run_id={run_id} | model={self._state.model_name}"
            if elapsed_str:
                header_text += f" | elapsed={elapsed_str}"
            layout["header"].update(Panel(Text(header_text), box=box.SQUARE))

            body_height = max(6, console.size.height - 3 - 9)
            tasks_height = max(4, int(body_height * 2 / 5))
            max_rows = max(3, tasks_height - 3)
            layout["tasks"].update(self._render_tasks(Panel, Table, Text, box, max_rows=max_rows))
            layout["current"].update(self._render_current(Panel, Text, console, box))
            layout["tool"].update(self._render_tool(Panel, Text, box))
            layout["whiteboard"].update(self._render_whiteboard(Panel, Text, box))
            if self.ui_debug:
                layout["debug"].update(self._render_debug(Panel, Text, box))

            feed_text = Text("\n".join(self._state.event_feed))
            layout["footer"].update(Panel(feed_text, title="Events", box=box.SQUARE))
            return layout

        if self._show_splash and not self._cleared:
            console.clear()
            self._cleared = True
        with Live(
            render_splash() if self._show_splash else render_dashboard(),
            console=console,
            refresh_per_second=self._refresh,
            screen=True,
        ) as live:
            while self._running.is_set():
                if self._pause.is_set():
                    live.stop()
                    self._paused.set()
                    while self._pause.is_set() and self._running.is_set():
                        time.sleep(0.05)
                    self._paused.clear()
                    if not self._running.is_set():
                        break
                    live.start()

                item = self._bus.get(timeout=0.2)
                if item is SENTINEL:
                    break
                if item is not None:
                    self._handle_event(item)

                renderable = render_splash() if self._show_splash else render_dashboard()
                live.update(renderable, refresh=True)

    def prompt_plan_feedback(self, *, todo: List[str], plan_description: str) -> str:
        try:
            import os
            from rich.console import Console
            from rich.live import Live
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.text import Text
            from rich.align import Align
            from rich import box
        except Exception as exc:
            print(f"Rich UI unavailable: {exc}", file=sys.stderr)
            return ""

        self.pause()
        try:
            cols = os.get_terminal_size(sys.stdout.fileno()).columns
        except Exception:
            cols = None
        safe_cols = max(40, (cols - 2)) if cols else None
        console = Console(width=safe_cols, force_terminal=True)

        scroll = 0
        scroll_max = 0
        input_buf = ""
        plan_lines: List[str] = []
        page_size = 10
        last_key = ""

        def viewport() -> tuple[int, int]:
            sz = console.size
            w, h = sz.width, sz.height

            w = max(40, int(w or 0))
            h = max(10, int(h or 0))
            if h > 200:
                h = 60
            return w, h

        def build_lines(inner_width: int) -> List[str]:
            inner_width = max(20, inner_width)
            lines: List[str] = []
            lines.append("Plan Description:")
            lines.extend(_wrap_lines(plan_description.strip() or "(none)", inner_width, 999))
            lines.append("")
            lines.append("Proposed Tasks:")

            for idx, item in enumerate(todo, start=1):
                prefix = f"{idx}. "
                indent = " " * len(prefix)

                # 关键：wrap 的宽度要扣掉 prefix，否则 prefix + 文本会再次超宽
                wrap_width = max(10, inner_width - len(prefix))
                wrapped = _wrap_lines(item, wrap_width, 999) or [""]

                lines.append(prefix + wrapped[0])
                for extra in wrapped[1:]:
                    lines.append(indent + extra)
            return lines

        def render() -> Layout:
            nonlocal plan_lines, scroll, scroll_max, page_size

            w, h = viewport()

            header_size = 3

            footer_lines = [
                "Enter feedback to revise, or type 'yes' to approve.",
                "Use PgUp/PgDn (or ↑/↓) to scroll.",
                f"> {input_buf}",
            ]
            if self.ui_debug:
                footer_lines.append(f"[debug] size={w}x{h} scroll={scroll}/{scroll_max} key={last_key}")

            footer_size = max(5, len(footer_lines) + 2)

            body_size = max(1, h - header_size - footer_size)
            body_inner = max(1, body_size - 2)

            page_size = body_inner

            content_width = max(20, w - 4)
            plan_lines = build_lines(content_width)

            scroll_max = max(0, len(plan_lines) - body_inner)
            scroll = min(max(scroll, 0), scroll_max)

            visible = plan_lines[scroll : scroll + body_inner]
            plan_text = "\n".join(visible)

            layout = Layout()
            layout.split(
                Layout(name="header", size=header_size),
                Layout(name="body", size=body_size),
                Layout(name="footer", size=footer_size),
            )

            run_id = self._state.run_id or "(pending)"
            model = self._state.model_name or ""
            header = f"PLAN REVIEW | run_id={run_id} | model={model}"
            layout["header"].update(Panel(Text(header), box=box.SQUARE))

            layout["body"].update(
                Panel(
                    Align.left(Text(plan_text)),
                    title="Plan · PgUp/PgDn",
                    title_align="right",
                    box=box.SQUARE,
                )
            )

            layout["footer"].update(Panel(Text("\n".join(footer_lines)), box=box.SQUARE))
            return layout

        try:
            with Live(
                render(),
                console=console,
                refresh_per_second=12,
                screen=True,
                transient=True,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="crop",
            ) as live:
                with _RawKeyReader() as reader:
                    while True:
                        key = reader.read_key()
                        if key is None:
                            continue
                        last_key = key

                        if key == "ENTER":
                            break
                        if key == "CTRL_C":
                            raise KeyboardInterrupt

                        if key == "BACKSPACE":
                            input_buf = input_buf[:-1]
                        elif key == "UP":
                            scroll -= 1
                        elif key == "DOWN":
                            scroll += 1
                        elif key == "PGUP":
                            scroll -= max(1, page_size)
                        elif key == "PGDN":
                            scroll += max(1, page_size)
                        elif key == "HOME":
                            scroll = 0
                        elif key == "END":
                            scroll = scroll_max
                        elif key == "ESC":
                            continue
                        else:
                            if len(key) == 1 and key.isprintable():
                                input_buf += key

                        live.update(render(), refresh=True)
        finally:
            self.resume()

        return input_buf.strip()

    def prompt_hitl_feedback(self, *, report_text: str, report_path: str) -> str:
        try:
            import os
            from rich.console import Console
            from rich.live import Live
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.text import Text
            from rich.align import Align
            from rich import box
        except Exception as exc:
            print(f"Rich UI unavailable: {exc}", file=sys.stderr)
            return ""

        self.pause()
        try:
            cols = os.get_terminal_size(sys.stdout.fileno()).columns
        except Exception:
            cols = None
        safe_cols = max(40, (cols - 2)) if cols else None
        console = Console(width=safe_cols, force_terminal=True)

        scroll = 0
        scroll_max = 0
        input_buf = ""
        report_lines: List[str] = []
        page_size = 10
        last_key = ""

        def viewport() -> tuple[int, int]:
            sz = console.size
            w, h = sz.width, sz.height
            w = max(40, int(w or 0))
            h = max(10, int(h or 0))
            if h > 200:
                h = 60
            return w, h

        def build_lines(inner_width: int) -> List[str]:
            inner_width = max(20, inner_width)
            lines: List[str] = []
            if report_path:
                lines.append(f"Report: {report_path}")
                lines.append("")
            raw_lines = report_text.splitlines() if report_text else ["(none)"]
            for chunk in raw_lines:
                if not chunk.strip():
                    lines.append("")
                    continue
                wrapped = textwrap.wrap(chunk, width=inner_width) or [""]
                lines.extend(wrapped)
            return lines

        def render() -> Layout:
            nonlocal report_lines, scroll, scroll_max, page_size

            w, h = viewport()

            header_size = 3

            footer_lines = [
                "Enter feedback to continue, or press Enter to keep paused.",
                "Use PgUp/PgDn (or ↑/↓) to scroll.",
                f"> {input_buf}",
            ]
            if self.ui_debug:
                footer_lines.append(f"[debug] size={w}x{h} scroll={scroll}/{scroll_max} key={last_key}")

            footer_size = max(5, len(footer_lines) + 2)
            body_size = max(1, h - header_size - footer_size)
            body_inner = max(1, body_size - 2)
            page_size = body_inner

            content_width = max(20, w - 4)
            report_lines = build_lines(content_width)
            scroll_max = max(0, len(report_lines) - body_inner)
            scroll = min(max(scroll, 0), scroll_max)

            visible = report_lines[scroll : scroll + body_inner]
            report_text_block = "\n".join(visible)

            layout = Layout()
            layout.split(
                Layout(name="header", size=header_size),
                Layout(name="body", size=body_size),
                Layout(name="footer", size=footer_size),
            )

            run_id = self._state.run_id or "(pending)"
            model = self._state.model_name or ""
            header = f"HITL REPORT | run_id={run_id} | model={model}"
            layout["header"].update(Panel(Text(header), box=box.SQUARE))

            layout["body"].update(
                Panel(
                    Align.left(Text(report_text_block)),
                    title="Report · PgUp/PgDn",
                    title_align="right",
                    box=box.SQUARE,
                )
            )

            layout["footer"].update(Panel(Text("\n".join(footer_lines)), box=box.SQUARE))
            return layout

        try:
            with Live(
                render(),
                console=console,
                refresh_per_second=12,
                screen=True,
                transient=True,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="crop",
            ) as live:
                with _RawKeyReader() as reader:
                    while True:
                        key = reader.read_key()
                        if key is None:
                            continue
                        last_key = key

                        if key == "ENTER":
                            break
                        if key == "CTRL_C":
                            raise KeyboardInterrupt

                        if key == "BACKSPACE":
                            input_buf = input_buf[:-1]
                        elif key == "UP":
                            scroll -= 1
                        elif key == "DOWN":
                            scroll += 1
                        elif key == "PGUP":
                            scroll -= max(1, page_size)
                        elif key == "PGDN":
                            scroll += max(1, page_size)
                        elif key == "HOME":
                            scroll = 0
                        elif key == "END":
                            scroll = scroll_max
                        elif key == "ESC":
                            continue
                        else:
                            if len(key) == 1 and key.isprintable():
                                input_buf += key

                        live.update(render(), refresh=True)
        finally:
            self.resume()

        return input_buf.strip()

    def show_final_summary(self, summary: str) -> None:
        try:
            import os
            from rich.console import Console
            from rich.live import Live
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.text import Text
            from rich.align import Align
            from rich.markdown import Markdown
            from rich.segment import Segment
            from rich import box
        except Exception as exc:
            print(f"Rich UI unavailable: {exc}", file=sys.stderr)
            return

        self.pause()

        try:
            cols = os.get_terminal_size(sys.stdout.fileno()).columns
        except Exception:
            cols = None
        safe_cols = max(40, (cols - 2)) if cols else None
        console = Console(width=safe_cols, force_terminal=True)

        scroll = 0
        scroll_max = 0
        page_size = 10
        last_key = ""

        summary_text = summary or "(none)"
        md = Markdown(summary_text, code_theme="monokai", justify="left")

        cached_width: Optional[int] = None
        cached_lines: Optional[List[List[Segment]]] = None

        def viewport() -> tuple[int, int]:
            sz = console.size
            w, h = sz.width, sz.height
            w = max(40, int(w or 0))
            h = max(10, int(h or 0))
            if h > 200:
                h = 60
            return w, h

        def render_markdown_lines(inner_width: int) -> List[List[Segment]]:
            """Render markdown to segment-lines at a given width; cache by width."""
            nonlocal cached_width, cached_lines
            inner_width = max(20, inner_width)

            if cached_lines is None or cached_width != inner_width:
                opts = console.options.update(width=inner_width)
                cached_lines = console.render_lines(md, options=opts, pad=False, new_lines=False)
                cached_width = inner_width

            return cached_lines or []

        def segments_to_text(segs: List[Segment]) -> Text:
            t = Text("", no_wrap=True, overflow="crop")
            for seg in segs:
                if seg.text:
                    t.append(seg.text, seg.style)
            t.rstrip()
            return t

        def render() -> Layout:
            nonlocal scroll, scroll_max, page_size

            w, h = viewport()
            header_size = 3

            footer_lines = [
                "Use PgUp/PgDn (or ↑/↓) to scroll.",
                "Press Enter to exit.",
            ]
            if self.ui_debug:
                footer_lines.append(f"[debug] size={w}x{h} scroll={scroll}/{scroll_max} key={last_key}")

            footer_size = max(4, len(footer_lines) + 2)

            body_size = max(1, h - header_size - footer_size)
            body_inner = max(1, body_size - 2)
            page_size = body_inner

            inner_width = max(20, w - 4)

            all_lines = render_markdown_lines(inner_width)
            scroll_max = max(0, len(all_lines) - body_inner)
            scroll = min(max(scroll, 0), scroll_max)

            visible = all_lines[scroll : scroll + body_inner]
            if visible:
                body_text = Text("\n").join([segments_to_text(line) for line in visible])
            else:
                body_text = Text("(none)", no_wrap=True)

            layout = Layout()
            layout.split(
                Layout(name="header", size=header_size),
                Layout(name="body", size=body_size),
                Layout(name="footer", size=footer_size),
            )

            run_id = self._state.run_id or "(pending)"
            model = self._state.model_name or ""
            header = f"FINAL SUMMARY | run_id={run_id} | model={model}"
            layout["header"].update(Panel(Text(header, no_wrap=True, overflow="crop"), box=box.SQUARE))

            layout["body"].update(
                Panel(
                    Align.left(body_text),
                    title="Summary · PgUp/PgDn",
                    title_align="right",
                    box=box.SQUARE,
                )
            )

            layout["footer"].update(Panel(Text("\n".join(footer_lines), no_wrap=True, overflow="crop"), box=box.SQUARE))
            return layout

        try:
            with Live(
                render(),
                console=console,
                screen=True,
                transient=True,
                auto_refresh=False,
                refresh_per_second=12,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="crop",
            ) as live:
                with _RawKeyReader() as reader:
                    while True:
                        key = reader.read_key()
                        if key is None:
                            continue
                        last_key = key

                        if key == "ENTER":
                            break
                        if key == "CTRL_C":
                            raise KeyboardInterrupt

                        if key == "UP":
                            scroll -= 1
                        elif key == "DOWN":
                            scroll += 1
                        elif key == "PGUP":
                            scroll -= max(1, page_size)
                        elif key == "PGDN":
                            scroll += max(1, page_size)
                        elif key == "HOME":
                            scroll = 0
                        elif key == "END":
                            scroll = scroll_max

                        live.update(render(), refresh=True)
        finally:
            self.resume()


    def _spinner(self) -> str:
        frames = ["-", "\\", "|", "/"]
        idx = int(time.time() * 4) % len(frames)
        return frames[idx]

    @staticmethod
    def _fit_logo(logo: str, width: int) -> str:
        if not logo:
            return "CatMaster"
        lines = [line.rstrip("\n") for line in logo.splitlines() if line.strip()]
        if not lines:
            return "CatMaster"
        max_width = max(len(line) for line in lines)
        if width >= max_width:
            return "\n".join(lines)
        if width < 8:
            return "CatMaster"[:width]
        start = max(0, (max_width - width) // 2)
        cropped = [line[start:start + width] if len(line) > width else line for line in lines]
        return "\n".join(cropped)

    def _handle_event(self, event: UIEvent) -> None:
        payload = event.payload or {}
        self._state.event_feed.append(self._format_feed_line(event))
        if event.run_id:
            self._state.run_id = event.run_id
        if event.name == "RUN_INIT_DONE":
            self._state.run_dir = payload.get("run_dir", self._state.run_dir)
            self._state.model_name = payload.get("model_label") or payload.get("model_name", self._state.model_name)
            self._state.llm_log_path = payload.get("llm_log_path", self._state.llm_log_path)
            self._state.phase = "INIT"
        if event.name == "SPLASH_SHOW":
            self._state.splash_logo = payload.get("logo", "")
            self._state.splash_tagline = payload.get("tagline", "CatMaster")
            self._state.splash_subtitle = payload.get("subtitle", "Initializing...")
            if self._show_splash:
                self._state.phase = "INIT"
        if event.name == "SPLASH_HIDE":
            self._show_splash = False
        if event.name == "PLAN_START":
            self._state.phase = "PLAN"
            self._state.activity = "Planning"
        if event.name == "PLAN_CREATED":
            self._state.plan_todo = payload.get("todo", []) or []
            self._state.plan_description = payload.get("plan_description_snippet", "")
        if event.name == "PLAN_REVIEW_SHOW":
            self._state.plan_todo = payload.get("todo", []) or []
            self._state.plan_description = payload.get("plan_description_snippet", "")
            self._state.activity = "Plan review"
        if event.name == "PLAN_REVIEW_APPROVED":
            self._state.activity = "Plan approved"
        if event.name == "TASKS_COMPILED":
            tasks = payload.get("tasks", []) or []
            self._state.tasks = {t["task_id"]: {"goal": t.get("goal", ""), "status": t.get("status", "pending")} for t in tasks if isinstance(t, dict) and t.get("task_id")}
            self._state.task_order = [t["task_id"] for t in tasks if isinstance(t, dict) and t.get("task_id")]
            self._state.phase = "TASK"
            if self._state.start_time is None:
                self._state.start_time = event.ts
                self._state.event_feed.clear()
        if event.name == "TASK_START":
            self._state.phase = "TASK"
            self._state.activity = "Task running"
            self._state.current_task_id = event.task_id or ""
            self._state.current_task_goal = payload.get("goal", "")
            if event.task_id and event.task_id in self._state.tasks:
                self._state.tasks[event.task_id]["status"] = "running"
        if event.name == "TASK_STEP_START":
            self._state.activity = "Task step"
            self._state.current_instruction = payload.get("instruction_snippet", "")
        if event.name == "TASK_DECISION":
            action = payload.get("action", "")
            method = payload.get("method", "")
            params = payload.get("params_compact", "")
            self._state.last_decision = " | ".join([item for item in [action, method, params] if item])
        if event.name == "TASK_END":
            outcome = payload.get("outcome", "")
            self._state.last_summary = payload.get("summary_snippet", "")
            if event.task_id and event.task_id in self._state.tasks:
                self._state.tasks[event.task_id]["status"] = outcome or "done"
        if event.name == "TOOL_CALL_START":
            self._state.phase = "TOOL"
            self._state.activity = f"Tool: {payload.get('tool', '')}".strip()
            self._state.last_tool = {
                "tool": payload.get("tool", ""),
                "params": payload.get("params_compact", ""),
                "status": "running",
            }
        if event.name == "TOOL_CALL_END":
            self._state.phase = "TASK"
            self._state.last_tool = {
                "tool": payload.get("tool", ""),
                "status": payload.get("status", ""),
                "highlights": payload.get("highlights", ""),
            }
        if event.name == "TASK_SUMMARIZE_START":
            self._state.phase = "SUMMARY"
            self._state.activity = "Summarizing"
        if event.name == "TASK_SUMMARY":
            self._state.last_summary = payload.get("summary_snippet", "")
        if event.name == "WHITEBOARD_APPLY_OK":
            self._state.whiteboard_counts = payload.get("counts", {}) or {}
        if event.name == "FINAL_SUMMARY_START":
            self._state.phase = "FINAL"
            self._state.activity = "Final summary"
        if event.name == "FINAL_SUMMARY_DONE":
            self._state.phase = "FINAL"
            self._state.activity = "Final report ready"
        if event.name == "RUN_END":
            self._state.phase = "DONE"
            self._state.status = payload.get("status", "")
        if event.name == "LLM_CALL_START":
            self._state.activity = f"LLM: {payload.get('kind', '')}".strip()
        if event.name == "LLM_CALL_END":
            self._state.last_llm_kind = payload.get("kind", "")
            self._state.last_llm_elapsed_ms = payload.get("elapsed_ms")
            if self.ui_debug:
                self._state.last_llm_snippet = payload.get("raw_snippet", self._state.last_llm_snippet)

    def _format_feed_line(self, event: UIEvent) -> str:
        base = self._state.start_time or event.ts
        delta = event.ts - base
        prefix = f"+{delta:5.1f}s"
        summary = _summarize_event(event)
        return f"{prefix} {summary}".rstrip()

    def _render_tasks(self, Panel, Table, Text, box, *, max_rows: int = 6) -> Any:
        if self._state.tasks:
            table = Table(show_header=True, box=box.MINIMAL)
            table.add_column("ID", width=8, no_wrap=True)
            table.add_column("Status", width=12, no_wrap=True)
            table.add_column("Goal")

            order = self._state.task_order[:]
            running_id = next((tid for tid in order if self._state.tasks.get(tid, {}).get("status") == "running"), None)
            if running_id:
                run_idx = order.index(running_id)
                prev_id = order[run_idx - 1] if run_idx > 0 else None
                display_ids = []
                if prev_id:
                    display_ids.append(prev_id)
                display_ids.append(running_id)
                for tid in order[run_idx + 1:]:
                    if len(display_ids) >= max_rows:
                        break
                    display_ids.append(tid)
            else:
                first_pending = next((tid for tid in order if self._state.tasks.get(tid, {}).get("status") == "pending"), None)
                if first_pending:
                    idx = order.index(first_pending)
                    prev_id = order[idx - 1] if idx > 0 else None
                    display_ids = []
                    if prev_id:
                        display_ids.append(prev_id)
                    display_ids.append(first_pending)
                    for tid in order[idx + 1:]:
                        if len(display_ids) >= max_rows:
                            break
                        display_ids.append(tid)
                else:
                    display_ids = order[:max_rows]

            for task_id in display_ids:
                task = self._state.tasks.get(task_id, {})
                goal = _truncate(task.get("goal", ""), 60)
                status = task.get("status", "")
                table.add_row(task_id, status, goal)

            completed = sum(1 for t in self._state.tasks.values() if t.get("status") not in {"pending", "running"})
            total = len(self._state.tasks)
            title = f"Tasks · {completed}/{total} completed"
            return Panel(table, title=title, title_align="center", box=box.SQUARE)
        if self._state.plan_todo:
            lines = []
            max_rows = 8
            for idx, item in enumerate(self._state.plan_todo, start=1):
                if idx > max_rows:
                    lines.append(f"... +{len(self._state.plan_todo) - max_rows} more")
                    break
                lines.append(f"{idx}. {_truncate(item, 70)}")
            return Panel(Text("\n".join(lines)), title="Plan", box=box.SQUARE)
        return Panel(Text("(no tasks yet)"), title="Tasks", box=box.SQUARE)

    def _render_current(self, Panel, Text, console, box) -> Any:
        width = max(20, (console.size.width // 2) - 6)
        lines: List[str] = []
        lines.extend(self._format_section("Task", self._state.current_task_id or "(none)", width, 1))
        lines.extend(self._format_section("Goal", self._state.current_task_goal, width, 3))
        lines.extend(self._format_section("Step", self._state.current_instruction, width, 2))
        lines.extend(self._format_section("Decision", self._state.last_decision, width, 2))
        lines.extend(self._format_section("Summary", self._state.last_summary, width, 2))
        return Panel(Text("\n".join(lines),no_wrap=True), title="Current", box=box.SQUARE)

    def _render_tool(self, Panel, Text, box) -> Any:
        tool = self._state.last_tool.get("tool", "(none)") if self._state.last_tool else "(none)"
        status = self._state.last_tool.get("status", "") if self._state.last_tool else ""
        highlights = self._state.last_tool.get("highlights", "") if self._state.last_tool else ""
        params = self._state.last_tool.get("params", "") if self._state.last_tool else ""
        lines = [f"Tool: {tool}"]
        if status:
            lines.append(f"Status: {status}")
        if params:
            lines.append(f"Params: {params}")
        if highlights:
            lines.append(f"Info: {_truncate(highlights, 120)}")
        return Panel(Text("\n".join(lines)), title="Tool", box=box.SQUARE)

    def _render_whiteboard(self, Panel, Text, box) -> Any:
        counts = self._state.whiteboard_counts or {}
        lines = ["Whiteboard"]
        if counts:
            for key in ["FACT", "FILE", "CONSTRAINT", "QUESTION", "DEPRECATE"]:
                if key in counts:
                    lines.append(f"{key}: {counts[key]}")
        else:
            lines.append("(no updates)")
        return Panel(Text("\n".join(lines)), title="Whiteboard", box=box.SQUARE)

    def _render_debug(self, Panel, Text, box) -> Any:
        lines = []
        if self._state.llm_log_path:
            lines.append(f"LLM log: {self._state.llm_log_path}")
        if self._state.last_llm_kind:
            lines.append(f"Last LLM: {self._state.last_llm_kind} ({self._state.last_llm_elapsed_ms} ms)")
        if self._state.last_llm_snippet:
            lines.append(f"Raw: {_truncate(self._state.last_llm_snippet, 200)}")
        if not lines:
            lines = ["(debug disabled)"]
        return Panel(Text("\n".join(lines)), title="Debug", box=box.SQUARE)

    @staticmethod
    def _format_section(label: str, text: str, width: int, max_lines: int) -> List[str]:
        label_prefix = f"{label}: "
        available = max(100, width - len(label_prefix) - 4)
        lines = _wrap_lines(text, available, max_lines) or ["(none)"]
        formatted = [label_prefix + lines[0]]
        indent = " " * len(label_prefix)
        for line in lines[1:]:
            formatted.append(indent + line)
        return formatted


def create_reporter(
    ui_mode: str,
    *,
    ui_debug: bool = False,
    show_splash: bool = True,
    is_tty: Optional[bool] = None,
) -> Reporter:
    if is_tty is None:
        is_tty = sys.stdout.isatty()
    mode = ui_mode
    if mode == "rich" and not is_tty:
        print("stdout is not a TTY; falling back to plain UI", file=sys.stderr)
        mode = "plain"
    if mode == "off":
        return NullReporter()
    if mode == "plain":
        return PlainConsoleReporter(ui_debug=ui_debug)
    if mode == "rich":
        try:
            import rich  # noqa: F401
        except Exception as exc:
            print(f"Rich UI unavailable ({exc}); falling back to plain", file=sys.stderr)
            return PlainConsoleReporter(ui_debug=ui_debug)
        return RichLiveReporter(ui_debug=ui_debug, show_splash=show_splash)
    raise ValueError(f"Unknown ui mode: {ui_mode}")


class _RawKeyReader:
    def __enter__(self) -> "_RawKeyReader":

        self._os = os
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)

        tty.setcbreak(self._fd)

        mode = termios.tcgetattr(self._fd)
        mode[3] = mode[3] & ~(termios.ISIG | termios.IEXTEN)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, mode)

        self._buf = b""
        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def _read1(self) -> Optional[int]:
        """Read exactly 1 byte from fd (buffer-first). Returns int 0-255 or None."""
        if self._buf:
            b0 = self._buf[0]
            self._buf = self._buf[1:]
            return b0
        data = self._os.read(self._fd, 1)
        if not data:
            return None
        return data[0]

    def _read_available(self, timeout: float) -> bytes:
        """Read whatever becomes available within timeout into a bytes chunk."""
        chunks = []
        # wait for at least one byte
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return b""
        # drain quickly
        while True:
            data = self._os.read(self._fd, 64)
            if not data:
                break
            chunks.append(data)
            r, _, _ = select.select([self._fd], [], [], 0)
            if not r:
                break
        return b"".join(chunks)

    def read_key(self) -> Optional[str]:
        import time

        b0 = self._read1()
        if b0 is None:
            return None

        # Common keys
        if b0 in (10, 13):  # \n \r
            return "ENTER"
        if b0 in (8, 127):  # backspace
            return "BACKSPACE"
        if b0 == 3:
            return "CTRL_C"

        # ESC sequence
        if b0 == 27:
            # Some terminals deliver ESC then the rest slightly later.
            tail = self._read_available(0.20)
            if not tail:
                return "ESC"

            # Put tail in a working buffer; we will parse and only "unread" if needed.
            seq = tail

            # If we didn't reach a terminator (alpha or '~'), wait a bit more to finish it.
            def is_terminator(x: int) -> bool:
                return x == 126 or (65 <= x <= 90) or (97 <= x <= 122)  # '~' or alpha

            deadline = time.time() + 0.15
            while seq and not is_terminator(seq[-1]) and time.time() < deadline:
                more = self._read_available(0.03)
                if not more:
                    break
                seq += more

            # Now normalize common CSI / SS3 sequences
            # Arrow keys: ESC [ ... A/B/C/D or ESC O A/B/C/D (and modifier variants)
            if seq.startswith(b"[") or seq.startswith(b"O"):
                last = seq[-1:]
                if last == b"A":
                    return "UP"
                if last == b"B":
                    return "DOWN"
                if last == b"C":
                    return "RIGHT"
                if last == b"D":
                    return "LEFT"

                # PgUp/PgDn: ESC [ 5 ~ / ESC [ 6 ~, possibly with modifiers: [5;2~
                if seq.startswith(b"[5"):
                    return "PGUP"
                if seq.startswith(b"[6"):
                    return "PGDN"

                # Home/End variants
                if seq.startswith(b"[H") or seq.startswith(b"OH") or seq.startswith(b"[1~") or seq.startswith(b"[7~"):
                    return "HOME"
                if seq.startswith(b"[F") or seq.startswith(b"OF") or seq.startswith(b"[4~") or seq.startswith(b"[8~"):
                    return "END"

            # If unrecognized, drop it (do NOT leak bytes into input_buf)
            return "ESC"

        # Printable / UTF-8 char (best-effort)
        if b0 < 0x80:
            return chr(b0)

        # Try decode UTF-8 multibyte (up to 4 bytes)
        buf = bytes([b0])
        for _ in range(3):
            try:
                return buf.decode(sys.stdin.encoding or "utf-8", errors="strict")
            except Exception:
                b1 = self._read1()
                if b1 is None:
                    break
                buf += bytes([b1])
        return buf.decode(sys.stdin.encoding or "utf-8", errors="ignore") or ""



__all__ = [
    "Reporter",
    "NullReporter",
    "PlainConsoleReporter",
    "RichLiveReporter",
    "create_reporter",
]
