from __future__ import annotations

from typing import List

from .events import UIEvent


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

    def wants_graph_streaming(self) -> bool:
        return False

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def prompt_proposal_feedback(self, *, todo: List[str], proposal_description: str) -> str:
        return ""

    def prompt_hitl_feedback(self, *, report_text: str, report_path: str) -> str:
        return ""

    def prompt_interrupt_feedback(
        self,
        *,
        guidance: str,
        run_id: str,
        phase: str,
    ) -> str:
        return ""

    def show_final_summary(self, summary: str) -> None:
        return


class NullReporter(Reporter):
    def emit(self, event: UIEvent) -> None:
        return


__all__ = ["Reporter", "NullReporter"]
