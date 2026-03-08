from __future__ import annotations

import asyncio

from catmaster.runtime.literature import LiteratureContextPack, LiteratureSubagent


class ResearchLiteratureRunner:
    def __init__(
        self,
        *,
        subagent: LiteratureSubagent | None = None,
        allow_deep_report: bool = False,
    ) -> None:
        self.subagent = subagent or LiteratureSubagent.create_default()
        self.allow_deep_report = bool(allow_deep_report)

    def run(self, payload) -> LiteratureContextPack:
        data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
        depth = str(data.get("depth") or "quick")
        if depth == "deep_report" and not self.allow_deep_report:
            raise ValueError("deep_report is disabled for this research request")
        return self.subagent.run(
            query=str(data.get("query") or "").strip(),
            requested_depth=depth,
            topic=data.get("topic"),
            seed_papers=list(data.get("seed_papers") or []),
            role="research_lead",
        )

    async def arun(self, payload) -> LiteratureContextPack:
        return await asyncio.to_thread(self.run, payload)


__all__ = ["ResearchLiteratureRunner"]
