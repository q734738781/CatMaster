"""Opt-in real literature test: one Persistent request, then autonomous completion.

Run with the catmaster interpreter from the repository root. Model, literature
and search calls are real; workspace and DBOS stores stay under --output.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.runtime.execution import ExecutionHost
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore


PROMPT = """请调研 CO₂ 电还原铜电极通向 C₂+ 产物的近期原始研究，向合作实验室推荐值得优先做的外部实验。
请自主组织研究，重点找能改变实验取舍的证据、相互竞争的解释和方法局限。比较动态重构、局部反应环境与传质等因素时，
注意电解槽、工作条件和检测方法的可比性，区分来源支持的事实与待验证的新提案。
最终交付中文报告 co2_cu_external_experiments.md，说明关键进展和分歧，推荐 2–3 个优先实验，
写清科学依据、对照、观测量和什么结果会推翻预期，并提供可核查的原始论文链接。实验方案应能交给实验室进一步实施。
本次工作是文献调研和实验推荐，仅需检索、阅读、比较证据与必要的轻量本地整理；不执行新计算或物理实验。
报告和必要的科学记录完成后交付即可。
"""


async def main(output: Path, model_config: str):
    output = output.resolve()
    workspace = output / "workspace"
    if (workspace / "metadata/workspace.sqlite").exists():
        raise ValueError("Use a fresh output directory for a new acceptance run.")
    (workspace / "files").mkdir(parents=True, exist_ok=True)
    (output / "request.md").write_text(PROMPT)
    store = ThreadStore(workspace=workspace)
    host = ExecutionHost(output / "execution.sqlite", lambda *_: service)
    service = LocalThreadService(workspace=workspace, workspace_id="co2-cu-persistent-test", store=store,
        broker=ThreadEventBroker(workspace=workspace),
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="co2-cu-persistent-test"),
        normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto", execution=host)
    await host.start()
    try:
        root = await service.create_thread(entrypoint="persistent_research", title="CO2 Cu external experiment recommendations")
        submitted = await service.submit(thread_id=root.thread_id,
            payload=ThreadSubmitRequest(text=PROMPT, model_config=model_config))
        (output / "test.json").write_text(json.dumps({"root_thread_id": root.thread_id,
            "initial_run_id": submitted["run_id"], "workspace": str(workspace)}, indent=2))
        print(json.dumps({"started": root.thread_id, "output": str(output)}), flush=True)
        previous, quiet = "", 0
        with (output / "activity.jsonl").open("a") as trace:
            while True:
                threads = store.list_threads()
                tasks = [await service.task(t.parent_thread_id, t.thread_id, include_result=False)
                         for t in threads if t.meta.get("background_task")]
                root_now = store.get_thread(root.thread_id)
                state = {"root_status": root_now.status.value,
                    "tasks": [{k: task[k] for k in ("task_id", "agent_name", "status", "task_cost")}
                              for task in sorted(tasks, key=lambda task: task["task_id"])]}
                encoded = json.dumps(state, sort_keys=True)
                if encoded != previous:
                    trace.write(json.dumps({"time": time.time(), **state}) + "\n")
                    trace.flush()
                    print(encoded, flush=True)
                    previous = encoded
                active = await host.client.list_workflows_async(
                    status=["PENDING", "ENQUEUED", "DELAYED"], load_input=False, load_output=False)
                quiet = quiet + 1 if not active else 0
                if quiet >= 3:
                    break
                await asyncio.sleep(2)
        graph_service = ResearchGraphService(workspace=workspace)
        gid = store.get_thread(root.thread_id).active_research_graph_id
        snapshot = graph_service.store.get_snapshot(gid)
        (output / "graph.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
        (output / "root_messages.json").write_text(json.dumps(
            [message.model_dump(mode="json") for message in store.list_messages(root.thread_id)],
            ensure_ascii=False, indent=2))
        tasks = [await service.task(t.parent_thread_id, t.thread_id)
                 for t in store.list_threads() if t.meta.get("background_task")]
        (output / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
        report = workspace / "files/co2_cu_external_experiments.md"
        outcome = {"report_delivered": report.is_file(), "graph_completed": snapshot["graph"]["completed"],
            "research_branches": sum(task["agent_name"] == "research_specialist" for task in tasks),
            "results": sum(node["kind"] == "result" for node in snapshot["nodes"]),
            "external_proposals": sum(node["kind"] == "experiment" and node["body"].get("execution_lane") == "external"
                                      for node in snapshot["nodes"]),
            "task_outcomes": [{"agent": task["agent_name"], "status": task["status"]} for task in tasks],
            "root_status": store.get_thread(root.thread_id).status.value}
        (output / "outcome.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2))
        print(json.dumps(outcome), flush=True)
        assert report.is_file(), "Persistent Research stopped without the requested report."
        assert store.get_thread(root.thread_id).status.value == "idle", outcome
        # A failed delegate may be recovered by the root through another route.
        # Preserve every task outcome without treating that valid recovery as a
        # failed scientific delivery.
        assert all(task["status"] not in {"pending", "running"} for task in tasks), outcome
    finally:
        await host.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", default="")
    args = parser.parse_args()
    asyncio.run(main(args.output, args.model_config))
