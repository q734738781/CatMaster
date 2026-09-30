"""Real, opt-in literature investigation through the deployed code path.

Run in catmaster from the repo root. All outputs and SQLite stores are isolated
below --output. This makes real model/search calls, never submits calculations.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from catmaster.runtime.execution import ExecutionHost
from catmaster.runtime.research_capacity import ResearchPoolConfig
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore


PROMPT = """请调研 CO₂ 电还原 Cu 电极通向 C₂+ 产物的近期进展，提出有证据依据、尚值得检验的潜在新实验思路。
本次是并行研究测试：由独立 ResearchSpecialist 研究者并行承担至少三个能改变实验选择的不同科学问题，
每个研究者自行完成假说/方法比较、文献结果解释、可推翻的结论与下一实验建议；不是仅分工搜索再交给单个提案者。
可以从动态重构与活性位、局部反应环境与离子效应、传质/润湿与长期稳定性等方面判断真正独立的问题，具体由你决定。
研究近期原始论文和必要基线，准确区分条件、来源支持的事实与自己的新提案；不要凭标题确认全文结论。
每条路线记录 methods/results/conclusion 到绑定 Research Graph，保存有真实来源链接的分支备忘录到不同目录。
最后交付中文 Markdown 综合报告 co2_cu_parallel_report.md，比较路线证据和局限，提出有对照、观测量与否证条件的潜在实验。
请尊重验证成本：仅授权在线文献检索、阅读、证据比较和必要的轻量本地整理；不授权 MLFF、DFT、远端作业、模型训练或物理实验。
这是文献研究，不需要为占满并发池添加路线，也不需要为测试而升级成本档位。
实验建议和文献综合达到上述要求后交付即可，不制作论文/PDF/正式同行评审，不启动下一研究阶段。
"""


async def main(output, capacity_config=None):
    output = output.resolve()
    workspace = output / "workspace"
    (workspace / "files").mkdir(parents=True, exist_ok=True)
    store, broker = ThreadStore(workspace=workspace), ThreadEventBroker(workspace=workspace)
    host = ExecutionHost(output / "execution.sqlite", lambda *_: service,
        capacity_config=capacity_config or ResearchPoolConfig(3, {"low": 2, "medium": 1, "high": 1}))
    service = LocalThreadService(workspace=workspace, workspace_id="co2-cu-parallel-test", store=store, broker=broker,
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="co2-cu-parallel-test"),
        normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto", execution=host)
    await host.start()
    root = await service.create_thread(entrypoint="persistent_research", title="CO2 Cu parallel literature test")
    submitted = await service.submit(thread_id=root.thread_id,
        payload=ThreadSubmitRequest(text=PROMPT, entrypoint="persistent_research"))
    (output / "test.json").write_text(json.dumps({"root_thread_id": root.thread_id,
        "initial_run_id": submitted["run_id"], "workspace": str(workspace)}, indent=2))
    print(json.dumps({"started": root.thread_id, "output": str(output)}), flush=True)
    previous, interaction, quiet = None, False, 0
    try:
        with (output / "activity.jsonl").open("a") as trace:
            while True:
                threads = store.list_threads()
                tasks = [await service.task(t.parent_thread_id, t.thread_id, include_result=False)
                         for t in sorted(threads, key=lambda t: t.thread_id) if t.meta.get("background_task")]
                pool = host.capacity.snapshot()
                assert pool["active"] <= pool["agent_pool_size"], pool
                assert all(t["active"] <= t["limit"] for t in pool["tiers"].values()), pool
                state = {"pool": pool, "tasks": [{k: t[k] for k in ("task_id", "agent_name", "status", "task_cost", "capacity_state")} for t in tasks]}
                encoded = json.dumps(state, sort_keys=True)
                if encoded != previous:
                    trace.write(json.dumps({"time": time.time(), **state}) + "\n")
                    trace.flush()
                    print(encoded, flush=True)
                    previous = encoded
                if pool["active"] >= 2 and not interaction:
                    turn = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(
                        text="这些后台研究请继续。这里仅简短确认已启动即可，不新增任务，不改变原定范围，完成后仍按原要求综合交付。",
                        entrypoint="persistent_research", strategy="enqueue"))
                    print(json.dumps({"foreground_interaction": turn["run_id"]}), flush=True)
                    interaction = True
                active = await host.client.list_workflows_async(status=["PENDING", "ENQUEUED", "DELAYED"], load_input=False, load_output=False)
                quiet = quiet + 1 if not active else 0
                if quiet >= 3:
                    break
                await asyncio.sleep(2)
        full = [await service.task(t.parent_thread_id, t.thread_id) for t in store.list_threads() if t.meta.get("background_task")]
        (output / "tasks.json").write_text(json.dumps(full, ensure_ascii=False, indent=2))
        messages = store.list_messages(root.thread_id)
        (output / "root_messages.json").write_text(json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False, indent=2))
        assert len(full) >= 3, "The requested independent research branches were not created."
        assert all(t["status"] == "success" for t in full), [(t["task_id"], t["status"]) for t in full]
        assert (workspace / "files/co2_cu_parallel_report.md").is_file(), "The requested synthesis was not delivered."
        print(json.dumps({"finished": True, "tasks": len(full), "foreground_interaction": interaction}), flush=True)
    finally:
        await host.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output))
