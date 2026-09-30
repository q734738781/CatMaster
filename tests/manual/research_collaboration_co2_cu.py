"""Opt-in real CO2/Cu peer collaboration, reusing existing literature evidence."""
import argparse
import asyncio
import shutil
from pathlib import Path

import parallel_research_co2_cu as harness
from catmaster.runtime.research_capacity import ResearchPoolConfig

PROMPT = """请基于已有 CO₂ 电还原 Cu 电极文献备忘录，继续提出值得验证的潜在新实验思路。
三份已完成的文献证据在 /evidence/active_sites.md、/evidence/interface_ions.md、/evidence/transport.md，
包含原始论文链接、条件和局限。优先复用；仅当影响实验取舍的实质问题无法从这些证据回答时，补充核查原始来源。
本轮重点比较三个独立问题：Cu 表面状态/重构的记忆效应、离子/界面环境的记忆效应、传质与润湿变化造成的假记忆。
请并行启动三个独立 ResearchSpecialist，各自负责完整的假说—方法比较—证据解释—下一实验建议，所有任务均为 low。
每位研究者先明确研究范围并发布简短当前进展，再查看同图在途任务。涉及共同对照、重复检索或可改变同伴判断的发现时，
使用共享讨论给相关研究者发一个有意义的定向问题/方法提醒；对收到的相关问题自行阅读并回答，不为测试反复闲聊或轮询。
各自保留独立上下文和判断，共享证据不等于照抄结论。每个分支写一份不超过800中文字的备忘录到独立目录，
记录必要的 methods/results/conclusion 到同一个 Research Graph，解释证据限定而不把提案当成已验证事实。
各分支不要再启动下一级后台研究，也不做额外完整文献综述或正式同行评审。你在三者完成后负责综合取舍。
最后写中文 co2_cu_parallel_report.md（约1200–1800字），包括最影响判断的已知进展、证据分歧，
2–3 个有对照、观测量、否证条件的潜在新实验，以及相比单一记忆解释为何值得优先区分。
只授权文献阅读、证据比较和轻量文件整理；不授权 MLFF、DFT、训练、远端作业或物理实验，不升级成本以启动这些工作。
综合报告与必要图谱记录完成后交付并停止；不开展下一研究阶段。
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path,
                        help="Directory containing active_sites.md, interface_ions.md and transport.md")
    args = parser.parse_args()
    destination = args.output / "workspace/files/evidence"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("active_sites.md", "interface_ions.md", "transport.md"):
        shutil.copyfile(args.evidence / name, destination / name)
    harness.PROMPT = PROMPT
    asyncio.run(harness.main(args.output, ResearchPoolConfig(3, {"low": 3, "medium": 1, "high": 1})))
