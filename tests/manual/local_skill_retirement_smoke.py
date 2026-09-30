"""Run real local writing and literature cases in isolated temporary workspaces.

Run from the repository root with the CatMaster interpreter and PYTHONPATH=.
Uses the configured models; no server startup, deployment or remote computation.
For a real revision case, select a writing case and supply --prompt-file plus
repeatable --input-file paths. Inputs are copied to input/<basename> in the
isolated workspace; the prompt should use those paths, not the original workspace.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from catmaster.llm.config import LLMProfile
from catmaster.specialists.runtime import build_specialist_runner


COMMUNICATION_EVIDENCE = """以下是专为软件测试编造的补锂研究记录，所有数值均为合成数据，
不是这些材料的真实预测或实验性能。比较同一测试定义下随截止电压变化的累计释锂容量：
截止电压_V, Li5FeO4_mAh_g, Li6CoO4_mAh_g, Li2NiO2_mAh_g
3.8,310,390,170
4.1,560,620,220
4.3,690,760,245
筛选问题起初围绕第一平台，现在调整为比较截至 4.3 V 的可释放容量。
Li2NiO2 是这里给定的传统对照；两种候选尚无本次测试定义下的实验验证。
记录另有独立宿主嵌锂搜索分支：加密插入点后，合成示例中的最低能结构没有改变，
耗时从 2 小时变为 20 小时，因此后续保留较疏的插入方案。
运行备忘还记有 21755 个候选、21702 次成功和三个脚本阶段，
没有提供额外的性能数据或可推断机制的结构证据。
"""


AUDIENCE_EVIDENCE = """以下均为软件测试编造的有机载体研究记录，不对应真实材料或实验。
有四种虚构载体P、Q、R、S，均可形成还原态；S还具有一个可与Li缔合的位点。
还原态载体用于向模拟受体H转移电子与锂。研究比较载体选择、处理结果及S的物种计量。
以P为共同参照，计算反应 X•− + P → X + P•−。各物种采用同一溶剂连续介质模型。
交换能/eV：P 0.000，Q −0.080，R −0.040，S +0.400。
同一处理定义下的模拟观测：H可取出的预存锂容量/mAh/g-H，P 40，Q 58，R 64，S 25。
S的模型物种：中性S(q=0)；LiS(q=0，一还原电子)；[Li2S]+(q=+1，一还原电子)；
Li2S(q=0，两还原电子)。Li配位数、物种总电荷和相对中性S的还原电子数分别记录。
反应 2LiS → S + Li2S，三种处理依次为旧几何单点、逐状态优化后的单点、高阶单点：
ΔE分别为+0.100、−0.020、−0.110 eV/所写反应。
自由阴离子对照 2S•− → S + S²−，相应ΔE为+1.100、+1.050、+0.940 eV。
上述ΔE仅包含电子能与连续介质贡献，无热熵校正，不是ΔG；没有显式溶剂、聚集或物种比例数据。
Q/R没有进一步验证结果。所有表中的已有计算均已完成，不要求重算。
未提供真实原子坐标；可以做解释性示意，不能制造实测结构或新的数据曲线。
"""


CASES = {
    "audience-report": ("writing", AUDIENCE_EVIDENCE + """
请用这些现有记录写一份中文科研进展报告，保存为 writing/report.md，正文约800至1200字。
保留四种载体及两类计算比较的覆盖，图文组织由你决定，直接完成交付。
不查文献、不开展新科学计算或科学核验；保留合成数据演示标识。"""),
    "audience-technical": ("writing", AUDIENCE_EVIDENCE + """
请给已经熟悉这项研究的组内计算人员写一份简洁内部技术摘要，保存为 writing/note.md。
读者熟悉反应定义与模型，供周会前快速定位结果差异；不需要背景教学或演讲式铺垫。
保留四种载体和两类计算比较的覆盖，突出定义、关键数值、差异和未决项；正文不超过350字，
表格不计字数。不查文献、不开展新科学计算或科学核验；保留合成数据演示标识。"""),
    "revision-slides": ("writing", COMMUNICATION_EVIDENCE + """
请根据这些合成记录，修订一份六页中文研究汇报样稿，面向没有参与计算的实验合作者。
旧稿开头直接罗列三个脚本阶段，用几块不同颜色的大数字展示候选数和成功数；
结果页标题是“数字如何进入下一阶段”，正文只有“重新匹配数据，统一口径后保留敏感端点”。
旧稿未展示给定材料的容量曲线，也没有交代独立搜索分支的结果。
我的反馈：不要用空洞的五彩数字撑页，但不是让你改成单色；缺少封面，
标题和正文很生硬。请说明研究对象和结果对材料选择有什么意义，
不要只写内部工作步骤，也不要把方法比较和重要的负结果删掉。
请直接完成可编辑PPTX及讲稿备注，保留制作源文件；渲染并查看实际页面，
预览放 tmp/。完整数据、条件与数值不变，并注明合成数据演示。
不查文献，不调用图片生成服务，不开展远程作业或新科学计算。"""),
    "progress-report": ("writing", COMMUNICATION_EVIDENCE + """
请面向实验合作者写一份简洁、图文并茂的中文研究进展报告，正文约 600 到 900 字，
讲清材料比较、研究目标的调整以及下一步讨论的问题。保存一份嵌入本地图件的 Markdown。
独立搜索分支只需说明其结果与取舍；不要补造结构、机理或实验验证。
所有图文都明确这是合成数据演示。请直接完成并检查图件，无需额外报告或确认。
不查文献，不调用图片生成服务，不开展远程作业或任何新科学计算。"""),
    "progress-slides": ("writing", COMMUNICATION_EVIDENCE + """
请面向实验合作者做一个四页中文进展汇报 PPTX，配简洁讲稿备注。
解释材料比较、筛选问题为何调整，以及独立搜索分支的结果和取舍。
使用可编辑正文和数据图，并标清合成数据演示。不要补造结构或机制。
直接完成并渲染查看实际页面，只交付 PPTX 和生成源文件，预览放临时目录。
不查文献，不调用图片生成服务，不开展远程作业或任何新科学计算。"""),
    "plot": ("writing", """Create one scientific data figure from these synthetic observations:
time_h, A_conversion_pct, B_conversion_pct
0,10,12
1,24,31
2,37,49
3,45,62
4,51,70
Clearly label the data as synthetic. Preserve every point; there are no replicates
or error estimates. Deliver one high-resolution PNG and its Python source.
Use the normal plotting worker and current local style requirements, inspect the
actual image and correct any visual defects. No literature search or computation
outside this isolated workspace is needed. Do not make a report or a slide deck."""),
    "slides": ("writing", """Make an editable English PPTX for a brief internal demo,
with exactly three slides: the question, synthetic evidence, and conclusion.
Topic: comparing two fictional catalyst-screening candidates. Synthetic data:
A: conversion 64%, selectivity 72%; B: conversion 51%, selectivity 91%.
The only conclusion is an activity-selectivity tradeoff; do not infer a mechanism.
Label the synthetic data and preserve all four values. Choose the layout yourself.
Use native editable slide elements where practical and concise speaker notes.
Use the existing writing capabilities and current local guidance. Render and
inspect every slide; fix actual clipping or unreadable labels. Only the PPTX and
its generation source are final deliverables; previews may stay in /tmp.
No image-generation API, literature search or remote scientific jobs are needed."""),
    "literature": ("literature_review", """核对 DOI 10.1038/nature14539 的题名、
年份和全部作者，给出实际访问过的来源链接。只做这一个文献身份查询，最终五句话以内。
不扩展为领域综述，不下载正文，不建 corpus，不从元数据推断研究结论。
使用当前可用的检索工具与本地文献指导。无需生成任何额外文档。"""),
    "citations": ("literature_review", """使用内置 OpenAlex 或 Semantic Scholar 元数据工具
核对 DOI 10.1038/nature14539 的题名、年份和全部作者，然后用内置引用定稿工具
按默认格式导出参考文献。只处理这一篇；不下载正文，不建 corpus，不扩展综述。
最终简要给出文献信息和输出文件链接，不要额外制作报告。"""),
}


def stage_inputs(workspace: Path, paths: list[Path]) -> None:
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Input files need distinct basenames in input/.")
    destination = workspace / "files" / "input"
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copyfile(path, destination / path.name)


async def run_case(
    name: str, profile: LLMProfile, *, prompt_file: Path | None = None,
    input_files: list[Path] | None = None,
) -> dict:
    workspace = Path(tempfile.mkdtemp(prefix=f"catmaster-skill-{name}-"))
    entrypoint, prompt = CASES[name]
    if prompt_file is not None:
        prompt = prompt_file.read_text(encoding="utf-8")
    if input_files:
        stage_inputs(workspace, input_files)
    built = build_specialist_runner(
        workspace=workspace, llm_profile=profile, reporter=None, run_control=None,
        project_id=f"skill-smoke-{name}", preferred_entrypoint=entrypoint,
    )
    print(json.dumps({"case": name, "workspace": str(workspace), "run_dir": str(built.run_context.run_dir)}, ensure_ascii=False), flush=True)
    result = await built.runner.arun(
        prompt, entrypoint=entrypoint, proposal_review=False, thread_id=f"smoke-{name}",
    )
    print(json.dumps({"case": name, "result": result}, ensure_ascii=False, default=str), flush=True)
    if result.get("status") != "done":
        raise RuntimeError(f"{name} did not finish: {result.get('status')}")
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=(*CASES, "all"), default="all", nargs="?")
    parser.add_argument("--config")
    parser.add_argument("--prompt-file", type=Path, help="Replace the selected case's request.")
    parser.add_argument("--input-file", type=Path, action="append", default=[],
                        help="Copy a file to isolated input/<basename>; may be repeated.")
    args = parser.parse_args()
    if args.case == "all" and (args.prompt_file or args.input_file):
        parser.error("Select one case when supplying a prompt or input files.")
    profile = LLMProfile.from_env_or_file(args.config)
    names = list(CASES) if args.case == "all" else [args.case]
    await asyncio.gather(*(run_case(
        name, profile, prompt_file=args.prompt_file, input_files=args.input_file,
    ) for name in names))


if __name__ == "__main__":
    asyncio.run(main())
