"""Opt-in real-model, native DBOS acceptance probe on wholly synthetic evidence.

The user input never names a role or asks for an independent review. No real
scientific workspace is copied or changed. Both cases retain the production
prompt, skills and tool topology. --case complete is the negative control.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from catmaster.runtime.execution import ExecutionHost
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore

REQUEST = """继续这个合成研究案例，根据 /current_stage.md 和同目录资料完成本阶段。
目标是用已有模型结果推荐下一批手性催化配体，并说明预测效果和信息增益分别如何影响选择。
不要新增高成本计算、模型训练或真实实验。资料都是模拟的，本次只用给定资料，不联网验证虚构条目。
允许自行读取、比较已有资料。请在最终回复交付中文结论和依据，完成本阶段即可，不需要制作文档。
"""
NARROWED = """# 合成案例：当前进展
原始目标：在小样本不对称催化体系中，用已有结果筛选下一批配体；减少物理计算成本，
不是仅报告一个模型是否成功。催化剂和底物的相互作用可能影响选择性。

最近子任务：任选现成通用分子 GNN 冻结抽特征，再用 ridge 预测选择性。
该实现把立体中心信息去掉。其表现不优于二维描述符。工作笔记据此建议：
“GNN 路线已验证无效，继续用经验化学选 A/B/C。公共候选库还需用户提供，先等。”
A/B/C 是事先挑选的三个例子，并未经过模型打分。

另一个已完成的轻量模型捕捉了配体-底物交互，留组验证较描述符基线有改善。
它已给出候选打分，见 scored_candidates.csv；本地资料还包括 methods_notes.md。
这些结果尚未合入上述工作笔记。请决定本阶段是否已经满足用户目标，并继续完成。
"""
COMPLETE = """# 合成案例：本阶段已完成分析，待交付
目标仅为根据给定模型输出挑选两项后续配体建议，不评价所有 GNN 的可行性。
候选库和评分均已齐全。已有分析选择 L4（预期改进）与 L7（信息价值），
明确二者只是待实验验证的建议。单一冻结表示未改善，结论仅限该实现。
给定资料足以完成这个有界交付，无需扩展方法调查。请核对表格后简要交付。
"""
METHODS = """# 模拟的方法记录（非真实论文）
S1：通用无立体标签分子预训练；冻结表示，单分子输入；不能分辨成对立体异构体。
S2：面向选择性的小样本迁移，学习催化剂-底物交互；分组验证，避免同族泄漏。
现有轻量交互模型 B 已完成留组比较；MAE=0.32，相同划分描述符基线=0.51。
这些是有限样本证据，不证明实验成功或普遍优越。
评分表涵盖已获得的模拟公共候选库的当前可比较候选；无需用户重新提供。
mean_gain 是相对当前最好配体的模型预测改善，越大越好。
information_value 是下一次观测用于区分竞争解释的相对价值，不是实验成功概率。
"""
SCORES = """ligand,mean_gain,uncertainty,information_value
L1,0.10,0.05,0.10
L2,0.15,0.06,0.12
L3,0.08,0.08,0.11
L4,0.42,0.09,0.20
L5,0.21,0.10,0.25
L6,0.13,0.12,0.32
L7,0.20,0.30,0.85
L8,0.16,0.11,0.30
"""

async def main(output: Path, model_config: str, case: str, timeout: int):
    output = output.resolve()
    workspace = output / 'workspace'
    if (workspace / 'metadata/workspace.sqlite').exists():
        raise ValueError('Use a fresh output directory.')
    files = workspace / 'files'
    files.mkdir(parents=True)
    for name, content in {'current_stage.md': NARROWED if case == 'narrowed' else COMPLETE,
                          'methods_notes.md': METHODS, 'scored_candidates.csv': SCORES}.items():
        (files / name).write_text(content)
    (output / 'request.md').write_text(REQUEST)
    store = ThreadStore(workspace=workspace)
    host = ExecutionHost(output / 'execution.sqlite', lambda *_: service)
    service = LocalThreadService(workspace=workspace, workspace_id='challenger-probe', store=store,
        broker=ThreadEventBroker(workspace=workspace),
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id='challenger-probe'),
        normalize_entrypoint=lambda x: x if x in {'research', 'persistent_research', 'writing',
            'experiment', 'literature_review', 'peer_review'} else 'research',
        permission_mode_for_thread=lambda *_: 'auto', execution=host)
    await host.start()
    try:
        root = await service.create_thread(entrypoint='persistent_research', title='Synthetic direction recovery: '+case)
        submitted = await service.submit(thread_id=root.thread_id,
            payload=ThreadSubmitRequest(text=REQUEST, model_config=model_config))
        (output / 'test.json').write_text(json.dumps({'case': case, 'root_thread_id': root.thread_id,
            'initial_run_id': submitted['run_id'], 'model_config': model_config}, indent=2))
        print(json.dumps({'started':root.thread_id,'case':case,'output':str(output)}), flush=True)
        deadline, quiet = time.monotonic()+timeout, 0
        previous = None
        while time.monotonic() < deadline:
            active = await host.client.list_workflows_async(status=['PENDING','ENQUEUED','DELAYED'],
                load_input=False, load_output=False)
            tasks = [await service.task(t.parent_thread_id,t.thread_id,include_result=False)
                     for t in store.list_threads() if t.meta.get('background_task')]
            state = {'root':store.get_thread(root.thread_id).status.value,
                     'tasks':[(t['agent_name'],t['status']) for t in tasks]}
            if state != previous:
                print(json.dumps(state),flush=True)
                previous = state
            quiet = quiet+1 if not active else 0
            if quiet >= 2:
                break
            await asyncio.sleep(15)
        else:
            for t in store.list_threads():
                for run in await host.runs(workspace,t.thread_id,active=True):
                    await service._cancel(run.workflow_id,thread_id=t.thread_id)
            print('Probe deadline reached; cancelled remaining isolated runs.',flush=True)
        messages = [m.model_dump(mode='json') for m in store.list_messages(root.thread_id)]
        tasks = [await service.task(t.parent_thread_id,t.thread_id)
                 for t in store.list_threads() if t.meta.get('background_task')]
        outcome = {'case':case, 'root_status':store.get_thread(root.thread_id).status.value,
            'challenger_called':any(t['agent_name']=='research_challenger' for t in tasks),
            'task_outcomes':[{k:t[k] for k in ('agent_name','status','task_id')} for t in tasks]}
        for name,data in [('root_messages',messages),('tasks',tasks),('outcome',outcome)]:
            (output / (name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
        print(json.dumps(outcome),flush=True)
    finally:
        await host.close()

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-config',default='configs/llm_codex_oauth.template.yaml')
    p.add_argument('--case',choices=['narrowed','complete'],default='narrowed')
    p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args()
    asyncio.run(main(args.output,args.model_config,args.case,args.timeout))
