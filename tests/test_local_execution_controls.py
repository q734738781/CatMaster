import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from catmaster.runtime.execution import ExecutionHost
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_store import ThreadStore
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_models import ThreadSubmitRequest, ThreadStopRequest, ThreadCheckpointContinueRequest, ThreadResumeRequest

class State(TypedDict):
    messages: Annotated[list, add_messages]
    evidence: str

@pytest.mark.parametrize('control', ['interrupt', 'rollback', 'stop_rollback', 'continue', 'hitl'])
def test_native_controls_preserve_the_correct_checkpoint(tmp_path, control):
    async def scenario():
        workspace=tmp_path/'workspace';(workspace/'files').mkdir(parents=True)
        store=ThreadStore(workspace=workspace);started=asyncio.Event();released=asyncio.Event()
        writes=[];running=0;peak=0;fail=True
        @asynccontextmanager
        async def factory(service, packet):
            async def evidence(state):
                writes.append('saved')
                return {'evidence':'accepted evidence'}
            async def answer(state):
                nonlocal running,peak,fail
                current=state['messages'][-1].content
                if str(current).endswith('initial'):
                    started.set()
                    if control=='continue' and fail:
                        fail=False
                        raise RuntimeError('recoverable tool failure')
                    if control=='hitl':
                        interrupt({'action_requests':[{'name':'approve_action','args':{},'description':'Approve'}],
                            'review_configs':[{'action_name':'approve_action','allowed_decisions':['approve','reject']}]})
                    elif control!='continue':
                        running+=1;peak=max(peak,running)
                        try: await released.wait()
                        finally: running-=1
                return {'messages':[AIMessage(content='Result: '+str(current))]}
            async with AsyncSqliteSaver.from_conn_string(str(workspace/'metadata/deepagent_threads.sqlite')) as saver:
                g=StateGraph(State).add_node('evidence',evidence).add_node('answer',answer)
                g.add_edge(START,'evidence').add_edge('evidence','answer').add_edge('answer',END)
                yield g.compile(checkpointer=saver)
        service=None;host=ExecutionHost(tmp_path/'execution.sqlite',lambda *_:service)
        service=LocalThreadService(workspace=workspace,workspace_id='workspace',store=store,
            broker=ThreadEventBroker(workspace=workspace),artifact_registry=ArtifactRegistry(workspace=workspace,workspace_id='workspace'),
            normalize_entrypoint=lambda x:x or 'research',permission_mode_for_thread=lambda *_:'auto',
            execution=host,graph_factory=factory)
        async def finish(rid):
            handle=await host.client.retrieve_workflow_async(rid)
            return await asyncio.wait_for(handle.get_result(polling_interval_sec=.02),20)
        await host.start()
        try:
            root=await service.create_thread()
            first=await service.submit(thread_id=root.thread_id,payload=ThreadSubmitRequest(text='initial'))
            await asyncio.wait_for(started.wait(),15)
            if control=='continue':
                assert (await finish(first['run_id']))['status']=='error'
                answer=store.get_message(root.thread_id,first['assistant_message'].id)
                assert any(p.meta.get('checkpoint_resume_available') for p in answer.parts if p.type=='error')
                second=await service.continue_from_checkpoint(thread_id=root.thread_id,payload=ThreadCheckpointContinueRequest(message_id=answer.id))
            elif control=='hitl':
                assert (await finish(first['run_id']))['status']=='interrupted'
                second=await service.resume(thread_id=root.thread_id,payload=ThreadResumeRequest(decisions=[{'type':'approve'}]))
            else:
                with pytest.raises(HTTPException) as rejected:
                    await service.submit(thread_id=root.thread_id,payload=ThreadSubmitRequest(text='must reject',strategy='reject'))
                assert rejected.value.status_code==409
                if control=='stop_rollback':
                    await service.stop(thread_id=root.thread_id,payload=ThreadStopRequest(action='rollback'))
                    assert running==0
                    second=await service.submit(thread_id=root.thread_id,payload=ThreadSubmitRequest(text='replacement'))
                else:
                    second=await service.submit(thread_id=root.thread_id,payload=ThreadSubmitRequest(text='replacement',strategy=control))
                    assert running==0
            assert (await finish(second['run_id']))['status']=='success'
            async with factory(service,{}) as graph:
                state=await graph.aget_state({'configurable':{'thread_id':root.thread_id}})
            users=[m.content for m in state.values['messages'] if m.type=='human']
            assert len(users)==(2 if control=='interrupt' else 1)
            assert len(writes)==(1 if control in {'continue','hitl'} else 2)
            assert peak<=1
        finally:
            released.set();await host.close()
    asyncio.run(scenario())
