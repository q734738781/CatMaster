import asyncio
from types import SimpleNamespace

from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.storage import connect_workspace_db
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_store import ThreadStore
from catmaster.webui.thread_models import ThreadSubmitRequest

class Queue:
    def __init__(self):self.accepted={}
    async def runs(self,*args,**kwargs):return []
    async def run(self,*args):return None
    async def enqueue(self,packet):self.accepted[packet['run_id']]=packet

def test_graph_results_do_not_bypass_task_completion_policy(tmp_path):
    async def scenario():
        workspace=tmp_path/'workspace';(workspace/'files').mkdir(parents=True)
        store=ThreadStore(workspace=workspace);queue=Queue()
        service=LocalThreadService(workspace=workspace,workspace_id='workspace',store=store,
            broker=ThreadEventBroker(workspace=workspace),artifact_registry=ArtifactRegistry(workspace=workspace,workspace_id='workspace'),
            normalize_entrypoint=lambda x:x or 'research',permission_mode_for_thread=lambda *_:'auto',execution=queue)
        root=await service.create_thread(entrypoint='persistent_research')
        first=await service.submit(thread_id=root.thread_id,payload=ThreadSubmitRequest(text='Interpret the existing results.',entrypoint='persistent_research'))
        root=store.get_thread(root.thread_id);gid=root.active_research_graph_id;graph=ResearchGraphStore(workspace)
        child=store.create_thread(parent_thread_id=root.thread_id,meta={'background_task':True,'on_completion':'notify'})
        nested=store.create_thread(parent_thread_id=child.thread_id)
        def event(thread_id):
            with connect_workspace_db(workspace) as connection:
                graph._write_event(connection,graph_id=gid,revision=1,change='result.recorded',thread_id=thread_id)
        event(child.thread_id);event(nested.thread_id)
        await service.reconcile_research_graph_updates(gid,root.thread_id)
        assert len(queue.accepted)==1
        event('external-experiment')
        with connect_workspace_db(workspace) as connection:
            graph._write_event(connection,graph_id=gid,revision=1,change='node.updated',
                thread_id='external-experiment',details={'previous_node':{'kind':'result'}})
        await service.reconcile_research_graph_updates(gid,root.thread_id)
        await service.reconcile_research_graph_updates(gid,root.thread_id)
        assert len(queue.accepted)==2
        packet=list(queue.accepted.values())[-1]
        assert 'node.updated' in store.get_message(root.thread_id,packet['input_message_id']).parts[0].text
        current=store.get_thread(root.thread_id)
        store.update_thread(root.thread_id,meta={**current.meta,'automation_paused':True})
        event('external-experiment')
        await service.reconcile_research_graph_updates(gid,root.thread_id)
        assert len(queue.accepted)==2
        # Explicit user continuation includes the current graph and advances its
        # event cursor, so the same old evidence is not delivered a second time.
        await service.submit(thread_id=root.thread_id,payload=ThreadSubmitRequest(text='Continue interpreting current evidence.'))
        await service.reconcile_research_graph_updates(gid,root.thread_id)
        assert len(queue.accepted)==3
        # Attaching the same graph to ordinary Research does not opt the thread
        # into Persistent evidence-triggered continuation.
        current=store.get_thread(root.thread_id)
        store.update_thread(root.thread_id,entrypoint='research',meta={**current.meta,'automation_paused':False})
        event('external-experiment')
        await service.reconcile_research_graph_updates(gid,root.thread_id)
        assert len(queue.accepted)==3
    asyncio.run(scenario())
