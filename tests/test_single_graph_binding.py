"""The host selects one graph; scientific tools operate inside that binding."""
import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from catmaster.research.knowledge_graph.models import GraphCreateRequest
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.misc import research_graph as graph_tools
from catmaster.tools.registry import ToolRegistry
from catmaster.webui.thread_support import ThreadServiceSupport


MUTATIONS = (
    'add_research_hypothesis', 'add_research_experiment', 'revise_research_claim',
    'record_research_result', 'set_research_result_judgment',
    'mark_research_experiment_failed', 'set_research_graph_completion',
)


def support_for(workspace, service):
    support = object.__new__(ThreadServiceSupport)
    support.workspace = workspace
    support.store = service.thread_store
    return support


@pytest.mark.parametrize('entrypoint', ['research', 'persistent_research', 'experiment', 'literature_review'])
def test_unbound_turn_reuses_newest_unarchived_graph_including_completed(tmp_path, entrypoint):
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    old = service.create_graph(GraphCreateRequest(question='Earlier open work'))['graph']
    latest = service.create_graph(GraphCreateRequest(question='Latest finished stage'))['graph']
    archived = service.create_graph(GraphCreateRequest(question='Archived work'))['graph']
    service.store.update_graph(archived['graph_id'], expected_revision=archived['revision'], changes={'archived': True})
    service.store.update_graph(latest['graph_id'], expected_revision=latest['revision'], changes={'completed': True})
    service.store.update_graph(old['graph_id'], expected_revision=old['revision'], changes={'title': 'Recently edited older graph'})
    thread = service.thread_store.create_thread(title='Continue work', entrypoint=entrypoint)
    support = support_for(tmp_path, service)
    bound = support._ensure_default_research_graph_binding(
        thread=thread, prompt='Reconsider the existing evidence', entrypoint=entrypoint, inherited=None)
    assert bound.active_research_graph_id == latest['graph_id']
    assert service.store.get_graph(latest['graph_id'])['completed']
    assert len(service.store.list_graphs(include_archived=True)) == 3

    # A deliberate existing selection is retained even when another graph is newer.
    service.bind_thread(thread.thread_id, graph_id=old['graph_id'])
    thread = service.thread_store.get_thread(thread.thread_id)
    rebound = support._ensure_default_research_graph_binding(
        thread=thread, prompt='Continue', entrypoint=entrypoint, inherited=None)
    assert rebound.active_research_graph_id == old['graph_id']


@pytest.mark.parametrize('entrypoint', ['research', 'persistent_research', 'experiment', 'literature_review'])
def test_first_graph_is_initialized_once_and_inherited_unbound_tasks_stay_unbound(tmp_path, entrypoint):
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    support = support_for(tmp_path, service)
    bindings = []
    for _ in range(2):
        thread = service.thread_store.create_thread(title='Study', entrypoint=entrypoint)
        bound = support._ensure_default_research_graph_binding(
            thread=thread, prompt='Study selectivity', entrypoint=entrypoint, inherited=None)
        bindings.append(bound.active_research_graph_id)
    assert bindings[0] == bindings[1]
    assert len(service.store.list_graphs()) == 1
    child = service.thread_store.create_thread(title='Isolated task', entrypoint=entrypoint)
    unchanged = support._ensure_default_research_graph_binding(
        thread=child, prompt='Read one report', entrypoint=entrypoint,
        inherited={'research_graph_id': ''})
    assert not unchanged.active_research_graph_id
    assert len(service.store.list_graphs()) == 1


def test_final_scientific_tool_schemas_and_mutations_use_only_accepted_graph(tmp_path):
    ensure_project_space_layout(tmp_path, create=True)
    service = ResearchGraphService(workspace=tmp_path)
    target = service.create_graph(GraphCreateRequest(question='Which route is supported?'))['graph']
    other = service.create_graph(GraphCreateRequest(question='Unrelated study'))['graph']
    thread = service.thread_store.create_thread(title='Research', entrypoint='research')
    service.bind_thread(thread.thread_id, graph_id=other['graph_id'])
    context = {'thread_id': thread.thread_id, 'research_graph_id': target['graph_id'], 'entrypoint': 'research'}
    registry = ToolRegistry()
    tools = {t.name: t for t in registry.as_langchain_tools(
        allowlist=list(MUTATIONS), workspace=str(tmp_path), runtime_context=context)}
    for schema in registry.as_openai_tools(allowlist=list(MUTATIONS)):
        assert 'graph_id' not in schema['parameters']['properties']
    for tool in tools.values():
        schema = convert_to_openai_tool(tool)['function']['parameters']
        assert 'graph_id' not in schema['properties']

    def invoke(name, **args):
        args['expected_revision'] = service.store.get_graph(target['graph_id'])['revision']
        result = tools[name].invoke({'type': 'tool_call', 'id': name, 'name': name, 'args': args})
        assert result.artifact['data']['graph_id'] == target['graph_id']
        return result.artifact['data']['changed']

    h1 = invoke('add_research_hypothesis', claim='Route A controls selectivity')['node']['node_id']
    h2 = invoke('add_research_hypothesis', claim='Route A controls selectivity at low temperature')['node']['node_id']
    invoke('revise_research_claim', new_node_id=h2, old_node_id=h1, action='qualify',
           scope='Low temperature only', rationale='High-temperature evidence is inconclusive')
    exp = invoke('add_research_experiment', objective='Compare routes',
                 plan_summary='Matched conditions', decision_rule='Lower measured barrier supports route A',
                 state='ready', tests_hypothesis_ids=[h2])['node']['node_id']
    result = invoke('record_research_result', summary='Route A has the lower measured barrier',
                    experiment_node_id=exp, refs=[{'ref_kind': 'url', 'ref_id': 'https://example.org/evidence'}])['node']['node_id']
    invoke('set_research_result_judgment', result_node_id=result, hypothesis_node_id=h2,
           relation='supports', scope='Measured temperature', rationale='Matched measured barriers')
    failed = invoke('add_research_experiment', objective='Measure high-temperature selectivity',
                    plan_summary='Matched measurement', decision_rule='Resolve the temperature dependence',
                    state='ready')['node']['node_id']
    invoke('mark_research_experiment_failed', experiment_node_id=failed,
           reason='Sample decomposed before the requested measurement')
    invoke('set_research_graph_completion', completed=False)
    assert service.store.get_snapshot(other['graph_id'])['nodes'] == []
    assert service.thread_store.get_thread(thread.thread_id).active_research_graph_id == other['graph_id']

    # Older callers cannot override a bound target, including an explicit empty binding.
    with workspace_scope(tmp_path), toolcall_context('legacy', context=context):
        for name in MUTATIONS:
            with pytest.raises(Exception, match='differs from this turn'):
                getattr(graph_tools, name)({'graph_id': other['graph_id']})
        with pytest.raises(Exception, match='revision'):
            graph_tools.add_research_hypothesis({'expected_revision': 1, 'claim': 'Stale write'})
    with workspace_scope(tmp_path), toolcall_context('unbound', context={**context, 'research_graph_id': ''}):
        with pytest.raises(Exception, match='not bound'):
            graph_tools.add_research_hypothesis({'graph_id': other['graph_id'], 'expected_revision': 1, 'claim': 'Invalid fallback'})
