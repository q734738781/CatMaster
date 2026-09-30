from __future__ import annotations

import asyncio
import json

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from catmaster.research.knowledge_graph.models import GraphCreateRequest, ResultCreateRequest, ResultJudgmentSetRequest
from catmaster.research.knowledge_graph.query import ResearchGraphSQLQuery
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.specialists.research_closeout import ResearchCloseoutMiddleware


def setup_research(tmp_path):
    service = ResearchGraphService(workspace=tmp_path)
    graph = service.create_graph(GraphCreateRequest(question='Literature to Cu/CO2 experiment recommendations', orchestration_mode='auto'))['graph']
    hook = ResearchCloseoutMiddleware(store=service.store, graph_id=graph['graph_id'],
        thread_id='root_thread', run_id='native_run', persistent=True)
    return service, graph['graph_id'], hook


def stall(service, graph_id, **changes):
    return service.store.record_disposition(graph_id, thread_id='root_thread', run_id='native_run', body={
        'disposition': 'stalled', 'problem': 'Cannot distinguish transport from intrinsic selectivity',
        'reason': 'The current sources use incompatible conditions',
        'authorized_scope': 'Literature and experimental recommendations only; no calculations or laboratory execution',
        **changes,
    })


def final_state(**changes):
    return {'messages': [AIMessage(content='Final answer')], **changes}


def test_completed_stage_and_actual_boundary_skip_reconsideration(tmp_path):
    service, gid, hook = setup_research(tmp_path)
    stall(service, gid, disposition='boundary', reason='User requested pause')
    assert hook.after_model(final_state(), None) is None
    graph = service.store.get_graph(gid)
    service.store.update_graph(gid, expected_revision=graph['revision'], changes={'completed': True})
    assert hook.after_model(final_state(), None) is None
    assert not service.store.list_decisions(gid)[0]['review']


def test_ordinary_reply_does_not_require_a_campaign_disposition(tmp_path):
    service, gid, hook = setup_research(tmp_path)
    state = final_state()
    assert hook.after_model(state, None) is None
    assert service.store.get_graph(gid)['completed'] is False
    assert service.store.list_decisions(gid) == []


def test_declared_stall_requests_review_without_fabricating_model_actions(tmp_path):
    service, gid, hook = setup_research(tmp_path)
    decision = stall(service, gid)
    update = hook.after_model(final_state(), None)
    assert update['jump_to'] == 'model'
    assert all(isinstance(message, HumanMessage) for message in update['messages'])
    assert decision['decision_id'] in update['messages'][0].content
    graph = service.store.get_graph(gid)
    service.store.update_graph(gid, expected_revision=graph['revision'], changes={'title': 'A new title'})
    with pytest.raises(RuntimeError, match='incomplete'):
        hook.after_model(final_state(research_closeout_reminder=update['research_closeout_reminder']), None)


def test_recommended_validation_requires_actual_outcome_or_concrete_exception(tmp_path):
    service, gid, hook = setup_research(tmp_path)
    decision = stall(service, gid)
    experiment, _ = service.store.add_node_bundle(gid, expected_revision=service.store.get_graph(gid)['revision'],
        kind='experiment', title='Check electrolyte in the original paper', state='ready',
        body={'objective': 'Resolve the electrolyte condition', 'plan_summary': 'Read Methods and device-specific results',
              'decision_rule': 'Correct the recommendation if the device conditions differ', 'execution_lane': 'literature_review'})
    service.store.record_review(gid, review={'decision_id': decision['decision_id'],
        'assessment': 'Device-specific Methods can resolve the mismatch', 'remedy_experiment_id': experiment['node_id'],
        'authorization_basis': 'Source verification within literature review', 'resume_when': 'New device-matched evidence'})
    reminder = hook.after_model(final_state(), None)
    assert reminder['jump_to'] == 'model'
    with pytest.raises(ValueError, match='validation once'):
        stall(service, gid, decision_id=decision['decision_id'], disposition='parked')
    result = service.record_result(gid, ResultCreateRequest(expected_revision=service.store.get_graph(gid)['revision'],
        experiment_node_id=experiment['node_id'], summary='Methods identifies the MEA electrolyte; recommendation corrected'))['node']
    stall(service, gid, decision_id=decision['decision_id'], disposition='parked',
          validation_result_ids=[result['node_id']], resume_when='New evidence resolving the remaining gap')
    assert hook.after_model(final_state(), None) is None
    assert len(service.store.list_decisions(gid)) == 1


def test_external_proposal_is_never_an_executable_remedy(tmp_path):
    service, gid, _ = setup_research(tmp_path)
    decision = stall(service, gid)
    experiment, _ = service.store.add_node_bundle(gid, expected_revision=service.store.get_graph(gid)['revision'],
        kind='experiment', title='Lab control', state='ready', body={'objective': 'Measure Cu selectivity',
            'plan_summary': 'Matched electrolyte control', 'decision_rule': 'Compare product changes', 'execution_lane': 'external'})
    with pytest.raises(ValueError, match='external proposals remain handoffs'):
        service.store.record_review(gid, review={'decision_id': decision['decision_id'], 'assessment': 'A lab control may help',
            'remedy_experiment_id': experiment['node_id'], 'authorization_basis': 'Recommendations only', 'resume_when': 'User supplies laboratory results'})
    assert not service.store.list_decisions(gid)[0]['review']


def test_scope_and_revision_remain_queryable_without_reopening_delivery(tmp_path):
    service, gid, _ = setup_research(tmp_path)
    old, _ = service.store.add_node_bundle(gid, expected_revision=service.store.get_graph(gid)['revision'],
        kind='hypothesis', title='Broad proxy claim', body={'claim': 'The proxy explains both ee and yield'})
    new, _ = service.store.add_node_bundle(gid, expected_revision=service.store.get_graph(gid)['revision'],
        kind='hypothesis', title='Scoped proxy claim', body={'claim': 'The proxy is useful for ee under the tested conditions'})
    result = service.record_result(gid, ResultCreateRequest(expected_revision=service.store.get_graph(gid)['revision'],
        summary='ee relation remains meaningful; yield discrimination failed', judgments=[{
            'hypothesis_node_id': old['node_id'], 'relation': 'opposes', 'scope': 'yield only', 'rationale': 'The proxy did not distinguish yield'}]))['node']
    service.store.update_graph(gid, expected_revision=service.store.get_graph(gid)['revision'], changes={'completed': True})
    service.store.revise_claim(gid, expected_revision=service.store.get_graph(gid)['revision'], new_node_id=new['node_id'],
        old_node_id=old['node_id'], action='qualify', scope='Retain ee; withdraw yield extrapolation', rationale='Different observables require different evidence')
    service.set_result_judgment(gid, result['node_id'], old['node_id'], ResultJudgmentSetRequest(
        expected_revision=service.store.get_graph(gid)['revision'], relation='opposes', scope='yield', rationale='No yield discrimination'))
    assert service.store.get_graph(gid)['completed'] is True
    query = ResearchGraphSQLQuery(tmp_path)
    rows = query.execute(graph_id=gid, sql="SELECT relation,scope,rationale,action FROM research_edges ORDER BY relation")
    assert any(row['action'] == 'qualify' for row in rows['rows'])
    assert service.store.get_node(gid, old['node_id'])['body']['claim'] == 'The proxy explains both ee and yield'
    with pytest.raises(ValueError, match='cycles'):
        service.store.revise_claim(gid, expected_revision=service.store.get_graph(gid)['revision'], new_node_id=old['node_id'],
            old_node_id=new['node_id'], action='replace', scope='same claim', rationale='invalid loop')


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_legacy_native_deepagent_routes_review_and_returns_to_same_root(tmp_path):
    service, gid, hook = setup_research(tmp_path)
    decision = stall(service, gid)
    calls = []

    @tool
    def save_review() -> str:
        """Persist the independent assessment."""
        calls.append('review')
        service.store.record_review(gid, review={'decision_id': decision['decision_id'],
            'assessment': 'No authorized source check resolves the remaining ambiguity',
            'resume_when': 'New matched-condition evidence'})
        return 'review saved'

    @tool
    def park() -> str:
        """Persist the final scientific disposition."""
        calls.append('park')
        stall(service, gid, decision_id=decision['decision_id'], disposition='parked', resume_when='New matched-condition evidence')
        return 'parked'

    root_model = ToolModel(responses=[AIMessage(content='Stop at the unresolved gap'),
        AIMessage(content='', tool_calls=[{'name': 'task', 'args': {
            'subagent_type': 'hypothesis_proposer', 'description': 'Assess the unresolved issue and save its review.'},
            'id': 'model_review', 'type': 'tool_call'}]),
        AIMessage(content='', tool_calls=[{'name': 'park', 'args': {}, 'id': 'park_call', 'type': 'tool_call'}]),
        AIMessage(content='Here are the partial findings and open premise')])
    reviewer_model = ToolModel(responses=[AIMessage(content='', tool_calls=[{'name': 'save_review', 'args': {}, 'id': 'save_call', 'type': 'tool_call'}]),
        AIMessage(content='Independent assessment recorded')])
    agent = create_deep_agent(model=root_model, tools=[park], middleware=[hook], subagents=[{
        'name': 'hypothesis_proposer', 'description': 'Independent scientific reasoner',
        'system_prompt': 'Assess the supplied evidence independently', 'model': reviewer_model, 'tools': [save_review]}])
    result = asyncio.run(agent.ainvoke({'messages': [HumanMessage(content='Complete the authorized literature study')]}))
    assert calls == ['review', 'park']
    assert result['messages'][-1].content == 'Here are the partial findings and open premise'
    task_results = [m for m in result['messages'] if getattr(m, 'name', '') == 'task']
    assert len(task_results) == 1
    review_calls = [call for message in result['messages'] if isinstance(message, AIMessage)
                    for call in message.tool_calls if call['name'] == 'task']
    assert [call['id'] for call in review_calls] == ['model_review']


def test_tick_does_not_create_planner_or_launch_ready_science(tmp_path):
    service, gid, _ = setup_research(tmp_path)
    service.store.add_node_bundle(gid, expected_revision=service.store.get_graph(gid)['revision'],
        kind='experiment', title='Ready but not dispatched', state='ready', body={'objective': 'A scientific check',
            'plan_summary': 'A bounded comparison', 'decision_rule': 'Discriminate alternatives'})
    asyncio.run(service.tick())
    assert service.store.get_snapshot(gid)['launches'] == []
    assert service.store.latest_planning_preview(gid, current_revision_only=False) is None
    assert service.thread_store.list_threads() == []


def test_async_challenger_can_record_independent_review(tmp_path, monkeypatch):
    from catmaster.tools.misc import research_graph as tools
    service, gid, _ = setup_research(tmp_path)
    decision = stall(service, gid)
    monkeypatch.setattr(tools, '_service', lambda: service)
    monkeypatch.setattr(tools, 'current_tool_context', lambda: {'research_graph_id': gid})
    monkeypatch.setattr(tools, 'current_tool_audience', lambda: 'research_challenger')
    tools.record_research_review({'decision_id': decision['decision_id'],
        'assessment': 'The current evidence cannot distinguish the competing explanations.',
        'resume_when': 'A new discriminating observation becomes available.'})
    saved = service.store.list_decisions(gid)[0]
    assert saved['review']['assessment'].startswith('The current evidence')
    assert not service.store.get_graph(gid)['completed']
