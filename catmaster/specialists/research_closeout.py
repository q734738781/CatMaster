"""Research stopping semantics on native DeepAgents middleware and task routing.

DBOS owns execution; LangGraph owns checkpoints. The workspace graph retains only the
scientific decision and review; this module neither polls nor launches runs.
"""
from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import NotRequired

from catmaster.research.knowledge_graph.store import ResearchGraphStore


class ResearchCloseoutState(AgentState):
    research_closeout_run: NotRequired[str]
    research_closeout_reminder: NotRequired[str]


class ResearchCloseoutMiddleware(AgentMiddleware):
    """Enforce the evidence obligations of a declared scientific stopping decision."""

    state_schema = ResearchCloseoutState

    def __init__(self, *, store: ResearchGraphStore, graph_id: str,
                 thread_id: str, run_id: str, persistent: bool, pending_tasks=None) -> None:
        self.store = store
        self.graph_id = graph_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.persistent = persistent
        self.pending_tasks = pending_tasks

    def before_agent(self, state: ResearchCloseoutState, runtime: Any) -> dict[str, Any] | None:
        if state.get('research_closeout_run') != self.run_id:
            return {'research_closeout_run': self.run_id, 'research_closeout_reminder': ''}
        return None

    async def abefore_agent(self, state: ResearchCloseoutState, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @staticmethod
    def _remind(state: ResearchCloseoutState, key: str, message: str) -> dict[str, Any]:
        if state.get('research_closeout_reminder') == key:
            # A missing scientific decision is an incomplete native run, never
            # fabricated success. Native checkpoint continuation remains available.
            raise RuntimeError('Research closeout is incomplete: ' + message)
        return {'messages': [HumanMessage(content=message, additional_kwargs={'catmaster_in_turn': True})], 'jump_to': 'model',
                'research_closeout_reminder': key}

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: ResearchCloseoutState, runtime: Any) -> dict[str, Any] | None:
        last = state['messages'][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        graph = self.store.get_graph(self.graph_id)
        if graph['completed'] or graph['archived']:
            return None
        if any(task.get('status') in {'pending', 'created', 'running', 'cancel_requested'}
               for task in (state.get('async_tasks') or {}).values() if isinstance(task, dict)):
            return None  # Native child completion will continue the same root.
        if not self.persistent:
            return None
        decisions = self.store.list_decisions(self.graph_id, thread_id=self.thread_id, run_id=self.run_id)
        if not decisions:
            # A conversational answer, source handoff or status update does not
            # declare campaign termination. The root owns that semantic choice;
            # do not make every final message pass through a disposition stage.
            return None
        decision = max(decisions, key=lambda row: row['updated_at'])
        body, review = decision['body'], decision['review']
        if body['disposition'] in {'waiting', 'boundary', 'parked'}:
            return None
        if body['disposition'] == 'continue':
            return self._remind(state, 'continue:' + decision['decision_id'],
                'The recorded research decision is to continue. Carry out the authorized '
                'next action, or record the concrete new boundary before closing.')
        decision_id = decision['decision_id']
        if not review:
            brief = (
                'This declared scientific stall needs one independent assessment before '
                'parking. Obtain it through the available independent reasoning capability, '
                'reusing an existing assessment on unchanged evidence. Supply the unresolved '
                'question, decisive sources, counterevidence and actual authorization below. '
                'The assessment should identify any missed authorized check, preserve valid '
                'partial results and open premises, and persist its review on this decision. '
                'You retain responsibility for acting on its findings within existing authority.\n'
                + json.dumps({'decision_id': decision_id, 'question': graph['question'],
                    'completion_criterion': graph['completion_criterion'],
                    'problem': body['problem'], 'stopping_reason': body['reason'],
                    'authorized_scope': body['authorized_scope'],
                    'basis_node_ids': body['basis_node_ids']}, ensure_ascii=False)
            )
            # Native middleware returns a HumanMessage and jumps to the model.
            # The model chooses the delegation; the host never fabricates its
            # AI tool call or selects a scientific task on its behalf.
            return self._remind(state, 'review:' + decision_id, brief)
        remedy = review.get('remedy_experiment_id')
        if remedy and not body['validation_result_ids'] and not body['exception']:
            message = (f'Independent review recommended Experiment {remedy}. Execute this one '
                'bounded authorized validation through the existing specialist capability. '
                'Then link its actual Result to this decision. A concrete authorization '
                'conflict, unavailable input/resource, moot check, equivalent completed '
                'check, or achieved user goal can justify an exception; conservatism or '
                'fear of an inconclusive outcome cannot. Preserve the original stopping scope.')
        else:
            message = ('The one-time reconsideration is complete. Record the final disposition '
                'on the same decision: park with open premises/resumption conditions if no '
                'new evidence changes the situation, continue if it does, or mark the '
                'requested stage complete. Do not repeat this review on unchanged evidence.')
        return self._remind(state, 'validation:' + decision_id, message)

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: ResearchCloseoutState, runtime: Any) -> dict[str, Any] | None:
        if self.pending_tasks and await self.pending_tasks():
            return None
        return self.after_model(state, runtime)
