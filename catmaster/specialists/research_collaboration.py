"""Persistent research discussion at native model boundaries, never a wakeup loop."""
import asyncio

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import NotRequired


class ResearchCollaborationState(AgentState):
    research_discussion_notice_seq: NotRequired[int]
    research_discussion_closeout_seq: NotRequired[int]
    research_discussion_closeout_run: NotRequired[str]


class ResearchCollaborationMiddleware(AgentMiddleware):
    state_schema = ResearchCollaborationState

    def __init__(self, *, discussions, graph_id, thread_id, publish_progress, run_id=""):
        self.discussions, self.graph_id, self.thread_id = discussions, graph_id, thread_id
        self.publish_progress, self.run_id = publish_progress, run_id

    def _owner(self):
        return self.discussions.persistent_owner(self.graph_id, self.thread_id)

    def system_guidance(self):
        """Stable guidance for this invocation's actual discussion authority."""
        owner = self._owner()
        if not owner:
            return ""
        common = (
            "Read relevant full discussion and replies when they can affect the assigned question. "
            "Assess them against the actual Methods, Results and user's objective. Discussion does not "
            "grant new execution authority or override a user pause; new user instructions can authorize "
            "further work after a completed stage. Read the collaboration skill reference before "
            "continuing or correcting a task or recording a discussion decision. "
        )
        if owner == self.thread_id:
            return common + (
                "Answer consequential questions from existing evidence or continue the appropriate owned "
                "investigation when needed and authorized. Check existing follow-ups to avoid duplicates. "
                "Record consequential decisions as replies, distinguishing an answer, accepted follow-up "
                "and justified deferral. Assignment is not a scientific finding; unanswered messages "
                "do not require work merely to clear discussion."
            )
        return common + (
            "Reply directly when useful and continue independent work without polling for replies "
            "or repeating finished work. Peer visibility does not grant control of another task."
        )

    async def abefore_model(self, state, runtime):
        owner = await asyncio.to_thread(self._owner)
        if not owner:
            return None
        # Publish only the researcher's successful explicit update, never infer
        # scientific intention from reasoning text or tool activity labels.
        returned = {}
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage):
                returned[message.tool_call_id] = message
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    result = returned.get(call["id"])
                    if call["name"] == "notify_progress" and result is not None and result.status != "error":
                        await self.publish_progress(call["id"], call["args"])
                break
        cursor = state.get("research_discussion_notice_seq", 0)
        root = owner == self.thread_id
        notice = await asyncio.to_thread(self.discussions.notice, self.graph_id, self.thread_id, cursor, all_graph=root)
        if not notice["count"]:
            return None
        end = notice["last_seq"]
        scope = "" if root else f"target_task_id = '{self.thread_id}' AND "
        return {"research_discussion_notice_seq": end, "messages": [HumanMessage(
            id=f"research-discussion-notice-{self.graph_id}-{self.thread_id}-{end}",
            content=(f"{notice['count']} new shared research discussion message(s). "
                     f"SELECT * FROM research_discussions WHERE {scope}seq > {int(cursor)} "
                     f"AND seq <= {int(end)} ORDER BY seq. Follow topic/reply links for earlier context as needed."),
            additional_kwargs={"catmaster_notification": "research_discussion", "catmaster_in_turn": True})]}

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state, runtime):
        last = state['messages'][-1]
        if not isinstance(last, AIMessage) or last.tool_calls or await asyncio.to_thread(self._owner) != self.thread_id:
            return None
        if state.get('research_discussion_closeout_run') == self.run_id:
            return None  # One opportunity per turn, not a mandatory reply/retry loop.
        cursor = state.get('research_discussion_closeout_seq', 0)
        notice = await asyncio.to_thread(self.discussions.notice, self.graph_id, self.thread_id, cursor, all_graph=True)
        if not notice['count']:
            return None
        end = notice['last_seq']
        # This cursor records a delivered check opportunity, never read/handled
        # status. Actual decisions remain explicit replies in shared discussion.
        return {'research_discussion_closeout_seq': end, 'research_discussion_closeout_run': self.run_id,
            'jump_to': 'model', 'messages': [HumanMessage(
                content=("Before closing this Persistent Research turn, check whether shared discussion leaves "
                    "a consequential question that needs a researcher's follow-up. Recent discussion is reachable with "
                    f"SELECT * FROM research_discussions WHERE seq > {int(cursor)} AND seq <= {int(end)} ORDER BY seq. "
                    "Follow topic/reply links to earlier context as needed."),
                additional_kwargs={'catmaster_notification': 'research_discussion_review', 'catmaster_in_turn': True})]}
