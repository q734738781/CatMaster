"""Keep cost switches at native tool checkpoints, before any new execution."""
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage


class ResearchCapacityBoundary(AgentMiddleware):
    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        calls = last.tool_calls
        if len(calls) > 1 and any(c["name"] == "set_research_task_cost" for c in calls):
            return {"jump_to": "model", "messages": [ToolMessage(
                content="No tools in this batch ran. Request a task cost change alone, then wait for admission before executing other work.",
                tool_call_id=call["id"], name=call["name"], status="error") for call in calls]}
        return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)
